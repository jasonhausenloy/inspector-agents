"""Inference workload — load checkpoint, generate forever, no backward, no
collectives. NCCL is initialized so the trace shows init traffic; no all-reduces.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT


def stable_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, help="ignored; kept for orchestrator parity")
    ap.add_argument("--ckpt_in", required=True)
    ap.add_argument("--phase_id", default="inference")
    ap.add_argument("--duration_s", type=float, default=None)
    ap.add_argument("--prompt", default="Once upon a time,")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    seed = stable_seed(args.phase_id) + rank
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31))

    cfg_globals: dict = {}
    exec(Path(args.config).read_text(), cfg_globals)

    model = GPT(
        n_layer=cfg_globals["n_layer"],
        n_head=cfg_globals["n_head"],
        n_embd=cfg_globals["n_embd"],
        block_size=cfg_globals["block_size"],
    ).to(device)
    ckpt = torch.load(args.ckpt_in, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=device)

    should_stop = {"flag": False}

    def _on_term(signum, frame):
        should_stop["flag"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    n = 0
    t_start = time.time()
    while not should_stop["flag"]:
        if args.duration_s is not None and (time.time() - t_start) >= args.duration_s:
            if rank == 0:
                print(f"hit --duration_s={args.duration_s}, stopping", flush=True)
            break
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model.generate(
                prompt_ids,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        n += 1
        if rank == 0 and n % 5 == 0:
            print(f"gen {n} runs, elapsed {time.time() - t_start:.0f}s", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
