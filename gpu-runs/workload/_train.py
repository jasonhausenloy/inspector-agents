"""Shared training loop for pretrain.py and finetune.py.

Caller picks the mode ("pretrain" or "finetune"); this module owns DDP setup,
LR scheduling, checkpoint I/O, signal handling, and the --duration_s self-stop.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import TokenDataset
from model import GPT


def stable_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--max_iters", type=int, default=100_000)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--ckpt_out", type=str, default=None)
    ap.add_argument("--ckpt_in", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--phase_id", type=str, default="default")
    ap.add_argument("--duration_s", type=float, default=None,
                    help="self-stop after this many wall seconds")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    return ap.parse_args(argv)


def lr_at(it: int, base_lr: float, schedule: str, max_iters: int,
          warmup_iters: int, min_lr_ratio: float) -> float:
    if it < warmup_iters:
        return base_lr * (it + 1) / max(1, warmup_iters)
    if schedule == "constant":
        return base_lr
    # cosine
    if it >= max_iters:
        return base_lr * min_lr_ratio
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * min_lr_ratio + coeff * (base_lr - base_lr * min_lr_ratio)


STOP_CHECK_EVERY = 5  # iters between synchronized stop checks (cheap all-reduce)


def _maybe_init_wandb(mode: str, args: argparse.Namespace, cfg: dict) -> object | None:
    """Init wandb on rank 0 if WANDB_API_KEY is set. Returns the run or None."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    group = os.environ.get("WANDB_RUN_GROUP", "default")
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "verifier-challenge-traces"),
            name=f"{group}/{args.phase_id}",
            group=group,
            tags=[mode, args.phase_id],
            config={
                "mode": mode, "phase_id": args.phase_id, "lr": args.lr,
                "dataset": args.dataset,
                "n_layer": cfg.get("n_layer"), "n_head": cfg.get("n_head"),
                "n_embd": cfg.get("n_embd"), "block_size": cfg.get("block_size"),
                "batch_size": cfg.get("batch_size"),
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            },
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"wandb init failed: {e}; continuing without wandb", flush=True)
        return None


def run(mode: str) -> None:
    assert mode in ("pretrain", "finetune"), mode
    lr_schedule = "cosine" if mode == "pretrain" else "constant"
    freeze_embed = mode == "finetune"

    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # Same model-init seed on every rank/host so DDP starts from identical weights;
    # data sampling uses a rank-dependent rng so each rank sees different windows.
    base_seed = stable_seed(args.phase_id)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed % (2**31))

    cfg_globals: dict = {}
    exec(Path(args.config).read_text(), cfg_globals)
    n_layer = cfg_globals["n_layer"]
    n_head = cfg_globals["n_head"]
    n_embd = cfg_globals["n_embd"]
    block_size = cfg_globals["block_size"]
    batch_size = cfg_globals["batch_size"]

    wandb_run = _maybe_init_wandb(mode, args, cfg_globals) if rank == 0 else None

    model = GPT(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size).to(device)

    if args.ckpt_in:
        if rank == 0:
            print(f"loading checkpoint from {args.ckpt_in}", flush=True)
        ckpt = torch.load(args.ckpt_in, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

    if freeze_embed:
        for name, p in model.named_parameters():
            if name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
                p.requires_grad_(False)

    if world_size > 1:
        ddp_kwargs: dict = {"device_ids": [local_rank]}
        if freeze_embed:
            # Tied lm_head + frozen wte means DDP sees an unused param on backward.
            ddp_kwargs["find_unused_parameters"] = True
        model = DDP(model, **ddp_kwargs)

    # Per-rank data RNG
    data_rng = np.random.default_rng(base_seed + rank * 1_000_003)

    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    inner_model = model.module if isinstance(model, DDP) else model
    ds = TokenDataset(args.dataset, block_size)

    should_stop = {"flag": False}

    def _on_term(signum, frame):
        should_stop["flag"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    iter_num = 0

    def save_checkpoint() -> None:
        if rank != 0 or args.ckpt_out is None:
            return
        path = Path(args.ckpt_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"model": inner_model.state_dict(), "iter": iter_num}, tmp)
        tmp.replace(path)
        print(f"saved checkpoint to {path} at iter {iter_num}", flush=True)

    t_start = time.time()
    model.train()
    stop_t = torch.zeros(1, dtype=torch.int32, device=device) if world_size > 1 else None
    while iter_num < args.max_iters:
        # Synchronized stop check: every rank votes; if any rank wants to stop,
        # all stop together. Without this, ranks self-stop on time.time() at
        # different iters and the trailing rank deadlocks NCCL backward against
        # rank 0 sitting at dist.barrier().
        if iter_num % STOP_CHECK_EVERY == 0:
            local_stop = int(
                should_stop["flag"]
                or (args.duration_s is not None and (time.time() - t_start) >= args.duration_s)
            )
            if world_size > 1:
                stop_t.fill_(local_stop)
                dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
                global_stop = int(stop_t.item())
            else:
                global_stop = local_stop
            if global_stop:
                if rank == 0:
                    why = "signal" if should_stop["flag"] else f"--duration_s={args.duration_s}"
                    print(f"stopping ({why}) at iter {iter_num}", flush=True)
                break

        lr = lr_at(iter_num, args.lr, lr_schedule, args.max_iters,
                   args.warmup_iters, args.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = ds.get_batch(batch_size, device, rng=data_rng)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
        optimizer.step()

        if iter_num % 50 == 0 and rank == 0:
            elapsed = time.time() - t_start
            loss_v = loss.item()
            print(
                f"iter {iter_num} loss {loss_v:.4f} lr {lr:.2e} elapsed {elapsed:.0f}s",
                flush=True,
            )
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {"iter": iter_num, "loss": loss_v, "lr": lr, "elapsed_s": elapsed},
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"wandb.log failed: {e}", flush=True)
        if iter_num > 0 and iter_num % args.save_every == 0:
            save_checkpoint()
        iter_num += 1

    save_checkpoint()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
