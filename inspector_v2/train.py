"""Train TinyGPT on Tiny Shakespeare and emit two log streams.

End-to-end:
  1. Download Tiny Shakespeare if absent.
  2. Build char vocab; serialize.
  3. Train for `--steps 2000` on M3 MPS, ~5–7 min wall.
  4. Per step: write a raw JSONL record (loss, grad_norm, per-group norms,
     latency breakdown, telemetry, heavy stats every 50 steps).
  5. Per ~200-step window: write a summary JSONL record.
  6. At step 200: render the "pause-and-show" snapshot HTML.
  7. Save checkpoint + sidecar files (_meta.json).

Usage:
    uv run python -m inspector_v2.train --steps 2000
    uv run python -m inspector_v2.train --steps 200    # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
import torch

from inspector_v2 import log_writer as lw
from inspector_v2.model import Config, TinyGPT, kaplan_flops_per_step

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRACES_DIR = ROOT / "traces" / "training"
WEB_DIR = ROOT / "web"

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

WINDOW_STEPS = 200  # one summary record per N steps (laptop-scale; frontier = 60s wall)
SNAPSHOT_STEP = 200
HEAVY_STATS_EVERY = 50
SAMPLE_TOKENS_EVERY = 100


def download_dataset() -> Path:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    out = DATA_DIR / "tiny_shakespeare.txt"
    if not out.exists():
        print(f"  downloading Tiny Shakespeare → {out}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, out)
    return out


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def grad_norms_per_group(model: TinyGPT) -> dict[str, float]:
    """L2 norm of gradients, grouped by module class."""
    groups = {"embed": [], "blocks": [], "head": [], "ln": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "tok_emb" in name or "pos_emb" in name:
            groups["embed"].append(p.grad)
        elif name.startswith("blocks"):
            groups["blocks"].append(p.grad)
        elif name == "head.weight":
            groups["head"].append(p.grad)
        elif "ln_f" in name:
            groups["ln"].append(p.grad)
    out = {}
    for k, gs in groups.items():
        if not gs:
            out[k] = 0.0
            continue
        sq = sum(float(g.detach().pow(2).sum().item()) for g in gs)
        out[k] = sq**0.5
    return out


def weight_norms_per_group(model: TinyGPT) -> dict[str, float]:
    groups = {"embed": [], "blocks": [], "head": [], "ln": []}
    for name, p in model.named_parameters():
        if "tok_emb" in name or "pos_emb" in name:
            groups["embed"].append(p)
        elif name.startswith("blocks"):
            groups["blocks"].append(p)
        elif name == "head.weight":
            groups["head"].append(p)
        elif "ln_f" in name:
            groups["ln"].append(p)
    return {
        k: (sum(float(p.detach().pow(2).sum().item()) for p in ps) ** 0.5) if ps else 0.0
        for k, ps in groups.items()
    }


def optimizer_state_norms(opt: torch.optim.AdamW) -> dict[str, float]:
    """L2 norm of AdamW first/second moments across all params."""
    m_sq = 0.0
    v_sq = 0.0
    n_tracked = 0
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            if "exp_avg" in st:
                m_sq += float(st["exp_avg"].detach().pow(2).sum().item())
                v_sq += float(st["exp_avg_sq"].detach().pow(2).sum().item())
                n_tracked += 1
    return {
        "adam_m_norm": m_sq**0.5,
        "adam_v_norm": v_sq**0.5,
        "tracked_params": n_tracked,
    }


def param_stats_top5(model: TinyGPT) -> list[dict]:
    """Top-5 largest parameter tensors by element count, with summary stats."""
    items = []
    for name, p in model.named_parameters():
        items.append((name, p))
    items.sort(key=lambda x: x[1].numel(), reverse=True)
    out = []
    for name, p in items[:5]:
        d = p.detach()
        out.append({
            "name": name,
            "n_elements": int(d.numel()),
            "min": float(d.min().item()),
            "max": float(d.max().item()),
            "mean": float(d.mean().item()),
            "std": float(d.std().item()),
        })
    return out


def system_telemetry(device: torch.device) -> dict:
    """RSS, CPU, mem, MPS-allocated. Captured per step."""
    p = psutil.Process()
    out = {
        "process_rss_mb": round(p.memory_info().rss / (1024**2), 2),
        "process_cpu_percent": p.cpu_percent(interval=None),
        "system_mem_used_pct": psutil.virtual_memory().percent,
        "system_load_1m": (os.getloadavg()[0] if hasattr(os, "getloadavg") else None),
        "n_threads": p.num_threads(),
        "mps_allocated_mb": None,
        "mps_driver_allocated_mb": None,
    }
    if device.type == "mps":
        try:
            out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / (1024**2), 2)
            out["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / (1024**2), 2)
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-snapshot", action="store_true", help="skip the step-200 HTML snapshot")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ---------- data ----------
    data_path = download_dataset()
    text = data_path.read_text(encoding="utf-8")
    chars, stoi, itos = build_vocab(text)
    print(f"  vocab: {len(chars)} chars; dataset: {len(text):,} chars")
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    (DATA_DIR).mkdir(exist_ok=True)
    (DATA_DIR / "vocab.json").write_text(json.dumps({"chars": chars, "stoi": stoi}, indent=2))

    # ---------- model + optimizer ----------
    cfg = Config(vocab_size=len(chars))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")
    model = TinyGPT(cfg).to(device)
    n_params = model.n_params
    print(f"  n_params: {n_params:,}  ({n_params/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    # ---------- identity ----------
    cfg_dict = {
        "vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
        "dropout": cfg.dropout, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed,
    }
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    identity = lw.JobIdentity(
        job_id=f"laptop-tinygpt-shakespeare-{today}-train",
        op_type="training",
        model_hash_prefix=lw.model_hash_prefix(model.state_dict()),
        config_hash=lw.config_hash(cfg_dict),
        dataset_fingerprint=lw.dataset_fingerprint(text),
    )
    print(f"  job_id: {identity.job_id}")
    print(f"  model_hash: {identity.model_hash_prefix}")

    # ---------- writers ----------
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = TRACES_DIR / "raw.jsonl"
    summary_path = TRACES_DIR / "summary.jsonl"
    raw_writer = lw.JSONLWriter(raw_path)
    summary_writer = lw.JSONLWriter(summary_path)

    # ---------- training loop ----------
    t0_run = time.perf_counter()
    window_t0 = t0_run
    window_t0_iso = lw.utc_iso()
    window_flops_acc = 0.0
    window_tokens_acc = 0
    window_steps_acc = 0
    summary_idx = 0

    flops_per_step = kaplan_flops_per_step(n_params, args.batch_size, cfg.block_size)
    print(f"  flops/step: {flops_per_step:.3e}; total est. flops: {flops_per_step * args.steps:.3e}")
    print(f"  raw → {raw_path}")
    print(f"  summary → {summary_path}")
    print()

    losses: list[float] = []  # for snapshot
    snapshot_records: list[dict] = []

    model.train()
    for step in range(args.steps):
        t_step = time.perf_counter()
        # data load
        t = time.perf_counter()
        x, y = get_batch(data, args.batch_size, cfg.block_size, device)
        data_load_ms = (time.perf_counter() - t) * 1000

        # forward
        t = time.perf_counter()
        _, loss = model(x, y)
        if device.type == "mps":
            torch.mps.synchronize()
        forward_ms = (time.perf_counter() - t) * 1000

        # backward
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn_groups = grad_norms_per_group(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if device.type == "mps":
            torch.mps.synchronize()
        backward_ms = (time.perf_counter() - t) * 1000

        # optimizer step
        t = time.perf_counter()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        optimizer_ms = (time.perf_counter() - t) * 1000

        step_time_s = time.perf_counter() - t_step
        loss_val = float(loss.item())
        losses.append(loss_val)
        grad_norm = (sum(v**2 for v in gn_groups.values())) ** 0.5

        tokens_this_step = args.batch_size * cfg.block_size

        # build raw record
        rt_block: dict = {
            "device": device.type,
            "n_params": n_params,
            "loss": loss_val,
            "grad_norm": grad_norm,
            "grad_norms_per_group": gn_groups,
            "data_load_ms": round(data_load_ms, 3),
            "forward_ms": round(forward_ms, 3),
            "backward_ms": round(backward_ms, 3),
            "optimizer_ms": round(optimizer_ms, 3),
            "step_time_seconds": round(step_time_s, 4),
            "tokens_per_second": round(tokens_this_step / step_time_s, 1),
            "learning_rate": args.lr,
            "telemetry": system_telemetry(device),
        }
        if step % HEAVY_STATS_EVERY == 0:
            rt_block["weight_norms_per_group"] = weight_norms_per_group(model)
            rt_block["optimizer_state_norms"] = optimizer_state_norms(opt)
            rt_block["param_stats_top5"] = param_stats_top5(model)
        if step % SAMPLE_TOKENS_EVERY == 0:
            rt_block["sample_input_ids"] = x[0, :32].tolist()
            rt_block["sample_target_ids"] = y[0, :32].tolist()

        now = lw.utc_iso()
        rec = lw.build_training_step_record(
            record_id=f"trec_{step:06d}",
            prev_hash="<placeholder>",  # JSONLWriter fills in
            identity=identity,
            window_start=window_t0_iso,
            window_end=now,
            step=step,
            flops_this_step=flops_per_step,
            tokens_this_step=tokens_this_step,
            batch_size=args.batch_size,
            sequence_length=cfg.block_size,
            real_training_block=rt_block,
        )
        rec = raw_writer.append(rec)
        if step < SNAPSHOT_STEP:
            snapshot_records.append(rec)

        # accumulate window
        window_flops_acc += flops_per_step
        window_tokens_acc += tokens_this_step
        window_steps_acc += 1

        # emit summary every WINDOW_STEPS
        if (step + 1) % WINDOW_STEPS == 0 or step == args.steps - 1:
            window_t1_iso = lw.utc_iso()
            srec = lw.build_summary_record(
                record_id=f"tsum_{summary_idx:04d}",
                prev_hash="<placeholder>",
                identity=identity,
                window_start=window_t0_iso,
                window_end=window_t1_iso,
                flops=window_flops_acc,
                tokens_processed=window_tokens_acc,
                batch_size=args.batch_size,
                sequence_length=cfg.block_size,
                steps_in_window=window_steps_acc,
            )
            summary_writer.append(srec)
            summary_idx += 1
            window_flops_acc = 0.0
            window_tokens_acc = 0
            window_steps_acc = 0
            window_t0 = time.perf_counter()
            window_t0_iso = window_t1_iso

        # snapshot at step 200
        if step + 1 == SNAPSHOT_STEP and not args.no_snapshot:
            print(f"  → step {step+1}: rendering pause-and-show snapshot")
            from inspector_v2 import snapshot
            snapshot.render(
                records=snapshot_records,
                identity=identity,
                config=cfg_dict,
                fingerprint=lw.system_fingerprint(),
                out_path=WEB_DIR / "snapshot.html",
            )

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - t0_run
            print(
                f"  step {step:5d}/{args.steps}  loss={loss_val:.3f}  "
                f"gn={grad_norm:.3f}  step_time={step_time_s*1000:.1f}ms  "
                f"tps={tokens_this_step/step_time_s:.0f}  rss={rt_block['telemetry']['process_rss_mb']:.0f}MB  "
                f"elapsed={elapsed:.1f}s"
            )

    raw_writer.close()
    summary_writer.close()

    # ---------- save checkpoint + meta ----------
    ckpt_path = TRACES_DIR / f"checkpoint_step{args.steps:05d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg_dict,
        "vocab": {"chars": chars, "stoi": stoi},
        "step": args.steps,
        "identity": {
            "job_id": identity.job_id, "model_hash_prefix": identity.model_hash_prefix,
            "config_hash": identity.config_hash, "dataset_fingerprint": identity.dataset_fingerprint,
            "code_commit": identity.code_commit_str,
        },
    }, ckpt_path)
    print(f"\n  ✓ checkpoint → {ckpt_path}")

    meta = {
        "system_fingerprint": lw.system_fingerprint(),
        "config": cfg_dict,
        "identity": {
            "job_id": identity.job_id,
            "operator": identity.operator,
            "chip_id": identity.chip_id,
            "cluster_region": identity.cluster_region,
            "model_hash_prefix": identity.model_hash_prefix,
            "config_hash": identity.config_hash,
            "dataset_fingerprint": identity.dataset_fingerprint,
            "code_commit": identity.code_commit_str,
        },
        "totals": {
            "steps": args.steps,
            "raw_records": raw_writer.n_records,
            "summary_records": summary_writer.n_records,
            "estimated_total_flops": flops_per_step * args.steps,
            "tokens_processed": args.steps * args.batch_size * cfg.block_size,
            "wall_seconds": round(time.perf_counter() - t0_run, 2),
        },
    }
    (TRACES_DIR / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  ✓ _meta.json → {TRACES_DIR / '_meta.json'}")
    print(f"\n  total wall: {meta['totals']['wall_seconds']:.1f}s")
    print(f"  raw records: {raw_writer.n_records}, summary records: {summary_writer.n_records}")
    print(f"  estimated total FLOPs: {meta['totals']['estimated_total_flops']:.3e}")


if __name__ == "__main__":
    main()
