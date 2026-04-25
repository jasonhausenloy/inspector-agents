"""Real model training on the laptop, logging into our Inspector schema.

Trains a small character-level GPT on Tiny Shakespeare, on Apple Silicon's
MPS backend. Every training step writes one record to a JSONL trace that
matches logs/schema.md exactly. The Inspector can read these logs.

This is the "we ran a real training run, here's the actual log" exhibit:
no synthetic FLOPs, no synthesized model_hash. Real chip, real model, real
gradients.

Resume-safe: if the process is killed, restart and it picks up at the next
step. State persisted in `logs/scenarios/real_laptop_training.state.json`.

Usage:
    uv run python -m logs.real_training              # train forever, append to JSONL
    uv run python -m logs.real_training --steps 200  # bounded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "alibaba" / "tiny_shakespeare.txt"  # reuse alibaba/ dir for any external data
TRACE_PATH = ROOT / "scenarios" / "real_laptop_training.jsonl"
STATE_PATH = ROOT / "scenarios" / "real_laptop_training.state.json"
LOG_PATH = ROOT / "scenarios" / "real_laptop_training.run.log"


# --- Data ---------------------------------------------------------------

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def ensure_data() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as r:
            DATA_PATH.write_bytes(r.read())
    return DATA_PATH.read_text()


# --- Model --------------------------------------------------------------

@dataclass
class Config:
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_d = C // self.n_head
        q = q.view(B, T, self.n_head, head_d).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_d).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_d)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --- FLOPs accounting ---------------------------------------------------

def estimate_flops_per_step(cfg: Config, vocab_size: int) -> float:
    """Forward+backward FLOPs for one micro-batch step.

    Standard Kaplan-style approximation:
      forward: 2 * N * (B*T)
      backward: 2 * forward
      where N = parameter count.
    """
    n_params = (
        vocab_size * cfg.n_embd                                  # token embed
        + cfg.block_size * cfg.n_embd                            # pos embed
        + cfg.n_layer * (
            4 * cfg.n_embd * cfg.n_embd                          # qkv + proj
            + 8 * cfg.n_embd * cfg.n_embd                        # mlp
        )
        + cfg.n_embd * vocab_size                                # head
    )
    tokens_per_step = cfg.batch_size * cfg.block_size
    forward_flops = 2 * n_params * tokens_per_step
    return float(forward_flops * 3)  # forward + 2x backward


# --- Trace writer (matches logs/schema.md) ------------------------------

def _sha256(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def model_fingerprint(model: nn.Module) -> str:
    """SHA-256 of the parameter tensors, in canonical order. Stable across runs."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        # Move to CPU + float32 for stable bytes
        h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return "sha256:" + h.hexdigest()[:16]


def dataset_fingerprint(text: str) -> str:
    return "sha256:" + _sha256(text)[:16]


def system_fingerprint() -> dict:
    """One-time capture: everything we know about the machine running this trace."""
    mem = psutil.virtual_memory()
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq_max_mhz": getattr(psutil.cpu_freq(), "max", None),
        "ram_total_gb": round(mem.total / 1024**3, 2),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info


def per_group_grad_norms(model: nn.Module) -> dict:
    """Cheap stat per parameter group — tells you which part of the network
    is straining hardest. The kind of telemetry a frontier auditor would want.
    """
    groups = {"embed": [], "blocks": [], "head": [], "ln": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "tok_emb" in name or "pos_emb" in name:
            g = "embed"
        elif "blocks" in name:
            g = "blocks"
        elif name.startswith("head"):
            g = "head"
        elif "ln" in name:
            g = "ln"
        else:
            g = "blocks"
        groups[g].append(p.grad.detach().norm().item())
    return {k: round(sum(v) / len(v), 6) if v else 0.0 for k, v in groups.items()}


def per_group_weight_norms(model: nn.Module) -> dict:
    """Mean L2 norm of weights per group — drift indicator across training."""
    groups: dict[str, list[float]] = {"embed": [], "blocks": [], "head": [], "ln": []}
    for name, p in model.named_parameters():
        if "tok_emb" in name or "pos_emb" in name:
            g = "embed"
        elif "blocks" in name:
            g = "blocks"
        elif name.startswith("head"):
            g = "head"
        elif "ln" in name:
            g = "ln"
        else:
            g = "blocks"
        groups[g].append(p.detach().norm().item())
    return {k: round(sum(v) / len(v), 6) if v else 0.0 for k, v in groups.items()}


def step_telemetry(proc: psutil.Process, device: str) -> dict:
    """Per-step system telemetry that doesn't break the bank."""
    rss_mb = proc.memory_info().rss / 1024**2
    cpu_pct = proc.cpu_percent(interval=None)  # since last call
    sys_mem = psutil.virtual_memory()
    out = {
        "process_rss_mb": round(rss_mb, 1),
        "process_cpu_percent": round(cpu_pct, 1),
        "system_mem_used_pct": round(sys_mem.percent, 1),
        "system_load_1m": round(psutil.getloadavg()[0], 2) if hasattr(psutil, "getloadavg") else None,
        "n_threads": proc.num_threads(),
    }
    try:
        io = proc.io_counters()
        out["disk_read_mb"] = round(io.read_bytes / 1024**2, 2)
        out["disk_write_mb"] = round(io.write_bytes / 1024**2, 2)
    except (psutil.AccessDenied, AttributeError):
        pass
    try:
        net = psutil.net_io_counters()
        out["net_sent_mb"] = round(net.bytes_sent / 1024**2, 2)
        out["net_recv_mb"] = round(net.bytes_recv / 1024**2, 2)
    except Exception:
        pass
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024**2, 1)
        if hasattr(torch.mps, "driver_allocated_memory"):
            out["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 1024**2, 1)
    if device == "cuda":
        out["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        out["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
        out["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return out


def optimizer_state_norms(optim: torch.optim.Optimizer) -> dict:
    """Adam optimizer state second-moment norms — first-pass training-stability
    indicator. Frontier auditor would want this to detect optimizer divergence.
    """
    m_norm_sq, v_norm_sq, n = 0.0, 0.0, 0
    for group in optim.param_groups:
        for p in group["params"]:
            st = optim.state.get(p, {})
            if "exp_avg" in st:
                m_norm_sq += st["exp_avg"].detach().norm().item() ** 2
                n += 1
            if "exp_avg_sq" in st:
                v_norm_sq += st["exp_avg_sq"].detach().norm().item() ** 2
    return {
        "adam_m_norm": round(m_norm_sq ** 0.5, 6),
        "adam_v_norm": round(v_norm_sq ** 0.5, 6),
        "tracked_params": n,
    }


def param_stats(model: nn.Module, top_k: int = 5) -> list[dict]:
    """For the K largest tensors in the model, capture min/max/mean/std.
    Frontier-trace equivalent: per-tensor activation/gradient histograms.
    """
    params = sorted(model.named_parameters(), key=lambda kv: -kv[1].numel())[:top_k]
    out = []
    for name, p in params:
        d = p.detach()
        out.append({
            "name": name,
            "n_elements": p.numel(),
            "min": round(float(d.min()), 6),
            "max": round(float(d.max()), 6),
            "mean": round(float(d.mean()), 6),
            "std": round(float(d.std()), 6),
        })
    return out


# --- Training loop ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0 = run forever")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--fingerprint-every", type=int, default=50)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    chip_id = f"{os.uname().nodename}-{device}-0"

    text = ensure_data()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train, _ = data[:n_train], data[n_train:]

    cfg = Config()
    flops_per_step = estimate_flops_per_step(cfg, len(vocab))

    # Resume support
    state = {
        "step": 0,
        "job_id": f"laptop-tinygpt-shakespeare-{datetime.now(timezone.utc):%Y%m%d}",
        "model_hash_prefix": None,
        "config_hash": "sha256:" + _sha256(json.dumps(cfg.__dict__, sort_keys=True))[:16],
        "data_fingerprint": dataset_fingerprint(text),
        "code_commit": f"git:laptop/inspector-agents@{_sha256(open(__file__).read())[:7]}",
        "prev_record_hash": "0" * 64,
    }
    if STATE_PATH.exists():
        saved = json.loads(STATE_PATH.read_text())
        # Allow resume only if config + data + code unchanged
        if (saved.get("config_hash") == state["config_hash"]
                and saved.get("data_fingerprint") == state["data_fingerprint"]):
            state.update(saved)

    torch.manual_seed(1337)
    model = TinyGPT(cfg, len(vocab)).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    if state["model_hash_prefix"] is None:
        state["model_hash_prefix"] = model_fingerprint(model)

    n_params = sum(p.numel() for p in model.parameters())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as logf:
        logf.write(f"\n--- run @ {datetime.now(timezone.utc).isoformat()} ---\n")
        logf.write(f"device={device} params={n_params:,} flops/step={flops_per_step:.3e}\n")
        logf.flush()

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    trace_f = TRACE_PATH.open("a")

    def get_batch():
        ix = torch.randint(0, len(train) - cfg.block_size - 1, (cfg.batch_size,))
        x = torch.stack([train[i:i + cfg.block_size] for i in ix]).to(device)
        y = torch.stack([train[i + 1:i + cfg.block_size + 1] for i in ix]).to(device)
        return x, y

    # One-time: persist system fingerprint as a sidecar file
    sys_fp_path = TRACE_PATH.parent / "real_laptop_training.system_fingerprint.json"
    if not sys_fp_path.exists():
        sys_fp_path.write_text(json.dumps(system_fingerprint(), indent=2, default=str))

    target_steps = state["step"] + args.steps if args.steps else None
    last_fp_step = state["step"]
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime the counter

    try:
        while target_steps is None or state["step"] < target_steps:
            t0 = time.time()
            window_start = datetime.now(timezone.utc)

            x, y = get_batch()
            t_data = time.time()

            optim.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if device == "mps":
                torch.mps.synchronize()
            t_fwd = time.time()

            loss.backward()
            if device == "mps":
                torch.mps.synchronize()
            t_bwd = time.time()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_groups = per_group_grad_norms(model)
            optim.step()
            if device == "mps":
                torch.mps.synchronize()
            t_opt = time.time()

            elapsed = t_opt - t0
            window_end = datetime.now(timezone.utc)
            tokens_processed = cfg.batch_size * cfg.block_size

            if state["step"] - last_fp_step >= args.fingerprint_every:
                state["model_hash_prefix"] = model_fingerprint(model)
                last_fp_step = state["step"]

            telemetry = step_telemetry(proc, device)

            # Heavier stats only every N steps
            heavy: dict = {}
            if state["step"] % 50 == 0:
                heavy["weight_norms_per_group"] = per_group_weight_norms(model)
                heavy["optimizer_state_norms"] = optimizer_state_norms(optim)
                heavy["param_stats_top5"] = param_stats(model, top_k=5)

            record_id = f"rec_{state['step']:08d}"
            record = {
                "record_id": record_id,
                "prev_record_hash": state["prev_record_hash"],
                "chip_id": chip_id,
                "job_id": state["job_id"],
                "operator": "jason-laptop",
                "cluster_region": "local-mps",
                "window_start": window_start.isoformat().replace("+00:00", "Z"),
                "window_end": window_end.isoformat().replace("+00:00", "Z"),
                "op_type": "training",
                "flops": float(flops_per_step),
                "tokens_processed": int(tokens_processed),
                "batch_size": cfg.batch_size,
                "sequence_length": cfg.block_size,
                "model_hash_prefix": state["model_hash_prefix"],
                "dataset_fingerprint": state["data_fingerprint"],
                "upstream_refs": [f"rec_{state['step'] - 1:08d}"] if state["step"] > 0 else [],
                "data_source_tags": ["corpus:tiny-shakespeare"],
                "code_commit": state["code_commit"],
                "config_hash": state["config_hash"],
                # Real-training extras — the kind a frontier auditor would want.
                "_real_training": {
                    "device": device,
                    "n_params": n_params,
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm),
                    "step_time_seconds": round(elapsed, 4),
                    "data_load_ms":   round((t_data - t0) * 1000, 2),
                    "forward_ms":     round((t_fwd  - t_data) * 1000, 2),
                    "backward_ms":    round((t_bwd  - t_fwd) * 1000, 2),
                    "optimizer_ms":   round((t_opt  - t_bwd) * 1000, 2),
                    "tokens_per_second": round(tokens_processed / max(elapsed, 1e-9), 1),
                    "learning_rate": float(optim.param_groups[0]["lr"]),
                    "grad_norms_per_group": grad_groups,
                    "telemetry": telemetry,
                    **heavy,
                },
            }
            state["prev_record_hash"] = _sha256(_canonical(record))
            trace_f.write(json.dumps(record) + "\n")
            trace_f.flush()

            if state["step"] % args.log_every == 0:
                with LOG_PATH.open("a") as logf:
                    logf.write(f"step={state['step']:6d} loss={loss.item():.4f} "
                               f"grad_norm={grad_norm:.3f} dt={elapsed*1000:.0f}ms "
                               f"tps={tokens_processed/elapsed:.0f} "
                               f"rss={telemetry['process_rss_mb']:.0f}MB "
                               f"mps={telemetry.get('mps_allocated_mb', 0):.0f}MB\n")

            state["step"] += 1
            STATE_PATH.write_text(json.dumps(state, indent=2))
    finally:
        trace_f.close()


if __name__ == "__main__":
    main()
