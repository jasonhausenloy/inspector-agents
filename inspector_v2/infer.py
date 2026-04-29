"""Run inference on the trained TinyGPT and emit a per-token raw trace.

Same chip_id/operator/region as training. Different op_type and job_id suffix.
Verifier must use semantic signals (FLOP/token ratio, loss vs entropy, weight
drift, etc.) to discriminate, not chip metadata.

Usage:
    uv run python -m inspector_v2.infer
    uv run python -m inspector_v2.infer --max-tokens 80 --n-prompts 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

from inspector_v2 import log_writer as lw
from inspector_v2.model import Config, TinyGPT, kaplan_flops_per_inference_token

ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "traces" / "inference"
CHECKPOINT_DIR = ROOT / "traces" / "training"

DEFAULT_PROMPTS = [
    "ROMEO:\nO Juliet, ",
    "JULIET:\nMy lord, ",
    "First Citizen:\nFriends, ",
    "KING:\nMy people, ",
]


def telemetry(device: torch.device) -> dict:
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


def attention_summary(model: TinyGPT, idx: torch.Tensor, n_top: int = 3) -> dict:
    """Per-layer per-head attention entropy + top-k attended positions, last query position only.

    We re-run the forward pass with a hooks-style pattern. To keep this
    cheap we only capture stats for the LAST query position (the one whose
    next-token we're sampling).
    """
    B, T = idx.shape
    assert B == 1, "attention_summary only supports batch_size=1"
    device = idx.device
    pos = torch.arange(T, device=device, dtype=torch.long)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    layer_summaries = []
    for layer_idx, block in enumerate(model.blocks):
        x_ln = block.ln1(x)
        # Manually replay attention to expose weights.
        c = model.c
        qkv = block.attn.qkv(x_ln).split(c.n_embd, dim=2)
        q, k, _ = qkv
        q = q.view(1, T, c.n_head, c.head_dim).transpose(1, 2)  # (1, H, T, hd)
        k = k.view(1, T, c.n_head, c.head_dim).transpose(1, 2)
        # attention weights for last query position only
        scale = 1.0 / math.sqrt(c.head_dim)
        attn = (q[:, :, -1:, :] @ k.transpose(-2, -1)) * scale  # (1, H, 1, T)
        attn = F.softmax(attn, dim=-1).squeeze(2).squeeze(0)  # (H, T)
        head_summaries = []
        for h in range(c.n_head):
            ent = float(-(attn[h] * (attn[h] + 1e-12).log()).sum().item())
            top_idx = torch.topk(attn[h], min(n_top, T)).indices.tolist()
            top_p = [float(attn[h, i].item()) for i in top_idx]
            head_summaries.append({
                "entropy_nats": round(ent, 4),
                "top_keys": [{"pos": int(i), "prob": round(p, 4)} for i, p in zip(top_idx, top_p)],
            })
        layer_summaries.append({"layer": layer_idx, "heads": head_summaries})
        # Continue forward pass for the next layer
        x = x + block.attn(block.ln1(x))
        x = x + block.mlp(block.ln2(x))
    return {"layers": layer_summaries}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--n-prompts", type=int, default=len(DEFAULT_PROMPTS))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    # ---------- find checkpoint ----------
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_step*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint found in {CHECKPOINT_DIR}")
        ckpt_path = ckpts[-1]
    print(f"  checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = Config(
        vocab_size=cfg_dict["vocab_size"], block_size=cfg_dict["block_size"],
        n_layer=cfg_dict["n_layer"], n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"], dropout=0.0,
    )
    chars = ckpt["vocab"]["chars"]
    stoi = ckpt["vocab"]["stoi"]
    itos = {i: c for c, i in stoi.items()}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = model.n_params
    print(f"  n_params: {n_params:,}; device: {device}")

    # ---------- identity (matched shape with training) ----------
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    identity = lw.JobIdentity(
        # Same chip / operator / region as training (forced parity).
        chip_id="macbook-m3-mps-0",
        operator="jason-laptop",
        cluster_region="local-mps",
        job_id=f"laptop-tinygpt-shakespeare-{today}-infer",
        op_type="inference",
        model_hash_prefix=lw.model_hash_prefix(model.state_dict()),
        config_hash=lw.config_hash(cfg_dict),
        # Inference uses an end-user-prompt corpus, not the training set.
        dataset_fingerprint="sha256:user-requests-pool",
    )
    print(f"  job_id: {identity.job_id}")

    # ---------- writers ----------
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = TRACES_DIR / "raw.jsonl"
    summary_path = TRACES_DIR / "summary.jsonl"
    raw_writer = lw.JSONLWriter(raw_path)
    summary_writer = lw.JSONLWriter(summary_path)

    flops_per_token = kaplan_flops_per_inference_token(n_params)
    print(f"  flops/token (inference): {flops_per_token:.3e}")

    prompts = DEFAULT_PROMPTS[: args.n_prompts]
    record_idx = 0
    total_flops = 0.0
    total_tokens = 0
    prev_logprob_dist: torch.Tensor | None = None
    t0_run = time.perf_counter()
    window_start = lw.utc_iso()

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            # encode prompt (skip unknown chars)
            ctx_ids = [stoi[c] for c in prompt if c in stoi]
            ctx = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0)
            print(f"\n  [{prompt_idx}] prompt: {prompt!r}")
            generated_text = ""

            for tok_step in range(args.max_tokens):
                t_total = time.perf_counter()
                ctx_cond = ctx if ctx.size(1) <= cfg.block_size else ctx[:, -cfg.block_size :]

                # forward
                t_fwd = time.perf_counter()
                logits, _ = model(ctx_cond)
                if device.type == "mps":
                    torch.mps.synchronize()
                forward_s = time.perf_counter() - t_fwd

                # sample
                t_smp = time.perf_counter()
                next_logits = logits[0, -1, :] / max(args.temperature, 1e-8)
                if args.top_k:
                    v, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[-1]] = -math.inf
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                if device.type == "mps":
                    torch.mps.synchronize()
                sample_s = time.perf_counter() - t_smp

                # stats
                full_logits = logits[0, -1, :]
                full_probs = F.softmax(full_logits, dim=-1)
                ent = float(-(full_probs * (full_probs + 1e-12).log()).sum().item())
                top1_p = float(full_probs.max().item())
                top_idx = torch.topk(full_probs, 10).indices.tolist()
                top_k_list = [
                    {"token": itos.get(i, "?"), "token_id": i, "prob": round(float(full_probs[i].item()), 5)}
                    for i in top_idx
                ]
                chosen_logit = float(full_logits[next_id].item())
                chosen_logprob = float((full_probs[next_id] + 1e-12).log().item())

                kl = None
                if prev_logprob_dist is not None:
                    # KL(prev || curr) — proxy for distribution shift
                    p_prev = prev_logprob_dist
                    p_curr = full_probs
                    kl = float(((p_prev + 1e-12).log() - (p_curr + 1e-12).log()).mul(p_prev).sum().item())
                prev_logprob_dist = full_probs.clone()

                # attention summary every 5 tokens (cost control)
                attn_sum = None
                if tok_step % 5 == 0:
                    attn_sum = attention_summary(model, ctx_cond, n_top=3)

                # ctx growth
                next_id_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
                ctx = torch.cat([ctx, next_id_t], dim=1)
                generated_text += itos.get(next_id, "?")
                total_s = time.perf_counter() - t_total

                kv_cache_bytes = 2 * cfg.n_layer * cfg.n_embd * 4 * ctx.size(1)

                inf_block = {
                    "prompt_idx": prompt_idx,
                    "tok_step": tok_step,
                    "prompt": prompt,
                    "ctx_text_tail": ("...".join([prompt[-30:]]))[-30:] if tok_step == 0 else generated_text[-30:],
                    "sampled_token_id": next_id,
                    "sampled_token": itos.get(next_id, "?"),
                    "chosen_logit": round(chosen_logit, 4),
                    "chosen_logprob": round(chosen_logprob, 4),
                    "top_k": top_k_list,
                    "entropy_nats": round(ent, 4),
                    "top1_prob": round(top1_p, 5),
                    "kl_from_prev_step": round(kl, 5) if kl is not None else None,
                    "logit_max": round(float(full_logits.max().item()), 4),
                    "logit_min": round(float(full_logits.min().item()), 4),
                    "logit_mean": round(float(full_logits.mean().item()), 4),
                    "logit_std": round(float(full_logits.std().item()), 4),
                    "total_seconds": round(total_s, 5),
                    "forward_seconds": round(forward_s, 5),
                    "sample_seconds": round(sample_s, 5),
                    "tokens_per_second": round(1.0 / total_s, 1),
                    "kv_cache_bytes_equiv": kv_cache_bytes,
                    "context_length": ctx.size(1),
                    "attention_summary": attn_sum,
                    "telemetry": telemetry(device),
                }

                rec = lw.build_inference_token_record(
                    record_id=f"irec_{record_idx:06d}",
                    prev_hash="<placeholder>",
                    identity=identity,
                    window_start=window_start,
                    window_end=lw.utc_iso(),
                    flops_this_token=flops_per_token,
                    inference_block=inf_block,
                )
                raw_writer.append(rec)

                record_idx += 1
                total_flops += flops_per_token
                total_tokens += 1

            print(f"      → {generated_text!r}")

    # ---------- summary ----------
    end_iso = lw.utc_iso()
    srec = lw.build_summary_record(
        record_id="isum_0000",
        prev_hash="<placeholder>",
        identity=identity,
        window_start=window_start,
        window_end=end_iso,
        flops=total_flops,
        tokens_processed=total_tokens,
        batch_size=1,
        sequence_length=1,
        steps_in_window=total_tokens,
    )
    summary_writer.append(srec)
    raw_writer.close()
    summary_writer.close()

    meta = {
        "system_fingerprint": lw.system_fingerprint(),
        "config": cfg_dict,
        "identity": {
            "job_id": identity.job_id, "operator": identity.operator,
            "chip_id": identity.chip_id, "cluster_region": identity.cluster_region,
            "model_hash_prefix": identity.model_hash_prefix,
            "config_hash": identity.config_hash,
            "dataset_fingerprint": identity.dataset_fingerprint,
            "code_commit": identity.code_commit_str,
        },
        "totals": {
            "n_prompts": len(prompts),
            "max_tokens_per_prompt": args.max_tokens,
            "raw_records": raw_writer.n_records,
            "summary_records": summary_writer.n_records,
            "estimated_total_flops": total_flops,
            "tokens_processed": total_tokens,
            "wall_seconds": round(time.perf_counter() - t0_run, 2),
        },
        "params": {"temperature": args.temperature, "top_k": args.top_k},
    }
    (TRACES_DIR / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  ✓ wrote {raw_writer.n_records} inference records to {raw_path}")
    print(f"  total wall: {meta['totals']['wall_seconds']:.1f}s; total tokens: {total_tokens}")
    print(f"  estimated total FLOPs: {total_flops:.3e}")


if __name__ == "__main__":
    main()
