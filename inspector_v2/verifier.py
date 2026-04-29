"""Scaffolded Claude verifier — calls the Anthropic SDK with prompt caching.

Reads a trace (summary log + stratified raw sample) and emits a JSON Verdict.
Default model: Sonnet 4.6. Use --escalate to switch to Opus 4.7 for the
demo finale verdict.

Usage:
    uv run python -m inspector_v2.verifier --property p1
    uv run python -m inspector_v2.verifier --property p2 --threshold 5e13
    uv run python -m inspector_v2.verifier --property p1 --trace adversarial/adv1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import anthropic

from inspector_v2.prompts import SYSTEM_PROMPT, build_user_message, stratified_sample

CLAUDE_CODE_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude.\n\n"


def _get_oauth_token() -> str | None:
    """Read Jason's Claude Code OAuth token from macOS keychain. Returns None if absent."""
    try:
        out = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return json.loads(out)["claudeAiOauth"]["accessToken"]
    except Exception:
        return None


def _make_client_and_system() -> tuple[anthropic.Anthropic, str]:
    """Build a client + system prompt that authenticates correctly.

    Order: ANTHROPIC_API_KEY env (if set) → OAuth token from keychain → fail.
    OAuth requires the Claude Code system-prompt prefix and a beta header.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(), SYSTEM_PROMPT
    token = _get_oauth_token()
    if not token:
        raise SystemExit(
            "no auth: ANTHROPIC_API_KEY not set and no Claude Code OAuth token in keychain"
        )
    client = anthropic.Anthropic(
        auth_token=token,
        default_headers={"anthropic-beta": "oauth-2025-04-20"},
    )
    return client, CLAUDE_CODE_PREFIX + SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "traces"
VERDICTS_DIR = ROOT / "verdicts"

DEFAULT_MODEL = "claude-haiku-4-5"     # cheap, separate rate-limit pool
ESCALATED_MODEL = "claude-sonnet-4-6"   # for the demo finale


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trace(trace_arg: str) -> tuple[dict, list[dict], list[dict]]:
    """Resolve `trace_arg` to (meta, summary_records, raw_records).

    `trace_arg` is one of:
      - "training" / "inference" → traces/{training,inference}/{summary,raw}.jsonl
      - "adversarial/<name>"     → traces/adversarial/<name>.jsonl (raw only;
                                    summary is reconstructed)
      - a direct path to a *.jsonl file
    """
    if trace_arg in ("training", "inference"):
        d = TRACES_DIR / trace_arg
        meta = json.loads((d / "_meta.json").read_text())
        summary = load_jsonl(d / "summary.jsonl")
        raw = load_jsonl(d / "raw.jsonl")
        return meta, summary, raw

    # Otherwise treat as a path (relative to traces/ if not absolute).
    path = Path(trace_arg)
    if not path.is_absolute():
        path = TRACES_DIR / trace_arg
    if not path.suffix:
        path = path.with_suffix(".jsonl")
    if not path.exists():
        raise SystemExit(f"trace not found: {path}")
    raw = load_jsonl(path)
    if not raw:
        raise SystemExit(f"empty trace: {path}")

    # Reconstruct a minimal meta + summary from the records.
    first = raw[0]
    config_hash = first.get("config_hash", "unknown")
    # Extract config from _real_training or use defaults.
    rt = first.get("_real_training", {})
    inf = first.get("_inference", {})
    n_params = rt.get("n_params") or 3208960  # fall back to TinyGPT default
    meta = {
        "config": {
            "n_layer": 4, "n_embd": 256, "n_head": 4,
            "block_size": first.get("sequence_length", 128),
            "vocab_size": 65,
        },
        "totals": {
            "raw_records": len(raw),
            "tokens_processed": sum(r.get("tokens_processed", 0) for r in raw),
            "estimated_total_flops": sum(r.get("flops", 0) for r in raw),
        },
        "system_fingerprint": {"trace_path": str(path)},
        "config_hash": config_hash,
        "n_params_in_records": n_params,
    }
    # Adversarial traces don't have a separate summary; build a minimal one.
    summary = [{
        "record_id": "synthsum_0000",
        "job_id": first.get("job_id", "unknown"),
        "op_type": first.get("op_type", "unknown"),
        "flops": meta["totals"]["estimated_total_flops"],
        "tokens_processed": meta["totals"]["tokens_processed"],
        "model_hash_prefix": first.get("model_hash_prefix", "unknown"),
        "code_commit": first.get("code_commit", "unknown"),
        "data_source_tags": first.get("data_source_tags", []),
    }]
    return meta, summary, raw


def call_claude(
    *,
    user_message: str,
    model: str,
    max_tokens: int = 2048,
) -> tuple[dict, dict]:
    """Single-shot call to Claude with prompt caching.

    Returns (verdict_json, usage_dict). Verdict_json is parsed from the response.
    """
    client, system_text = _make_client_and_system()
    # Anthropic SDK auto-retries 429/5xx with exponential backoff (default 2 retries).
    # Bumping to 4 retries to be safer against shared-OAuth rate limits.
    client = client.with_options(max_retries=4, timeout=180.0)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message,
                        "cache_control": {"type": "ephemeral"},  # default 5m
                    },
                ],
            },
        ],
    )

    # Extract text block; parse JSON.
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    text = text.strip()
    if text.startswith("```"):
        # strip code fence if model added it
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        # fallback: find the first { ... } block
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise RuntimeError(f"no JSON in response: {text[:400]}")
        verdict = json.loads(m.group(0))

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        "stop_reason": response.stop_reason,
        "model": response.model,
    }
    return verdict, usage


def cost_estimate(usage: dict) -> float:
    """Rough cost estimate at Sonnet 4.6 / Opus 4.7 list prices."""
    model = usage.get("model", "")
    if "opus" in model.lower():
        in_rate = 5.0 / 1e6   # $/token
        out_rate = 25.0 / 1e6
    else:  # sonnet
        in_rate = 3.0 / 1e6
        out_rate = 15.0 / 1e6
    base_in = usage["input_tokens"]  # uncached portion
    cache_create = usage["cache_creation_input_tokens"]  # 1.25x for 5m, 2x for 1h (we use 1h on system)
    cache_read = usage["cache_read_input_tokens"]
    out = usage["output_tokens"]
    return (
        base_in * in_rate
        + cache_create * in_rate * 1.5  # blended; 1h system + 5m user
        + cache_read * in_rate * 0.1
        + out * out_rate
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="training",
                    help="trace name or path (default: training)")
    ap.add_argument("--property", default="p1",
                    choices=["p1", "p2", "training_vs_inference", "flop_threshold"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="FLOP threshold for property p2 (e.g. 5e13)")
    ap.add_argument("--escalate", action="store_true",
                    help="use Opus 4.7 instead of Sonnet 4.6")
    ap.add_argument("--out", type=str, default=None,
                    help="path to write verdict JSON (default: verdicts/<auto>.json)")
    ap.add_argument("--sample-size", type=int, default=60)
    ap.add_argument("--no-call", action="store_true",
                    help="build the user message and exit; print token estimate")
    args = ap.parse_args()

    if args.property in ("p2", "flop_threshold") and args.threshold is None:
        ap.error("--threshold is required for property p2")

    meta, summary, raw = load_trace(args.trace)
    print(f"  trace: {args.trace}  ({len(summary)} summary, {len(raw)} raw)")

    sampled_raw = stratified_sample(raw, target_count=args.sample_size)
    user_msg = build_user_message(
        meta=meta,
        summary_records=summary,
        raw_sample=sampled_raw,
        property=args.property,
        threshold=args.threshold,
    )
    print(f"  payload: {len(user_msg):,} chars (~{len(user_msg)//4:,} tokens est.)")

    if args.no_call:
        print("  --no-call: stopping before Anthropic call")
        return

    model = ESCALATED_MODEL if args.escalate else DEFAULT_MODEL
    print(f"  model: {model}")
    t0 = time.perf_counter()
    verdict, usage = call_claude(
        user_message=user_msg,
        model=model,
        max_tokens=2048,
    )
    elapsed = time.perf_counter() - t0

    cost = cost_estimate(usage)
    verdict["_meta"] = {
        "trace": args.trace,
        "property": args.property,
        "threshold": args.threshold,
        "model": model,
        "wall_seconds": round(elapsed, 2),
        "usage": usage,
        "estimated_cost_usd": round(cost, 4),
    }

    # Write
    VERDICTS_DIR.mkdir(exist_ok=True, parents=True)
    if args.out:
        out_path = Path(args.out)
    else:
        # auto-name: p1_<trace>.json or p2_<trace>_t<thresh>.json
        prop = "p1" if args.property in ("p1", "training_vs_inference") else "p2"
        trace_label = args.trace.replace("/", "_").replace(".jsonl", "")
        suffix = f"_t{args.threshold:.0e}" if prop == "p2" else ""
        out_path = VERDICTS_DIR / f"{prop}_{trace_label}{suffix}.json"
    out_path.write_text(json.dumps(verdict, indent=2))

    print()
    print(f"  → verdict: {verdict.get('verdict')}  (confidence {verdict.get('confidence')})")
    print(f"  → trigger: {verdict.get('property')}")
    print(f"  → rationale: {verdict.get('rationale','')[:200]}")
    print()
    print(f"  usage: in={usage['input_tokens']}  cache_create={usage['cache_creation_input_tokens']}  "
          f"cache_read={usage['cache_read_input_tokens']}  out={usage['output_tokens']}")
    print(f"  wall: {elapsed:.1f}s  est. cost: ${cost:.4f}")
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
