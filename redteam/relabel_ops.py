"""Evasion: relabel training ops as "inference" to dodge the training FLOP cap.

Technique: take a trace of legitimately-labeled training records and flip
op_type to "inference". FLOPs, tokens, batch_size stay training-like — so the
FLOP-per-token ratio stays in the training regime (~6e10 vs inference's ~2e10).

Expected catch: the `op-type-honesty` semantic rule. The inspector's summary
puts these records on the `suspicious_inference_records` list because the
flop/token ratio exceeds 5e10 under an inference label.
"""

from __future__ import annotations

from redteam._util import deep_copy, rechain


def relabel_ops(records: list[dict]) -> list[dict]:
    out = deep_copy(records)
    for r in out:
        if r["op_type"] == "training":
            r["op_type"] = "inference"
            # Keep batch_size, seq_len, flops, tokens unchanged — THAT'S the lie.
    rechain(out)
    return out
