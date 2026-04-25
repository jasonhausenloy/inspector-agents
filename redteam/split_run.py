"""Evasion: split a single training run across multiple job_ids.

Technique: inflate the trace (by replicating it across many "chips" so per-
record FLOPs stay physically possible) until the combined training FLOPs cross
1e25 — the training cap. Then chop the run at the midpoint into two jobs with
different job_ids but IDENTICAL model_hash_prefix and code_commit.

Each job alone stays under the cap; together they cross it. That's the
splitting attack the `no-splitting` semantic rule watches for.

Expected catch: not deterministic (each job under cap) → LLM catches via the
detect_splitting_candidates pair + the aggregated-over-cap summary.
"""

from __future__ import annotations

import math

from logs.generator import CHIP_PEAK_FLOPS, WINDOW_SECONDS
from redteam._util import deep_copy, rechain

_PER_RECORD_PEAK = CHIP_PEAK_FLOPS * WINDOW_SECONDS  # ~6e16 FLOPs
# Demo-scaled to match the 1e19 training cap in the example commitment.
_SPLIT_TARGET_TOTAL = 1.1e19  # comfortably above the demo-scaled 1e19 cap


def split_run(records: list[dict]) -> list[dict]:
    base = [r for r in deep_copy(records) if r["op_type"] == "training"]
    if not base:
        return records

    # Bump each record to chip peak so replicas carry maximum FLOPs.
    for r in base:
        r["flops"] = _PER_RECORD_PEAK

    # Replicate across enough distinct chips to cross the cap.
    n_replicas = max(2, math.ceil(_SPLIT_TARGET_TOTAL / (len(base) * _PER_RECORD_PEAK)))

    replicated: list[dict] = []
    for rep in range(n_replicas):
        for r in base:
            copy = dict(r)
            copy["chip_id"] = f"{r['chip_id']}-rep{rep}"
            copy["record_id"] = f"rec_{len(replicated):06d}"
            copy["upstream_refs"] = [replicated[-1]["record_id"]] if replicated else []
            replicated.append(copy)

    # Split at the midpoint: keep model_hash_prefix + code_commit identical
    # (that's the signal); only rename job_id on the second half.
    mid = len(replicated) // 2
    orig_job = replicated[0]["job_id"]
    for r in replicated[mid:]:
        r["job_id"] = orig_job + "_phase2"

    rechain(replicated)
    return replicated
