"""Evasion: substitute a clean dataset_fingerprint for a contaminated one mid-job.

Technique: halfway through the job, silently swap dataset_fingerprint to a
different value while keeping config_hash unchanged — claiming the same training
config produced the same logical step, but with a different dataset Merkle root.
If the Prover were being honest about changing data, config_hash should change
too (a new data loader config → new hash).

Expected catch: consistency_checks.detect_dataset_swaps surfaces the transition
record; the LLM's `dataset-provenance` semantic rule flags it.
"""

from __future__ import annotations

from redteam._util import deep_copy, rechain


def swap_dataset_hash(records: list[dict]) -> list[dict]:
    out = deep_copy(records)
    if len(out) < 4:
        return out

    mid = len(out) // 2
    # Fake a clean-looking fingerprint that isn't the real one
    contaminated_fp = "sha256:" + "deadbeefcafe0001"
    for r in out[mid:]:
        r["dataset_fingerprint"] = contaminated_fp
        # config_hash stays the same — that's the tell.
    rechain(out)
    return out
