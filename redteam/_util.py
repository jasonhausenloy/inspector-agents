"""Shared helpers for red-team evasions."""

from __future__ import annotations

import copy
import hashlib
import json


def deep_copy(records: list[dict]) -> list[dict]:
    return copy.deepcopy(records)


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def rechain(records: list[dict]) -> list[dict]:
    """Recompute prev_record_hash so the trace is hash-chain-valid after mutation.

    Evasions that leave the hash chain broken would be caught by the
    deterministic `hash_chain_intact` check — which is fine, but then they'd
    all trigger deterministic. Rechaining lets subtler evasions (semantic-only)
    reach the LLM stage, which is the interesting test.
    """
    prev_hash = "0" * 64
    for r in records:
        r["prev_record_hash"] = prev_hash
        prev_hash = hashlib.sha256(_canonical(r).encode()).hexdigest()
    return records
