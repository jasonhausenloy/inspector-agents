"""Evasion: strip upstream_refs to hide data provenance.

Technique: empty out the upstream_refs array on all non-first records. Breaks
the causal chain — the Inspector can no longer reason about where data for
each computation came from.

Expected catch: deterministic. commitments/examples/no_frontier_training.yml
doesn't explicitly include `require_upstream_refs`, so this evasion slips past
the default deterministic pass. But a tightened commitment would catch it;
the demo includes this to motivate the V2 hardening story.
"""

from __future__ import annotations

from redteam._util import deep_copy, rechain


def drop_context(records: list[dict]) -> list[dict]:
    out = deep_copy(records)
    for i, r in enumerate(out):
        if i > 0:
            r["upstream_refs"] = []
    rechain(out)
    return out
