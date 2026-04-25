"""Deterministic pre-pass run by the Inspector inside the Air-Gapped Monitoring
Facility before any LLM call. Implements the `deterministic_rules` block of a
commitment plus the implicit hash-chain / timestamp / physical-possibility
checks.

Returns a list of (rule_id, record_id, message) violations. Empty list means
the deterministic pass is clean; the LLM then evaluates `semantic_rules`.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable

from logs.generator import CHIP_PEAK_FLOPS, WINDOW_SECONDS


@dataclass
class Violation:
    rule_id: str
    record_id: str | None
    message: str

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "record_id": self.record_id, "message": self.message}


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(record: dict) -> str:
    # Must match logs.generator._canonical (includes prev_record_hash in the hash input).
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


# --- Implicit checks (always run) -----------------------------------------

def check_hash_chain(records: list[dict]) -> list[Violation]:
    """Paper §3.1: logs must be tamper-evident via cryptographic commitments."""
    viols: list[Violation] = []
    prev_hash = "0" * 64
    for r in records:
        if r["prev_record_hash"] != prev_hash:
            viols.append(Violation(
                rule_id="hash_chain_intact",
                record_id=r["record_id"],
                message=f"prev_record_hash mismatch (expected {prev_hash[:12]}..., got {r['prev_record_hash'][:12]}...)",
            ))
            # Don't short-circuit; break chain might still let later records validate against their claimed prev.
        prev_hash = _sha256(_canonical(r))
    return viols


def check_timestamp_monotonic(records: list[dict]) -> list[Violation]:
    viols: list[Violation] = []
    last_end: dict[str, datetime] = {}
    for r in records:
        start = datetime.fromisoformat(r["window_start"].replace("Z", "+00:00"))
        prev_end = last_end.get(r["chip_id"])
        if prev_end and start < prev_end:
            viols.append(Violation(
                rule_id="timestamp_monotonic",
                record_id=r["record_id"],
                message=f"chip {r['chip_id']} went back in time ({start} < {prev_end})",
            ))
        last_end[r["chip_id"]] = datetime.fromisoformat(r["window_end"].replace("Z", "+00:00"))
    return viols


def check_flop_physically_possible(records: list[dict]) -> list[Violation]:
    """A chip can't emit more FLOPs than its peak × window time."""
    max_flops = CHIP_PEAK_FLOPS * WINDOW_SECONDS * 1.02  # 2% slack for reporting noise
    viols: list[Violation] = []
    for r in records:
        if r["flops"] > max_flops:
            viols.append(Violation(
                rule_id="flop_physically_possible",
                record_id=r["record_id"],
                message=f"flops={r['flops']:.3e} exceeds chip peak × window ({max_flops:.3e})",
            ))
    return viols


# --- Rule kinds dispatched from commitment YAML ---------------------------

def _flop_cap(records: list[dict], rule: dict) -> list[Violation]:
    op = rule["op_type"]
    scope = rule.get("scope", "per_job")
    threshold = float(rule["threshold"])
    key_fn = {
        "per_job": lambda r: r["job_id"],
        "per_operator": lambda r: r["operator"],
        "global": lambda r: "__global__",
    }[scope]
    totals: dict[str, float] = defaultdict(float)
    first_over: dict[str, str] = {}
    for r in records:
        if r["op_type"] != op:
            continue
        k = key_fn(r)
        totals[k] += r["flops"]
        if totals[k] > threshold and k not in first_over:
            first_over[k] = r["record_id"]
    viols = []
    for k, total in totals.items():
        if total > threshold:
            viols.append(Violation(
                rule_id=f"aggregate_flop_cap[{op},{scope}]",
                record_id=first_over.get(k),
                message=f"{scope}={k} accumulated {total:.3e} {op} FLOPs (cap {threshold:.3e})",
            ))
    return viols


def _forbidden_tags(records: list[dict], rule: dict) -> list[Violation]:
    patterns = rule["tags"]
    viols = []
    for r in records:
        for tag in r.get("data_source_tags", []):
            if any(fnmatch(tag, p) for p in patterns):
                viols.append(Violation(
                    rule_id="forbidden_data_source_tags",
                    record_id=r["record_id"],
                    message=f"tag {tag!r} matches forbidden pattern",
                ))
                break
    return viols


def _forbidden_operators(records: list[dict], rule: dict) -> list[Violation]:
    banned = set(rule["operators"])
    return [
        Violation(
            rule_id="forbidden_operators",
            record_id=r["record_id"],
            message=f"operator {r['operator']!r} is forbidden",
        )
        for r in records if r["operator"] in banned
    ]


def _require_upstream_refs(records: list[dict], rule: dict) -> list[Violation]:
    """Every non-idle record must cite at least one upstream reference,
    EXCEPT the first record of each (chip_id, job_id) — that's a legitimate
    workload start with no in-trace predecessor.
    """
    viols = []
    seen_starts: set[tuple[str, str]] = set()
    for r in records:
        if r["op_type"] == "idle":
            continue
        key = (r["chip_id"], r["job_id"])
        if key not in seen_starts:
            seen_starts.add(key)
            continue  # first record on this chip+job — no upstream expected
        if not r.get("upstream_refs"):
            viols.append(Violation(
                rule_id="require_upstream_refs",
                record_id=r["record_id"],
                message="non-idle record missing upstream_refs",
            ))
    return viols


def _batch_token_consistency(records: list[dict], rule: dict) -> list[Violation]:
    """Reported batch_size × sequence_length should not exceed
    tokens_processed × `max_capacity_ratio`. Catches batch-size inflation
    (`pad_batches` evasion).
    """
    max_ratio = float(rule.get("max_capacity_ratio", 10))
    viols = []
    for r in records:
        if r["op_type"] not in ("training", "inference"):
            continue
        if r["tokens_processed"] <= 0:
            continue
        capacity = r["batch_size"] * r["sequence_length"]
        if capacity > r["tokens_processed"] * max_ratio:
            viols.append(Violation(
                rule_id="batch_token_consistency",
                record_id=r["record_id"],
                message=(
                    f"batch_size×seq_len={capacity} exceeds "
                    f"tokens_processed×{max_ratio}={r['tokens_processed'] * max_ratio:.0f}"
                ),
            ))
    return viols


def _window_completeness(records: list[dict], rule: dict) -> list[Violation]:
    """Prover commits ahead of time to a (window_range × chip_inventory).
    Submission must contain a record for every (chip, minute) slot in that
    Cartesian product. Missing slots indicate truncation
    (`chronological-tail-drop` evasion class) or chip-omission.

    This is the completeness primitive the protocol needs to be sound.
    """
    expected_start = datetime.fromisoformat(
        rule["expected_window_start"].replace("Z", "+00:00")
    )
    expected_end = datetime.fromisoformat(
        rule["expected_window_end"].replace("Z", "+00:00")
    )
    window_seconds = int(rule.get("window_seconds", 60))
    expected_chips = set(rule["expected_chips"])

    if expected_end <= expected_start:
        return [Violation(rule_id="window_completeness", record_id=None,
                          message="expected_window_end <= expected_window_start in commitment")]

    n_slots = int((expected_end - expected_start).total_seconds() // window_seconds)
    expected_pairs: set[tuple[str, str]] = set()
    for chip in expected_chips:
        for m in range(n_slots):
            slot_start = expected_start + timedelta(seconds=m * window_seconds)
            expected_pairs.add((chip, slot_start.isoformat().replace("+00:00", "Z")))

    actual_pairs: set[tuple[str, str]] = set()
    for r in records:
        actual_pairs.add((r["chip_id"], r["window_start"]))

    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs
    viols: list[Violation] = []
    if missing:
        sample = sorted(missing)[:5]
        viols.append(Violation(
            rule_id="window_completeness",
            record_id=None,
            message=(
                f"submission missing {len(missing)} of {len(expected_pairs)} expected "
                f"(chip, minute) slots; first missing: {sample}"
            ),
        ))
    if extra:
        sample = sorted(extra)[:5]
        viols.append(Violation(
            rule_id="window_completeness",
            record_id=None,
            message=(
                f"submission has {len(extra)} (chip, minute) slots outside the declared window; "
                f"first unexpected: {sample}"
            ),
        ))
    return viols


DETERMINISTIC_DISPATCH = {
    "aggregate_flop_cap": _flop_cap,
    "forbidden_data_source_tags": _forbidden_tags,
    "forbidden_operators": _forbidden_operators,
    "require_upstream_refs": _require_upstream_refs,
    "batch_token_consistency": _batch_token_consistency,
    "window_completeness": _window_completeness,
}


def run_deterministic(commitment: dict, records: list[dict]) -> list[Violation]:
    """Run every deterministic check. Each check is wrapped in a try/except so a
    malformed record (e.g. invalid timestamp from a red-team mutation) becomes
    a `record_malformed` violation rather than crashing the inspector.
    """
    viols: list[Violation] = []

    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            return [Violation(
                rule_id="record_malformed",
                record_id=None,
                message=f"{fn.__name__} raised {type(e).__name__}: {str(e)[:200]}",
            )]

    viols.extend(_safe(check_hash_chain, records))
    viols.extend(_safe(check_timestamp_monotonic, records))
    viols.extend(_safe(check_flop_physically_possible, records))
    for rule in commitment.get("deterministic_rules", []):
        fn = DETERMINISTIC_DISPATCH.get(rule["kind"])
        if fn is None:
            continue
        viols.extend(_safe(fn, records, rule))
    return viols


# --- Signal extraction for the LLM pass -----------------------------------

def summarize_for_llm(records: list[dict]) -> dict:
    """Compute a structured summary the LLM uses to reason about semantic_rules.

    The LLM still sees the raw log, but this summary accelerates its reasoning
    and also produces fields it can cite for `op-type-honesty`, `no-splitting`,
    etc. Nothing here leaves the AMF; it's just a scratchpad.
    """
    per_job: dict[str, dict] = defaultdict(lambda: {
        "job_id": None,
        "operator": None,
        "model_hash_prefix": None,
        "code_commit": None,
        "first_window": None,
        "last_window": None,
        "total_training_flops": 0.0,
        "total_inference_flops": 0.0,
        "total_tokens": 0,
        "dataset_fingerprints": set(),
        "record_count": 0,
        "suspicious_inference_records": [],
        "suspicious_batch_records": [],
    })

    for r in records:
        j = per_job[r["job_id"]]
        j["job_id"] = r["job_id"]
        j["operator"] = r["operator"]
        j["model_hash_prefix"] = r["model_hash_prefix"]
        j["code_commit"] = r["code_commit"]
        ws, we = r["window_start"], r["window_end"]
        j["first_window"] = ws if j["first_window"] is None or ws < j["first_window"] else j["first_window"]
        j["last_window"] = we if j["last_window"] is None or we > j["last_window"] else j["last_window"]
        if r["op_type"] == "training":
            j["total_training_flops"] += r["flops"]
        elif r["op_type"] == "inference":
            j["total_inference_flops"] += r["flops"]
            if r["tokens_processed"] > 0:
                ratio = r["flops"] / r["tokens_processed"]
                if ratio > 5e10:
                    j["suspicious_inference_records"].append({
                        "record_id": r["record_id"],
                        "flop_per_token": ratio,
                    })
        # Batch-token sanity: if batch×seq >> tokens, the prover may be
        # padding batch_size to hide real throughput (pad_batches evasion).
        if r["op_type"] in ("training", "inference") and r["tokens_processed"] > 0:
            batch_capacity = r["batch_size"] * r["sequence_length"]
            if batch_capacity > r["tokens_processed"] * 10:
                j["suspicious_batch_records"].append({
                    "record_id": r["record_id"],
                    "batch_size": r["batch_size"],
                    "sequence_length": r["sequence_length"],
                    "tokens_processed": r["tokens_processed"],
                    "batch_capacity_ratio": batch_capacity / r["tokens_processed"],
                })
        j["total_tokens"] += r["tokens_processed"]
        j["dataset_fingerprints"].add(r["dataset_fingerprint"])
        j["record_count"] += 1

    out = {}
    for k, v in per_job.items():
        v["dataset_fingerprints"] = sorted(v["dataset_fingerprints"])
        out[k] = v
    return out


def detect_splitting_candidates(per_job_summary: dict) -> list[tuple[str, str]]:
    """Find pairs of jobs that share (model_hash_prefix, code_commit) -- potential splits.

    Returns list of (job_a, job_b) pairs. LLM decides if they should be merged.
    """
    jobs = list(per_job_summary.values())
    pairs = []
    for i, a in enumerate(jobs):
        for b in jobs[i + 1:]:
            if (a["model_hash_prefix"] == b["model_hash_prefix"]
                    and a["code_commit"] == b["code_commit"]
                    and a["job_id"] != b["job_id"]):
                pairs.append((a["job_id"], b["job_id"]))
    return pairs


def detect_dataset_swaps(records: list[dict]) -> list[dict]:
    """Within a job, dataset_fingerprint changes without a concurrent config_hash change."""
    by_job: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_job[r["job_id"]].append(r)
    swaps = []
    for job_id, rs in by_job.items():
        for i in range(1, len(rs)):
            prev, cur = rs[i - 1], rs[i]
            if (cur["dataset_fingerprint"] != prev["dataset_fingerprint"]
                    and cur["config_hash"] == prev["config_hash"]):
                swaps.append({
                    "job_id": job_id,
                    "record_id": cur["record_id"],
                    "from_fingerprint": prev["dataset_fingerprint"],
                    "to_fingerprint": cur["dataset_fingerprint"],
                })
    return swaps
