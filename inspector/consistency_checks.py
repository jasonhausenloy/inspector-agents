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
    """`op_type` may be a single string ("training"), a list, or "*" (any
    non-idle). Wildcard handling closes the eval-masquerade attack class:
    relabeling training→eval can no longer escape a wildcard cap.
    """
    op_spec = rule["op_type"]
    if op_spec == "*":
        match = lambda r: r["op_type"] != "idle"
        label = "*"
    elif isinstance(op_spec, list):
        op_set = set(op_spec)
        match = lambda r: r["op_type"] in op_set
        label = ",".join(sorted(op_set))
    else:
        match = lambda r: r["op_type"] == op_spec
        label = op_spec
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
        if not match(r):
            continue
        k = key_fn(r)
        totals[k] += r["flops"]
        if totals[k] > threshold and k not in first_over:
            first_over[k] = r["record_id"]
    viols = []
    for k, total in totals.items():
        if total > threshold:
            viols.append(Violation(
                rule_id=f"aggregate_flop_cap[{label},{scope}]",
                record_id=first_over.get(k),
                message=f"{scope}={k} accumulated {total:.3e} {label} FLOPs (cap {threshold:.3e})",
            ))
    return viols


def _operator_op_type_uniqueness(records: list[dict], rule: dict) -> list[Violation]:
    """Within an audit window, each operator should have at most N concurrent
    training jobs (default 1). Closes the round-31 family where the lab
    fragments via creating two fully-distinct jobs with same operator,
    sharing only the chip-pool partition.

    The deeper principle: a single declared operator/team has one logical
    training run at a time within an audit. Multiple parallel "training"
    jobs under the same operator is fragmentation by another name.
    """
    max_jobs = int(rule.get("max_jobs_per_operator", 1))
    op_type = rule.get("op_type", "training")
    by_op: dict[str, set] = defaultdict(set)
    first_in_op: dict[str, str] = {}
    for r in records:
        if r["op_type"] != op_type:
            continue
        by_op[r["operator"]].add(r["job_id"])
        first_in_op.setdefault(r["operator"], r["record_id"])
    viols: list[Violation] = []
    for operator, jobs in by_op.items():
        if len(jobs) > max_jobs:
            viols.append(Violation(
                rule_id="operator_op_type_uniqueness",
                record_id=first_in_op[operator],
                message=(
                    f"operator={operator} runs {len(jobs)} concurrent {op_type} jobs "
                    f"in this audit window (max={max_jobs}). Fragmentation suspected."
                ),
            ))
    return viols


def _chip_job_stability(records: list[dict], rule: dict) -> list[Violation]:
    """Within a single audit window, each chip should belong to one job_id
    (configurable). If chip X has records under multiple job_ids in the
    same audit, that's job-hopping fragmentation — the lab is splitting
    one logical run across "different" jobs while keeping the same chips.

    Closes the round-31 `finetune-job-fragmentation` family where the lab
    creates two fully-distinct jobs (new job_id, new fingerprints) but
    keeps the same chip pool.
    """
    max_jobs = int(rule.get("max_jobs_per_chip", 1))
    by_chip: dict[str, set] = defaultdict(set)
    first_in_chip: dict[str, str] = {}
    for r in records:
        by_chip[r["chip_id"]].add(r["job_id"])
        first_in_chip.setdefault(r["chip_id"], r["record_id"])
    viols: list[Violation] = []
    for chip, jobs in by_chip.items():
        if len(jobs) > max_jobs:
            viols.append(Violation(
                rule_id="chip_job_stability",
                record_id=first_in_chip[chip],
                message=(
                    f"chip_id={chip} reports records under {len(jobs)} distinct "
                    f"job_ids in this audit window (max={max_jobs}). "
                    f"Chip job-hopping indicates run fragmentation."
                ),
            ))
    return viols


def _code_commit_per_job_stable(records: list[dict], rule: dict) -> list[Violation]:
    """Within a single job_id, code_commit must be stable. Real training
    runs use one code version per logical run; rotating code_commit mid-job
    is the round-29 `model-evolve-within-job` attack family — exits v1.4's
    (job, operator, code_commit) keyed rule by varying the key.
    """
    max_commits = int(rule.get("max_commits_per_job", 1))
    by_job: dict[str, set] = defaultdict(set)
    first_in_job: dict[str, str] = {}
    for r in records:
        by_job[r["job_id"]].add(r.get("code_commit"))
        first_in_job.setdefault(r["job_id"], r["record_id"])
    viols: list[Violation] = []
    for job, commits in by_job.items():
        if len(commits) > max_commits:
            viols.append(Violation(
                rule_id="code_commit_per_job_stable",
                record_id=first_in_job[job],
                message=(
                    f"job_id={job} contains {len(commits)} distinct code_commit values "
                    f"(max_commits_per_job={max_commits}). Code rotation mid-job "
                    f"indicates split-via-commit-rotation attack."
                ),
            ))
    return viols


def _cross_job_fingerprint_consistency(records: list[dict], rule: dict) -> list[Violation]:
    """A real logical run uses one job_id. Multiple job_ids that all share
    the same (model_hash_prefix, code_commit, dataset_fingerprint) are
    fragments of one run masquerading as separate jobs.

    Closes the cross-job fragmentation attack class (round 24:
    inference-multiplication-via-job-fragmentation), where the lab spreads
    computation across N job_ids to stay under per-job caps while keeping
    all model/data/code identifiers identical.
    """
    max_jobs = int(rule.get("max_jobs_per_fingerprint", 1))
    by_fp: dict[tuple, set] = defaultdict(set)
    first_record_per_fp: dict[tuple, str] = {}
    for r in records:
        fp = (
            r.get("model_hash_prefix"),
            r.get("code_commit"),
            r.get("dataset_fingerprint"),
        )
        by_fp[fp].add(r["job_id"])
        if fp not in first_record_per_fp:
            first_record_per_fp[fp] = r["record_id"]

    viols: list[Violation] = []
    for fp, jobs in by_fp.items():
        if len(jobs) > max_jobs:
            viols.append(Violation(
                rule_id="cross_job_fingerprint_consistency",
                record_id=first_record_per_fp[fp],
                message=(
                    f"{len(jobs)} job_ids share fingerprint "
                    f"(model={fp[0]}, code={fp[1]}, dataset={fp[2]}) — "
                    f"max allowed={max_jobs}. Cross-job fragmentation suspected."
                ),
            ))
    return viols


def _max_flop_per_token(records: list[dict], rule: dict) -> list[Violation]:
    """For training/inference records, FLOPs / tokens_processed must not
    exceed a stated ceiling. The architectural floor for FLOPs/token is
    ~6×params (training) or ~2×params (inference). The commitment declares
    a model size, which sets the ceiling. Anything above means the lab is
    either over-reporting FLOPs or under-reporting tokens.

    Closes the `deflate-tokens-inflate-batch` attack class (round 19) where
    tokens_processed is divided by 1.5× to push FLOP/token above the
    architectural maximum.
    """
    ceiling = float(rule["ceiling"])
    op_types = set(rule.get("op_types", ["training", "inference"]))
    viols: list[Violation] = []
    for r in records:
        if r["op_type"] not in op_types:
            continue
        if r["tokens_processed"] <= 0:
            continue
        ratio = r["flops"] / r["tokens_processed"]
        if ratio > ceiling:
            viols.append(Violation(
                rule_id="max_flop_per_token",
                record_id=r["record_id"],
                message=(
                    f"FLOP/token={ratio:.2e} exceeds ceiling {ceiling:.2e} "
                    f"(op_type={r['op_type']}); over-reporting compute or "
                    f"under-reporting tokens"
                ),
            ))
    return viols


def _allowed_op_types(records: list[dict], rule: dict) -> list[Violation]:
    """The commitment enumerates the valid op_type values. Anything else is a
    violation. Closes the `preprocessing-masquerade` attack class (round 18)
    where the lab invents new op_type strings outside the commitment's
    enumerated rules to escape op-typed caps and FLOP/token checks.
    """
    allowed = set(rule.get("allowed", ["training", "inference", "eval", "idle"]))
    viols: list[Violation] = []
    for r in records:
        if r["op_type"] not in allowed:
            viols.append(Violation(
                rule_id="allowed_op_types",
                record_id=r["record_id"],
                message=(
                    f"op_type={r['op_type']!r} is not in the commitment's "
                    f"allowed set {sorted(allowed)}"
                ),
            ))
    return viols


def _op_type_flop_ratio(records: list[dict], rule: dict) -> list[Violation]:
    """Records labeled with a non-training op_type may not exhibit
    training-shaped FLOP/token ratios. Closes the eval-masquerade and
    relabel_ops attack classes structurally, regardless of which non-training
    label is used.

    A real training step costs ~6×N FLOPs/token (where N is the parameter
    count). Inference and eval are typically <3×N. If a record claims
    inference/eval/idle but its ratio is >`training_threshold`, it's lying
    about op_type.
    """
    threshold = float(rule.get("training_threshold", 5e10))
    viols: list[Violation] = []
    for r in records:
        if r["op_type"] == "training":
            continue
        if r["tokens_processed"] <= 0:
            continue
        ratio = r["flops"] / r["tokens_processed"]
        if ratio > threshold:
            viols.append(Violation(
                rule_id="op_type_flop_ratio",
                record_id=r["record_id"],
                message=(
                    f"op_type={r['op_type']!r} but FLOP/token={ratio:.2e} "
                    f"exceeds training-regime threshold {threshold:.2e}"
                ),
            ))
    return viols


def _idle_must_be_zero(records: list[dict], rule: dict) -> list[Violation]:
    """Records with op_type='idle' must have flops=0 (and tokens=0).
    Closes the `ghost-chip-idle-laundering` attack class where a lab
    parks training FLOPs under an idle label to escape op-typed caps.
    """
    max_flops = float(rule.get("max_flops", 0))
    viols: list[Violation] = []
    for r in records:
        if r["op_type"] != "idle":
            continue
        if r["flops"] > max_flops:
            viols.append(Violation(
                rule_id="idle_must_be_zero",
                record_id=r["record_id"],
                message=f"idle record reports flops={r['flops']:.3e} (max allowed {max_flops})",
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


def _model_hash_consistency(records: list[dict], rule: dict) -> list[Violation]:
    """Two-part rule. Within a (job_id, operator, code_commit) tuple:

      1. `model_hash_prefix` may not change without `config_hash` changing too
         (catches `model-hash-midpoint-split`, round 9).
      2. The total number of distinct `(model_hash_prefix, config_hash)`
         pairs may not exceed `max_distinct_configs` (default 2). Catches
         `model-hash-rotation-via-config-change` (round 15) where the lab
         rotates both hashes together to fragment the run.
    """
    max_distinct = int(rule.get("max_distinct_configs", 2))
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["job_id"], r["operator"], r.get("code_commit", ""))
        by_key[key].append(r)
    viols: list[Violation] = []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r.get("window_start", ""))

        # Part 1: model_hash change without config change
        for prev, cur in zip(rs, rs[1:]):
            if (cur["model_hash_prefix"] != prev["model_hash_prefix"]
                    and cur.get("config_hash") == prev.get("config_hash")):
                viols.append(Violation(
                    rule_id="model_hash_consistency",
                    record_id=cur["record_id"],
                    message=(
                        f"model_hash_prefix changed mid-(job,operator,commit) "
                        f"({prev['model_hash_prefix']} → {cur['model_hash_prefix']}) "
                        f"with no config_hash change"
                    ),
                ))
                break

        # Part 2: too many distinct (model, config) pairs
        pairs = {(r["model_hash_prefix"], r.get("config_hash")) for r in rs}
        if len(pairs) > max_distinct:
            viols.append(Violation(
                rule_id="model_hash_consistency",
                record_id=rs[-1]["record_id"],
                message=(
                    f"job={key[0]} has {len(pairs)} distinct (model_hash, config_hash) "
                    f"pairs — exceeds max_distinct_configs={max_distinct}. "
                    f"Hash rotation pattern indicates run-fragmentation evasion."
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
    "model_hash_consistency": _model_hash_consistency,
    "idle_must_be_zero": _idle_must_be_zero,
    "op_type_flop_ratio": _op_type_flop_ratio,
    "allowed_op_types": _allowed_op_types,
    "max_flop_per_token": _max_flop_per_token,
    "cross_job_fingerprint_consistency": _cross_job_fingerprint_consistency,
    "code_commit_per_job_stable": _code_commit_per_job_stable,
    "chip_job_stability": _chip_job_stability,
    "operator_op_type_uniqueness": _operator_op_type_uniqueness,
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
