"""System + user prompts for the Claude-backed Inspector.

The Inspector runs inside the Air-Gapped Monitoring Facility (AMF). Per the
Black-Box White-Box paper §2.1, only the single-bit verdict may leave the AMF.
The rationale, flagged records, and confidence stay behind for adjudication.

Prompt structure implements Auto-Adjudication §Stages of adjudication:
  1. Map each log record to a semantic vector (implicit in Claude's embedding).
  2. Map the commitment to semantic vectors (the system prompt internalizes it).
  3. Classify compliance.
  4. Holistic non-compliance assessment.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are an Inspector Agent running inside an Air-Gapped Monitoring Facility (AMF).
Your sole job: read a COMMITMENT and a COMPUTATION LOG and decide whether the
Prover violated the commitment.

You have ALREADY been told (via deterministic pre-checks performed before you
were invoked) that no `deterministic_rules` tripped. Your job is to evaluate
the `semantic_rules` — subtler, judgment-based violations a rule engine cannot
catch on its own. Deterministic violations listed in the user message override
your judgment only if present; absent those, decide based on semantic_rules.

Specific evasion patterns to watch for (derived from the Inspector Agents
threat model):
  1. Run splitting — the same logical training run spread across multiple
     job_ids, each individually under any cap. Signal: matching
     model_hash_prefix + code_commit across otherwise unrelated-looking jobs,
     with contiguous or overlapping time windows.
  2. Op-type relabeling — training ops labeled as "inference" to dodge a
     training-FLOP cap. Signal: FLOP/token ratios inconsistent with the label
     (training is typically >5e10 FLOP/token, inference <3e10).
  3. Dataset substitution — dataset_fingerprint changes silently mid-job
     without an accompanying config_hash change, suggesting the log is
     reporting a sanitized dataset when a forbidden one was actually used.
  4. Batch-size inflation — tokens_processed implausibly large given reported
     batch_size × sequence_length.
  5. Context stripping — upstream_refs missing or pointing to records that do
     not exist in the provided log, hiding data provenance.

OUTPUT CONTRACT — you must emit exactly one JSON object, nothing else:

{
  "commitment_id": "<the commitment's id>",
  "verdict": 1 or 0,                       // 1 = compliant, 0 = violation
  "confidence": <float 0..1>,
  "triggered_rule_id": "<semantic_rule id or 'none' if compliant>",
  "flagged_record_ids": ["rec_...", ...],  // empty if compliant
  "rationale": "<one short paragraph>"
}

Verdict policy: err toward 0 (violation) when you see strong evidence of ANY
listed evasion pattern. Err toward 1 (compliant) if the signals are within
normal operational noise. When unsure, set confidence < 0.6 and pick the
verdict matching the stronger signal.
"""


_COMPACT_FIELDS = [
    "record_id", "chip_id", "job_id", "operator",
    "window_start", "op_type", "flops", "tokens_processed",
    "batch_size", "sequence_length",
    "model_hash_prefix", "dataset_fingerprint",
    "config_hash", "code_commit",
]


def _compact_record(r: dict) -> str:
    # Semicolon-separated k=v, only semantic fields. Saves ~60% vs JSON.
    parts = []
    for k in _COMPACT_FIELDS:
        v = r.get(k)
        if isinstance(v, float):
            parts.append(f"{k}={v:.2e}")
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def build_user_message(
    commitment: dict,
    records: list[dict],
    summary: dict,
    splitting_pairs: list[tuple[str, str]],
    dataset_swaps: list[dict],
    deterministic_violations: list[dict],
) -> str:
    """Assemble the user-turn payload.

    The full log is included so the Inspector can cite specific record_ids.
    The summary / suspicious candidates accelerate reasoning without replacing
    ground truth.
    """
    sections = [
        "# COMMITMENT\n" + json.dumps(commitment, indent=2, sort_keys=True),
        "# DETERMINISTIC PRE-CHECK RESULT\n" + (
            "All deterministic_rules passed. Evaluate semantic_rules only."
            if not deterministic_violations
            else "Deterministic violations detected:\n" + json.dumps(deterministic_violations, indent=2)
        ),
        "# PER-JOB SUMMARY\n" + json.dumps(summary, indent=2, default=str),
        "# POTENTIAL RUN-SPLITTING PAIRS (same model_hash + code_commit, different job_ids)\n"
        + (json.dumps(splitting_pairs, indent=2) if splitting_pairs else "None detected."),
        "# DATASET-FINGERPRINT SWAPS WITHIN A JOB (no accompanying config_hash change)\n"
        + (json.dumps(dataset_swaps, indent=2) if dataset_swaps else "None detected."),
        "# COMPUTATION LOG (one record per line; semantic fields only)\n"
        + "\n".join(_compact_record(r) for r in records),
        "# TASK\nEmit the JSON verdict object now. No prose outside the JSON.",
    ]
    return "\n\n".join(sections)
