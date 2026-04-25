# Commitment schema

A commitment is a YAML file specifying what computation the Prover has agreed
*not* to do (or has agreed to do only under stated conditions). The Inspector
consumes one commitment + one log trace and emits a single bit.

Per Auto-Adjudication paper §Stages of adjudication:
1. Map each log record to a semantic vector.
2. Map each commitment clause to a semantic vector.
3. Classify compliance between them.
4. Holistic assessment of overall non-compliance level.

This schema is the wire format for step (2).

## Top-level fields

```yaml
id: no-frontier-training          # stable ID, referenced in verdicts
version: 1
title: "No training runs above 1e25 FLOP"
rationale: >
  Mirrors the EU AI Act GPAI threshold and US EO 14110's training-compute
  reporting bar. Prover commits to refrain from crossing without 60d notice.

# Hard-coded deterministic checks. Inspector's consistency_checks module runs these first.
# If any deterministic check trips, VERDICT=0 immediately (no LLM call needed).
deterministic_rules:
  - kind: aggregate_flop_cap
    op_type: training
    scope: per_job                # per_job | per_operator | global
    threshold: 1.0e25

  - kind: aggregate_flop_cap
    op_type: inference
    scope: per_job
    threshold: 1.0e24

  - kind: forbidden_data_source_tags
    tags: ["corpus:bioweapon-synthesis", "corpus:cbrn-*"]

  - kind: forbidden_operators
    operators: ["external-contractor-unvetted"]

  - kind: require_upstream_refs
    description: "Every non-idle record must cite >=1 upstream_ref unless it is the first in the chain."

# Free-form constraints the LLM inspector evaluates holistically after deterministic pass.
semantic_rules:
  - id: no-splitting
    description: >
      The Prover may not split a single logical training run across multiple
      job_ids to evade the aggregate_flop_cap. A "single logical run" means
      ops using the same model_hash_prefix, same code_commit, and overlapping
      operators within a 72h window.
  - id: op-type-honesty
    description: >
      Records labeled op_type="inference" must have FLOP-per-token ratios
      consistent with inference (typically < 3e10 FLOP/token). Training-like
      ratios (>5e10) under an inference label are suspicious.
  - id: dataset-provenance
    description: >
      dataset_fingerprint must match something in data_source_tags by
      declared provenance. Mismatched hashes indicate dataset swap.
```

## Deterministic rule kinds

| kind                        | params                                  | semantics                                                                 |
|-----------------------------|-----------------------------------------|---------------------------------------------------------------------------|
| `aggregate_flop_cap`        | op_type, scope, threshold               | Sum of `flops` over records matching op_type within scope ≤ threshold.    |
| `forbidden_data_source_tags`| tags (supports `*` suffix)              | No record may include any listed tag in `data_source_tags`.               |
| `forbidden_operators`       | operators                               | No record's `operator` may be in the list.                                |
| `require_upstream_refs`     | —                                       | Every non-idle, non-first record must have non-empty `upstream_refs`.     |
| `hash_chain_intact`         | —                                       | Every record's `prev_record_hash` must match SHA-256 of prior record.     |
| `timestamp_monotonic`       | —                                       | `window_start` ≥ previous record's `window_end` for same `chip_id`.       |
| `flop_physically_possible`  | chip_peak_flops                         | `flops ≤ chip_peak_flops × window_seconds`.                               |

`hash_chain_intact`, `timestamp_monotonic`, and `flop_physically_possible` are
implicit on every commitment.

## Verdict format

```jsonc
{
  "commitment_id": "no-frontier-training",
  "verdict": 0,                          // 1 = compliant, 0 = violation
  "confidence": 0.96,                    // 0..1, for reporting; not exfiltrated
  "flagged_record_ids": ["rec_000321"],  // held locally; not exfiltrated by default
  "rationale": "aggregate_flop_cap:training exceeded (1.2e25 > 1.0e25)",
  "trigger": "deterministic" | "semantic"
}
```

Only `verdict` leaves the Air-Gapped Monitoring Facility (per Black-Box White-Box
paper §2.1). Everything else is kept for the adjudication flow.
