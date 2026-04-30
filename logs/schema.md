# Computation log schema

One JSONL record = one chip × one time window (default: 60 s).

Each record has **Content** (what the chip did) and **Context** (what upstream caused it).

## Record fields

```jsonc
{
  // Identity
  "record_id": "rec_000042",          // unique, monotonic within a trace
  "prev_record_id": "rec_000041",     // hash-chain: SHA-256 of prev record's canonical JSON
  "prev_record_hash": "a1b2c3…",
  "chip_id": "gpu-cluster-a-0017",    // physical chip fingerprint
  "job_id": "job_pretrain_v3",        // logical workload
  "operator": "labX-research",        // human/team owning the run
  "cluster_region": "us-west-2",      // coarse geography

  // Time window
  "window_start": "2026-04-24T09:00:00Z",
  "window_end":   "2026-04-24T09:01:00Z",

  // Content: what computation happened
  "op_type": "training" | "inference" | "eval" | "idle",
  "flops": 2.4e18,                     // floating-point ops in this window
  "tokens_processed": 1.25e9,
  "batch_size": 2048,
  "sequence_length": 8192,
  "model_hash_prefix": "sha256:3f9a…", // first 16 hex chars of the model-weight hash
  "dataset_fingerprint": "sha256:7c2d…", // Merkle root of the dataset shard used

  // Context: upstream provenance
  "upstream_refs": ["rec_000001", "rec_000012"],  // logs whose outputs fed this op
  "data_source_tags": ["corpus:c4-clean", "corpus:wikipedia-2024"],
  "code_commit": "git:labX/training@a1b2c3d",
  "config_hash": "sha256:9e1f…"
}
```

## Invariants the Verifier can check locally (no LLM needed)

1. **Hash chain**: `prev_record_hash` matches SHA-256(prev record's canonical JSON). Tamper-evident.
2. **Timestamp monotonicity**: `window_start` ≥ previous record's `window_end` for the same `chip_id`.
3. **FLOP accounting**: `flops ≤ chip_peak_flops × window_seconds` — no chip can exceed its published peak.
4. **Batch/token consistency**: `tokens_processed ≈ batch_size × sequence_length × steps_in_window` within 5% slack.
5. **Op-type × FLOP signature**: training windows have characteristic FLOP/token ratios different from inference (~6× more). A record claiming `op_type=inference` but with training-like FLOP/token is suspicious.

## Aggregate quantities commitments may constrain

- `total_training_flops`: sum of `flops` across records where `op_type="training"` within a job's time-ordered ancestry (via `upstream_refs`).
- `total_inference_flops`: same for inference.
- `domains_served`: distinct `dataset_fingerprint` × `op_type=inference` pairs.
- `data_provenance_set`: union of `data_source_tags` across all training records in a job.
