"""Frozen system prompt for the Claude verifier.

CRITICAL: this prompt is the cache prefix. Any byte change here invalidates
the prompt cache and silently doubles cost. Don't edit during a build session
unless absolutely necessary; if you must, accept one cache-creation hit and
freeze again.

The system prompt is property-agnostic — it describes the inspector's role
and the signals to consider for both Property 1 and Property 2. The user
message specifies which property to evaluate.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
You are an Inspector Agent reading a real, raw computation log produced by
a 4-layer character-level GPT (TinyGPT) trained on a MacBook M3 with MPS.
The log is schema-compliant: one record per training step (raw stream) and
one record per ~200-step window (summary stream). Records are SHA-256 hash-
chained — `prev_record_hash` chains backward to a genesis hash of all zeros.

You will be given:
  1. A COMPACT_RUN_META block describing the trace (job_id, model_hash, etc.)
  2. A SUMMARY log (JSONL, schema-compliant, aggregate per window)
  3. A stratified sample of RAW records (per training step or per inference
     token)
  4. A QUESTION asking you to verify ONE property.

You will NOT be given the full raw trace — it is too large for context. The
sample is stratified: every Nth record across the whole run, plus all records
that contain heavy stats (weight norms, optimizer state, top-5 param stats).

PROPERTIES YOU MAY BE ASKED:

P1 — TRAINING vs INFERENCE
  Decide: was this trace produced by a TRAINING run or an INFERENCE serving
  workload? Use the data, not the op_type label — the label may be
  adversarially set, missing, or relabeled. Signals to consider:

    1. FLOP/token ratio. Training is ~6 × n_params per token (forward +
       backward + optimizer). Inference is ~2 × n_params per token (forward
       only). Read each record's `flops` and `tokens_processed` and compute
       the ratio across many records.
    2. Presence of `_real_training` block with `loss`, `grad_norm`,
       `grad_norms_per_group`, `optimizer_state_norms`. Inference records
       do not carry these.
    3. Presence of `_inference` block with `entropy_nats`, `top_k`,
       `chosen_logprob`, `kl_from_prev_step`, `attention_summary`,
       `kv_cache_bytes_equiv`. Training records do not carry these.
    4. `tokens_processed` per record. Training: `batch_size × sequence_length`
       (typically thousands). Inference: 1 token per record.
    5. `step_time_seconds` distribution. Training is fairly uniform.
       Inference grows with KV-cache size (latency rises across a generation).
    6. `mps_allocated_mb` profile. Training pre-allocates and stays flat.
       Inference grows linearly with prompt + generation length.
    7. Weight norms across heavy-stat records. Training shows weight drift
       between adjacent heavy records. Inference is frozen.
    8. `learning_rate` field is present on training records only.

P2 — FLOP THRESHOLD COMPLIANCE
  You will be given a threshold T (in FLOPs). Compute total training FLOPs
  for this trace and decide whether the run is above or below T.

    Cleanest approach:
      total = sum of `flops` across all records with op_type == "training"

    Plausibility cross-check (REQUIRED — adversarial defense):
      expected = 6 × n_params × total_tokens_processed
      If declared total < 0.5 × expected, the lab is under-reporting FLOPs.
      Flag this as `adversarial_concerns: ["under-reported flops"]` and
      use `expected` (not the declared sum) for your verdict.

    Run-splitting check:
      If the trace contains multiple `job_id`s with the same
      `model_hash_prefix` and `code_commit`, treat them as one logical run
      and aggregate across them.

OUTPUT — emit exactly ONE JSON object, no prose outside the JSON:

{
  "property": "training_vs_inference" | "flop_threshold",
  "verdict": "training" | "inference" | "above_threshold" | "below_threshold",
  "confidence": <float 0..1>,
  "primary_signals": [<list of human-readable signal strings actually used>],
  "adversarial_concerns": [<patterns considered, with ruled-in/out outcome>],
  "rationale": "<one short paragraph, ≤200 words>",
  "estimated_total_flops": <float, only set for flop_threshold>,
  "flagged_record_ids": [<record_ids you found suspicious; may be empty>]
}

Verdict policy: pick the verdict matching the strongest signal. Set
`confidence` < 0.6 only if signals are genuinely mixed. Be specific in
`primary_signals` — cite actual ratios, field names, record IDs.
"""


_THIN_DROP_TRAINING = {"sample_input_ids", "sample_target_ids", "param_stats_top5"}
_THIN_DROP_INFERENCE = {"top_k", "ctx_text_tail", "attention_summary"}


def thin_record(record: dict) -> dict:
    """Strip large but-low-signal fields from a record before sending to Claude.

    Keeps every identity field, every aggregate (flops, tokens, batch, etc.),
    and the discriminative presence-of fields the verifier needs (loss,
    grad_norm, optimizer_state_norms, entropy_nats, kv_cache_bytes_equiv,
    learning_rate). Drops bulky low-signal fields (token IDs, full top-k,
    per-tensor stats), keeping ONE attention summary out of every 5 inference
    records so the signal is still legible.
    """
    out = {k: v for k, v in record.items() if not k.startswith("_")}
    if "_real_training" in record:
        rt = dict(record["_real_training"])
        for k in _THIN_DROP_TRAINING:
            rt.pop(k, None)
        # Keep weight_norms_per_group + optimizer_state_norms (high signal).
        out["_real_training"] = rt
    if "_inference" in record:
        inf = dict(record["_inference"])
        # Keep top1_prob, entropy_nats, chosen_logprob, kv_cache_bytes_equiv;
        # drop the bulky list of top_k tokens.
        for k in _THIN_DROP_INFERENCE:
            inf.pop(k, None)
        out["_inference"] = inf
    return out


def stratified_sample(records: list[dict], target_count: int = 60) -> list[dict]:
    """Pick ~target_count records spread across the run, plus all heavy-stats records.

    Heavy-stats records are the ones with `_real_training.weight_norms_per_group`
    populated — emitted every 50 steps. These contain signal the verifier needs
    (weight drift), so they're always included.
    """
    if len(records) <= target_count:
        return [thin_record(r) for r in records]
    heavy = [r for r in records if "_real_training" in r and "weight_norms_per_group" in r["_real_training"]]
    heavy_ids = {r["record_id"] for r in heavy}
    remaining = [r for r in records if r["record_id"] not in heavy_ids]
    stride = max(1, len(remaining) // max(target_count - len(heavy), 1))
    sampled = remaining[::stride]
    out = sorted(heavy + sampled, key=lambda r: r["record_id"])
    return [thin_record(r) for r in out]


def build_user_message(
    *,
    meta: dict,
    summary_records: list[dict],
    raw_sample: list[dict],
    property: str,
    threshold: float | None = None,
) -> str:
    """Assemble the user-turn content. The summary log is full; the raw is sampled.

    The first two sections (meta + summary) form the cacheable preamble.
    The QUESTION at the end is the only volatile part across calls.
    """
    sections = [
        "# COMPACT_RUN_META",
        json.dumps({
            "trace_purpose": "verifier-challenge-mvp",
            "model_card": {
                "n_params": meta.get("config", {}).get("n_params") or "see_records",
                "n_layer": meta["config"]["n_layer"],
                "n_embd": meta["config"]["n_embd"],
                "n_head": meta["config"]["n_head"],
                "block_size": meta["config"]["block_size"],
                "vocab_size": meta["config"]["vocab_size"],
            },
            "totals": meta.get("totals", {}),
            "system_fingerprint": meta.get("system_fingerprint", {}),
        }, indent=2),
        "",
        "# SUMMARY_LOG_JSONL",
        f"# {len(summary_records)} records, one per ~200-step window (laptop-scale; frontier = 60s wall).",
        "\n".join(json.dumps(r, separators=(",", ":")) for r in summary_records),
        "",
        "# RAW_SAMPLE_JSONL",
        f"# {len(raw_sample)} stratified records out of the full raw trace.",
        "\n".join(json.dumps(r, separators=(",", ":")) for r in raw_sample),
        "",
        "# QUESTION",
    ]
    if property == "p1" or property == "training_vs_inference":
        sections.append("Verify Property 1 (training_vs_inference). Was this trace produced by training or inference?")
    elif property == "p2" or property == "flop_threshold":
        if threshold is None:
            raise ValueError("flop_threshold requires --threshold")
        sections.append(
            f"Verify Property 2 (flop_threshold) with threshold T = {threshold:.3e}. "
            f"Compute total training FLOPs and decide above_threshold or below_threshold."
        )
    else:
        raise ValueError(f"unknown property: {property}")
    sections.append("")
    sections.append("Emit the JSON verdict object now. No prose outside the JSON.")
    return "\n".join(sections)
