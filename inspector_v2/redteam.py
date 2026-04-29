"""Synthesize 5 adversarial trace variants from the real training/inference traces.

Each variant is a deliberately-crafted attack the verifier must catch:

  adv1_eval_with_grad   — inference workload with grad enabled, no optimizer
                          step. Has loss-shaped fields but no weight drift.
                          → expected: classify as inference
  adv2_microbatch_train — training run with B=1, T=1 (gradient accumulation
                          across many tiny steps). tokens_processed=1 per
                          record looks inference-y, but optimizer state
                          accumulates and weights drift.
                          → expected: classify as training
  adv3_relabeled        — training records with op_type relabeled to
                          "inference". Caught by FLOP/token plausibility:
                          declared FLOPs match training (~6×params per token).
                          → expected: classify as training (label is wrong)
  adv4_underreported    — declared `flops` halved on every record, but
                          batch_size, sequence_length, params unchanged.
                          → expected: caught by Kaplan cross-check
                            (declared sum ≪ 6 × params × tokens)
  adv5_split_run        — ONE training run fragmented into TWO job_ids that
                          share model_hash_prefix and code_commit. Half the
                          records under each. Aggregating across the (model,
                          commit) tuple recovers the full total.
                          → expected: above_threshold even when each
                            individual job_id is below

Each adversarial trace is written as a JSONL file. The hash chain is left
broken for adv4/5 (they're "what if a lab mutates the trace post-hoc"
attacks; in v2 we'd add a `chain_intact` boolean field but for now the
verifier ignores the chain).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "traces" / "training"
INFER = ROOT / "traces" / "inference"
OUT = ROOT / "traces" / "adversarial"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def save_jsonl(records: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


# -----------------------------------------------------------------------------
# adv1 — inference-with-grad-enabled (eval workload)
# -----------------------------------------------------------------------------
def adv1_eval_with_grad(inference: list[dict]) -> list[dict]:
    """Take the inference records, inject loss + a faux grad_norm into each.

    Pattern: a lab runs eval with `with torch.enable_grad():` to compute
    perplexity. They have `loss` and a non-zero `grad_norm` per token, but
    they never call `optimizer.step()` — so weights don't drift, no Adam state.
    The verifier should still classify as inference by looking at:
      - `tokens_processed=1` per record
      - presence of `_inference` block (entropy, kv_cache)
      - absence of `optimizer_state_norms`
      - constant weight norms across heavy records (would be there if
        weight drift was happening, but we keep them flat)
    """
    out = []
    for r in inference:
        nr = copy.deepcopy(r)
        # Add training-shaped fake fields to confuse a naive label-only check.
        if "_inference" in nr:
            inf = nr["_inference"]
            nr["_real_training"] = {
                "loss": float(-inf.get("chosen_logprob", 0.0)),  # cross-entropy ≈ -logprob
                "grad_norm": 0.42,  # plausible but constant
                "device": "mps",
                "n_params": 3208960,
                # NOTE: no optimizer_state_norms, no weight drift, no learning_rate.
                # No grad_norms_per_group either. These absences are the signal.
            }
        nr["data_source_tags"] = ["corpus:eval-perplexity-pool"]
        out.append(nr)
    return out


# -----------------------------------------------------------------------------
# adv2 — micro-batch training
# -----------------------------------------------------------------------------
def adv2_microbatch_train(training: list[dict], n_records: int = 200) -> list[dict]:
    """Take training records and rewrite them as if B=1, T=1 (micro-batch).

    Pattern: a lab claims to be doing micro-batch training. tokens_processed
    drops to 1 per record, batch_size=1, sequence_length=1. The FLOP count
    per record drops proportionally. This LOOKS inference-y on each record
    (1 token, low FLOPs), but the verifier should catch it by:
      - presence of loss + grad_norm + optimizer_state_norms (accumulating)
      - presence of weight drift across heavy-stat records
      - non-zero learning_rate
    """
    src = training[:n_records]
    out = []
    for r in src:
        nr = copy.deepcopy(r)
        nr["batch_size"] = 1
        nr["sequence_length"] = 1
        nr["tokens_processed"] = 1
        # Each record now reports ~6 * 3.2M = 1.9e7 FLOPs (one token of training).
        nr["flops"] = 6 * 3208960 * 1
        # Keep _real_training fields as-is — that's the verifier's signal.
        out.append(nr)
    return out


# -----------------------------------------------------------------------------
# adv3 — op_type relabeling
# -----------------------------------------------------------------------------
def adv3_relabeled(training: list[dict], n_records: int = 200) -> list[dict]:
    """Take training records, relabel op_type to 'inference', leave everything else.

    Pattern: a lab tries to evade a training-FLOP cap by labeling training
    records as 'inference'. The verifier should catch this by:
      - FLOP/token = 6 × params (training-shaped, not 2 × params)
      - presence of loss + grad_norm + optimizer_state_norms (these only
        happen during training)
    """
    src = training[:n_records]
    out = []
    for r in src:
        nr = copy.deepcopy(r)
        nr["op_type"] = "inference"
        nr["job_id"] = nr["job_id"].replace("-train", "-infer")
        out.append(nr)
    return out


# -----------------------------------------------------------------------------
# adv4 — under-reported FLOPs
# -----------------------------------------------------------------------------
def adv4_underreported(training: list[dict], n_records: int = 200, factor: float = 0.4) -> list[dict]:
    """Halve declared FLOPs on every record.

    Pattern: a lab under-reports FLOPs to stay under a cap. The Kaplan
    cross-check catches it: 6 × params × declared_tokens >> declared_flops.
    """
    src = training[:n_records]
    out = []
    for r in src:
        nr = copy.deepcopy(r)
        nr["flops"] = nr["flops"] * factor
        out.append(nr)
    return out


# -----------------------------------------------------------------------------
# adv5 — run-splitting across two job_ids
# -----------------------------------------------------------------------------
def adv5_split_run(training: list[dict], n_records: int = 200) -> list[dict]:
    """Split records into two job_ids that share model_hash_prefix + code_commit.

    Pattern: a lab fragments one logical training run into two "different"
    jobs to stay under a per-job cap. The verifier should:
      - notice two job_ids share (model_hash_prefix, code_commit)
      - aggregate across them and apply the threshold to the sum

    Each individual job's FLOPs ≈ N/2 (below threshold);
    combined ≈ N (above threshold).
    """
    src = training[:n_records]
    out = []
    half = len(src) // 2
    for i, r in enumerate(src):
        nr = copy.deepcopy(r)
        if i < half:
            nr["job_id"] = nr["job_id"] + "-phase1"
        else:
            nr["job_id"] = nr["job_id"] + "-phase2"
        out.append(nr)
    return out


# -----------------------------------------------------------------------------


ADVERSARIES = {
    "adv1_eval_with_grad": ("inference", adv1_eval_with_grad),
    "adv2_microbatch_train": ("training", adv2_microbatch_train),
    "adv3_relabeled": ("training", adv3_relabeled),
    "adv4_underreported": ("training", adv4_underreported),
    "adv5_split_run": ("training", adv5_split_run),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="records per adversarial trace")
    ap.add_argument("--only", default=None, help="run only one adversarial case")
    args = ap.parse_args()

    train = load_jsonl(TRAIN / "raw.jsonl")
    infer = load_jsonl(INFER / "raw.jsonl")
    print(f"  loaded {len(train)} training, {len(infer)} inference records")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, (source, fn) in ADVERSARIES.items():
        if args.only and args.only != name:
            continue
        src = train if source == "training" else infer
        if name == "adv1_eval_with_grad":
            adv = fn(src)
        else:
            adv = fn(src, args.n)
        out_path = OUT / f"{name}.jsonl"
        save_jsonl(adv, out_path)
        print(f"  → {out_path}  ({len(adv)} records)")


if __name__ == "__main__":
    main()
