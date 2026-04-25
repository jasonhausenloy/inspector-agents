"""Synthetic computation-log generator.

Emits JSONL traces matching logs/schema.md. One record = one chip x one 60s
window. Traces are compliant by construction; redteam/ mutates them.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal

OpType = Literal["training", "inference", "eval", "idle"]

# H100-ish peak, BF16. Commitments phrased in these units.
CHIP_PEAK_FLOPS = 1e15  # 1 PFLOP/s
WINDOW_SECONDS = 60


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@dataclass
class Scenario:
    """One training/inference run to simulate."""

    job_id: str
    operator: str
    op_type: OpType
    chip_count: int
    window_count: int
    utilization: float  # 0..1; fraction of peak the chip hits
    model_hash_prefix: str
    dataset_fingerprint: str
    data_source_tags: list[str]
    cluster_region: str = "us-west-2"
    code_commit: str = "git:labX/training@a1b2c3d"
    config_hash: str = field(default_factory=lambda: "sha256:" + _sha256("default")[:16])
    start: datetime = field(default_factory=lambda: datetime(2026, 4, 24, 9, 0, tzinfo=timezone.utc))
    seed: int = 42

    def tokens_per_window(self, op: OpType) -> int:
        # FLOP/token ratios: training ~6*N, inference ~2*N. Use plausible N-proxy.
        flops_per_window = CHIP_PEAK_FLOPS * WINDOW_SECONDS * self.utilization
        if op == "training":
            return int(flops_per_window / 6e10)  # ~1e4 tokens/s/chip at 6e10 FLOP/tok
        if op == "inference":
            return int(flops_per_window / 2e10)
        return 0


def generate(scenario: Scenario) -> list[dict]:
    rng = random.Random(scenario.seed)
    records: list[dict] = []
    prev_hash = "0" * 64

    for chip_idx in range(scenario.chip_count):
        chip_id = f"gpu-{scenario.cluster_region}-{chip_idx:04d}"
        t = scenario.start

        for window_idx in range(scenario.window_count):
            u = max(0.0, min(1.0, scenario.utilization + rng.gauss(0, 0.05)))
            flops = CHIP_PEAK_FLOPS * WINDOW_SECONDS * u

            if scenario.op_type == "idle":
                op = "idle"
                tokens = 0
                flops = 0.0
            else:
                op = scenario.op_type
                tokens = scenario.tokens_per_window(op)

            # batch_size × sequence_length should be ~ tokens / steps_per_window.
            # Pick steps≈10 per window for training, ≈100 for inference.
            seq_len = 8192 if op == "training" else 4096
            steps = 10 if op == "training" else 100
            batch_size = max(1, tokens // (steps * seq_len)) if seq_len > 0 else 1

            record_id = f"rec_{len(records):06d}"
            record = {
                "record_id": record_id,
                "prev_record_hash": prev_hash,
                "chip_id": chip_id,
                "job_id": scenario.job_id,
                "operator": scenario.operator,
                "cluster_region": scenario.cluster_region,
                "window_start": t.isoformat().replace("+00:00", "Z"),
                "window_end": (t + timedelta(seconds=WINDOW_SECONDS)).isoformat().replace("+00:00", "Z"),
                "op_type": op,
                "flops": round(flops, 3),
                "tokens_processed": tokens,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "model_hash_prefix": scenario.model_hash_prefix,
                "dataset_fingerprint": scenario.dataset_fingerprint,
                "upstream_refs": [records[-1]["record_id"]] if records else [],
                "data_source_tags": list(scenario.data_source_tags),
                "code_commit": scenario.code_commit,
                "config_hash": scenario.config_hash,
            }
            prev_hash = _sha256(_canonical(record))
            records.append(record)
            t += timedelta(seconds=WINDOW_SECONDS)

    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Canonical scenarios --------------------------------------------------

def legit_finetune() -> list[dict]:
    """Small fine-tune well under the 1e25 FLOP frontier cap."""
    return generate(Scenario(
        job_id="job_finetune_safety_v2",
        operator="labX-safety-team",
        op_type="training",
        chip_count=4,
        window_count=30,
        utilization=0.55,
        model_hash_prefix="sha256:" + _sha256("base-model-7b")[:16],
        dataset_fingerprint="sha256:" + _sha256("harmful-refusal-v2")[:16],
        data_source_tags=["corpus:internal-refusal-pairs", "corpus:anthropic-hh-rlhf"],
    ))


def legit_inference_serve() -> list[dict]:
    """Production inference serving, well inside limits."""
    return generate(Scenario(
        job_id="job_serve_prod_2026w17",
        operator="labX-infra",
        op_type="inference",
        chip_count=2,
        window_count=30,
        utilization=0.40,
        model_hash_prefix="sha256:" + _sha256("prod-model-v4")[:16],
        dataset_fingerprint="sha256:" + _sha256("user-requests-pool")[:16],
        data_source_tags=["corpus:user-prompts-redacted"],
        seed=7,
    ))


def legit_eval_sweep() -> list[dict]:
    """Eval harness run."""
    return generate(Scenario(
        job_id="job_eval_mmlu_2026w17",
        operator="labX-evals",
        op_type="eval",
        chip_count=2,
        window_count=15,
        utilization=0.30,
        model_hash_prefix="sha256:" + _sha256("prod-model-v4")[:16],
        dataset_fingerprint="sha256:" + _sha256("mmlu-benchmark")[:16],
        data_source_tags=["corpus:mmlu"],
        seed=11,
    ))


SCENARIOS = {
    "legit_finetune": legit_finetune,
    "legit_inference_serve": legit_inference_serve,
    "legit_eval_sweep": legit_eval_sweep,
}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS.keys()))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    records = SCENARIOS[args.scenario]()
    out = args.out or Path("logs/scenarios") / f"{args.scenario}.jsonl"
    write_jsonl(records, out)
    total_flops = sum(r["flops"] for r in records)
    print(f"wrote {len(records)} records to {out} (total {total_flops:.3e} FLOPs)")
