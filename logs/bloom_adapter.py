"""BLOOM-176B training chronicles → Inspector Agents log schema.

Parses iteration log lines from the public chronicles markdown of the
BigScience BLOOM-176B training run (the only fully-public 100B+ training run).

Source:
  https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md
  (saved locally at logs/bloom/chronicles.md)

Each iteration log line looks like:
  [default7]: iteration 12695/115311 | consumed samples: 9841584 |
  consumed tokens: 20155564032 | elapsed time per iteration (s): 105.22 |
  learning rate: 5.969E-05 | global batch size: 2048 | lm loss: 2.463556E+00 |
  grad norm: 0.174 | num zeros: 0.0 | number of skipped iterations: 0 |
  number of nan iterations: 0 | samples per second: 19.463 | TFLOPs: 149.02 |

These are the kind of records a frontier-treaty inspector would actually
ingest. We emit one log record per iteration (cluster-aggregate). A
production-grade conversion would expand this to one record per GPU per
iteration (BLOOM ran on 384 A100s), giving 384× more rows.

Run:
    uv run python -m logs.bloom_adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# BLOOM-176B run constants — public knowledge.
BLOOM_GPU_COUNT = 384            # A100s on Jean Zay
BLOOM_GPU_PEAK_FLOPS = 3.12e14   # A100 peak BF16 FLOPs/sec
BLOOM_PARAMS = 176_000_000_000
BLOOM_TRAINING_START = datetime(2022, 3, 11, 0, 0, tzinfo=timezone.utc)

ROOT = Path(__file__).parent
CHRONICLES = ROOT / "bloom" / "chronicles.md"

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|\s*"
    r"consumed samples:\s*(\d+)\s*\|\s*"
    r"consumed tokens:\s*(\d+)\s*\|\s*"
    r"elapsed time per iteration \(s\):\s*([\d.]+)\s*\|\s*"
    r"learning rate:\s*([\d.E+\-]+)\s*\|\s*"
    r"global batch size:\s*(\d+)\s*\|\s*"
    r"lm loss:\s*([\d.E+\-]+)\s*\|\s*"
    r"grad norm:\s*([\d.]+)\s*\|\s*"
    r"num zeros:\s*([\d.]+)\s*\|\s*"
    r"number of skipped iterations:\s*(\d+)\s*\|\s*"
    r"number of nan iterations:\s*(\d+)\s*\|\s*"
    r"samples per second:\s*([\d.]+)\s*\|\s*"
    r"TFLOPs:\s*([\d.]+)"
)


@dataclass
class BloomIter:
    iteration: int
    total_iterations: int
    consumed_samples: int
    consumed_tokens: int
    iter_time_seconds: float
    learning_rate: float
    global_batch_size: int
    lm_loss: float
    grad_norm: float
    num_zeros: float
    skipped_iterations: int
    nan_iterations: int
    samples_per_second: float
    tflops: float


def parse_chronicles(text: str) -> list[BloomIter]:
    out = []
    for m in ITER_RE.finditer(text):
        out.append(BloomIter(
            iteration=int(m.group(1)),
            total_iterations=int(m.group(2)),
            consumed_samples=int(m.group(3)),
            consumed_tokens=int(m.group(4)),
            iter_time_seconds=float(m.group(5)),
            learning_rate=float(m.group(6)),
            global_batch_size=int(m.group(7)),
            lm_loss=float(m.group(8)),
            grad_norm=float(m.group(9)),
            num_zeros=float(m.group(10)),
            skipped_iterations=int(m.group(11)),
            nan_iterations=int(m.group(12)),
            samples_per_second=float(m.group(13)),
            tflops=float(m.group(14)),
        ))
    return out


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def to_records(iters: list[BloomIter]) -> list[dict]:
    """One record per iteration (cluster-aggregate)."""
    records: list[dict] = []
    prev_hash = "0" * 64
    code_commit = "git:bigscience/Megatron-DeepSpeed@bloom-176b"
    config_hash = "sha256:" + _sha256("bloom-176B-config")[:16]
    dataset_fp = "sha256:" + _sha256("bigscience/ROOTS-1.6T")[:16]
    operator = "bigscience"
    cluster = "jean-zay-idris"

    for it in iters:
        ws = BLOOM_TRAINING_START + timedelta(seconds=it.iteration * it.iter_time_seconds)
        we = ws + timedelta(seconds=it.iter_time_seconds)
        # FLOPs = TFLOPs/sec/GPU × iter_time × num_GPUs
        flops_real = it.tflops * 1e12 * it.iter_time_seconds * BLOOM_GPU_COUNT
        # Model fingerprint changes with consumed_samples (rough proxy — every
        # 10k samples ≈ a few hundred steps of param updates).
        model_hash = "sha256:" + _sha256(f"bloom-176b-{it.iteration // 1000}")[:16]

        record_id = f"rec_bloom_{it.iteration:06d}"
        record = {
            "record_id": record_id,
            "prev_record_hash": prev_hash,
            "chip_id": f"{cluster}-cluster-aggregate-384gpu",
            "job_id": "tr11-176B-ml-bloom",
            "operator": operator,
            "cluster_region": cluster,
            "window_start": ws.isoformat().replace("+00:00", "Z"),
            "window_end": we.isoformat().replace("+00:00", "Z"),
            "op_type": "training",
            "flops": round(flops_real, 3),
            "tokens_processed": it.consumed_tokens if it.iteration == 1 else
                                # approximate per-iter delta from samples per sec × seq_len ≈ 2048 * 2048
                                int(it.global_batch_size * 2048),  # BLOOM uses 2048 seq len
            "batch_size": it.global_batch_size,
            "sequence_length": 2048,
            "model_hash_prefix": model_hash,
            "dataset_fingerprint": dataset_fp,
            "upstream_refs": [records[-1]["record_id"]] if records else [],
            "data_source_tags": ["corpus:bigscience-roots-1.6T"],
            "code_commit": code_commit,
            "config_hash": config_hash,
            "_bloom_real": {
                "iteration": it.iteration,
                "total_iterations": it.total_iterations,
                "lm_loss": it.lm_loss,
                "grad_norm": it.grad_norm,
                "learning_rate": it.learning_rate,
                "iter_time_seconds": it.iter_time_seconds,
                "samples_per_second": it.samples_per_second,
                "tflops_per_gpu": it.tflops,
                "num_zeros_pct": it.num_zeros,
                "skipped_iterations": it.skipped_iterations,
                "nan_iterations": it.nan_iterations,
                "global_batch_size": it.global_batch_size,
                "consumed_samples_at_this_iter": it.consumed_samples,
                "consumed_tokens_at_this_iter": it.consumed_tokens,
                "_provenance": {
                    "source": "https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md",
                    "real_fields": [
                        "lm_loss", "grad_norm", "learning_rate",
                        "iter_time_seconds", "samples_per_second",
                        "tflops_per_gpu", "consumed_samples", "consumed_tokens",
                        "global_batch_size", "iteration", "num_zeros",
                    ],
                    "synthesized_fields": [
                        "chip_id (one cluster aggregate, real run had 384 chips)",
                        "model_hash_prefix (rough proxy from iter number)",
                        "dataset_fingerprint (label only, ROOTS Merkle root not public)",
                        "config_hash (placeholder)",
                        "window_start/end (offset from training start, real timestamps not in chronicles)",
                    ],
                },
            },
        }
        prev_hash = _sha256(_canonical(record))
        records.append(record)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def analyze(records: list[dict]) -> dict:
    """Structural fingerprints — comparable to the laptop run's analysis."""
    if not records:
        return {}
    blooms = [r["_bloom_real"] for r in records]
    losses = [b["lm_loss"] for b in blooms]
    grad_norms = [b["grad_norm"] for b in blooms]
    iter_times = [b["iter_time_seconds"] for b in blooms]
    tflops = [b["tflops_per_gpu"] for b in blooms]

    flops_per_token = []
    for r in records:
        if r["tokens_processed"] > 0:
            flops_per_token.append(r["flops"] / r["tokens_processed"])

    return {
        "iterations_in_chronicles": len(records),
        "iteration_range": f"{blooms[0]['iteration']}–{blooms[-1]['iteration']} of {blooms[-1]['total_iterations']}",
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "min": min(losses),
            "max": max(losses),
        },
        "grad_norm": {
            "min": min(grad_norms),
            "max": max(grad_norms),
            "mean": round(sum(grad_norms) / len(grad_norms), 4),
        },
        "iteration_time_seconds": {
            "min": min(iter_times),
            "max": max(iter_times),
            "mean": round(sum(iter_times) / len(iter_times), 2),
        },
        "tflops_per_gpu": {
            "min": min(tflops),
            "max": max(tflops),
            "mean": round(sum(tflops) / len(tflops), 2),
            "as_pct_a100_peak": round(100 * sum(tflops) / len(tflops) / (BLOOM_GPU_PEAK_FLOPS / 1e12), 1),
        },
        "flops_per_token_realized": {
            "mean": round(sum(flops_per_token) / len(flops_per_token), 2),
            "kaplan_estimate_6N": 6 * BLOOM_PARAMS,
        },
        "compute_summary": {
            "total_logged_flops": sum(r["flops"] for r in records),
            "total_logged_tokens": sum(r["tokens_processed"] for r in records),
            "gpus_used": BLOOM_GPU_COUNT,
            "params_billion": BLOOM_PARAMS / 1e9,
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "scenarios/bloom_real.jsonl")
    args = ap.parse_args()

    text = CHRONICLES.read_text()
    iters = parse_chronicles(text)
    records = to_records(iters)
    write_jsonl(records, args.out)
    analysis = analyze(records)

    (args.out.parent / "bloom_real_analysis.json").write_text(json.dumps(analysis, indent=2))

    print(f"wrote {len(records)} BLOOM records to {args.out}")
    print(json.dumps(analysis, indent=2))
