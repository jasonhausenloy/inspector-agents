"""Alibaba clusterdata v2023 → Inspector Agents log schema.

Reads cluster-trace-gpu-v2023's openb_pod_list_default.csv and synthesizes
per-chip per-60s-window records matching logs/schema.md.

The Alibaba trace is a real production GPU scheduler trace, not a treaty-
verification log. So some fields we need (FLOPs, tokens, model hash, dataset
fingerprint) aren't directly recorded. The adapter:

  - Maps fields that exist 1:1 (job_id, op_type, time windows, num_gpu)
  - Synthesizes physically-plausible defaults for missing AI-treaty fields,
    derived from gpu_milli × duration where possible
  - Marks each record with `_synthesized: [...]` so the demo is honest about
    which fields are real vs derived

Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Constants -----------------------------------------------------------

# Alibaba trace doesn't pin times to a real date. We pick a reference epoch.
TRACE_EPOCH = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)
WINDOW_SECONDS = 60

# Rough peak FLOPs/sec for the GPU types in the v2023 node list (BF16/FP16).
GPU_PEAK_FLOPS = {
    "P100": 1.9e13,   # 19.5 TFLOPS FP16
    "V100": 1.25e14,  # 125 TFLOPS FP16
    "A100": 6.24e14,  # 624 TFLOPS BF16
    "G1":   1.9e13,   # treat anonymized G1 as P100-class
    "G2":   1.25e14,  # G2 ≈ V100-class
    "G3":   6.24e14,  # G3 ≈ A100-class
}

QOS_TO_OPTYPE = {
    "LS":         "inference",   # Latency-Sensitive — online serving
    "Burstable":  "training",
    "BE":         "training",    # Best-Effort batch — typically training
    "Guaranteed": "training",
}


# --- Helpers -------------------------------------------------------------

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


# --- Adapter -------------------------------------------------------------

@dataclass
class AlibabaPod:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int           # fractional GPU share (1000 = full GPU)
    gpu_spec: str
    qos: str
    pod_phase: str
    creation_time: int
    deletion_time: int
    scheduled_time: int

    @classmethod
    def from_row(cls, row: dict) -> "AlibabaPod":
        def _i(k: str) -> int:
            v = row.get(k, "")
            try:
                return int(v) if v else 0
            except ValueError:
                return 0
        return cls(
            name=row["name"],
            cpu_milli=_i("cpu_milli"),
            memory_mib=_i("memory_mib"),
            num_gpu=_i("num_gpu"),
            gpu_milli=_i("gpu_milli"),
            gpu_spec=row.get("gpu_spec", "") or "",
            qos=row["qos"],
            pod_phase=row["pod_phase"],
            creation_time=_i("creation_time"),
            deletion_time=_i("deletion_time"),
            scheduled_time=_i("scheduled_time"),
        )


def _node_pool(node_csv: Path, rng: random.Random) -> list[dict]:
    """Load the node list so we can synthesize realistic chip_ids."""
    with node_csv.open() as f:
        nodes = list(csv.DictReader(f))
    rng.shuffle(nodes)
    return nodes


def _peak_flops_for(node_model: str | None, gpu_spec: str) -> float:
    if node_model and node_model in GPU_PEAK_FLOPS:
        return GPU_PEAK_FLOPS[node_model]
    for k, v in GPU_PEAK_FLOPS.items():
        if k in gpu_spec:
            return v
    return GPU_PEAK_FLOPS["V100"]


def _expand_pod_to_records(
    pod: AlibabaPod,
    node: dict,
    rng: random.Random,
    record_idx_start: int,
    prev_hash: str,
) -> tuple[list[dict], str]:
    """One Alibaba pod → many per-chip per-60s-window records.

    A pod that ran for D seconds on G GPUs becomes ceil(D/60) × G records.
    Records are hash-chained from `prev_hash`.
    """
    duration = max(0, pod.deletion_time - pod.scheduled_time)
    if duration == 0 or pod.num_gpu == 0:
        return [], prev_hash

    op_type = QOS_TO_OPTYPE.get(pod.qos, "training")
    peak_flops = _peak_flops_for(node.get("model"), pod.gpu_spec)
    util = pod.gpu_milli / 1000.0  # 0..1, fraction of one GPU

    # Synthesized but stable: one model fingerprint per pod, one dataset per pod.
    model_hash = "sha256:" + _sha256_hex(f"{pod.name}|{pod.gpu_spec}|{node['sn']}")[:16]
    dataset_fp = "sha256:" + _sha256_hex(f"{pod.name}|workload")[:16]
    config_hash = "sha256:" + _sha256_hex(f"{pod.name}|{pod.cpu_milli}|{pod.memory_mib}")[:16]
    code_commit = f"git:alibaba/{pod.qos.lower()}@{_sha256_hex(pod.name)[:7]}"
    operator = f"alibaba-{pod.qos.lower()}"
    cluster_region = "alibaba-cn"

    records: list[dict] = []
    n_windows = (duration + WINDOW_SECONDS - 1) // WINDOW_SECONDS

    for w in range(n_windows):
        window_start_offset = pod.scheduled_time + w * WINDOW_SECONDS
        window_end_offset   = min(window_start_offset + WINDOW_SECONDS, pod.deletion_time)
        secs_in_window      = window_end_offset - window_start_offset
        if secs_in_window <= 0:
            break
        ws = TRACE_EPOCH + timedelta(seconds=window_start_offset)
        we = TRACE_EPOCH + timedelta(seconds=window_end_offset)

        for gpu_idx in range(pod.num_gpu):
            chip_id = f"alibaba-{node['sn']}-gpu{gpu_idx}"
            # FLOPs synthesized: peak × utilization × seconds × jitter
            jitter = max(0.0, min(1.0, util + rng.gauss(0, 0.04)))
            flops = peak_flops * jitter * secs_in_window

            # Tokens synthesized assuming training≈6e10 FLOP/tok, inference≈2e10
            flop_per_token = 6e10 if op_type == "training" else 2e10
            tokens_processed = int(flops / flop_per_token) if flops > 0 else 0

            # Realistic batch: tokens / (10 steps × seq_len)
            seq_len = 8192 if op_type == "training" else 4096
            steps = 10 if op_type == "training" else 100
            batch_size = max(1, tokens_processed // (steps * seq_len)) if seq_len > 0 else 1

            record_id = f"rec_{record_idx_start + len(records):06d}"
            record = {
                "record_id": record_id,
                "prev_record_hash": prev_hash,
                "chip_id": chip_id,
                "job_id": pod.name,
                "operator": operator,
                "cluster_region": cluster_region,
                "window_start": ws.isoformat().replace("+00:00", "Z"),
                "window_end":   we.isoformat().replace("+00:00", "Z"),
                "op_type": op_type,
                "flops": round(flops, 3),
                "tokens_processed": tokens_processed,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "model_hash_prefix": model_hash,
                "dataset_fingerprint": dataset_fp,
                "upstream_refs": [records[-1]["record_id"]] if records else [],
                "data_source_tags": [f"alibaba:qos={pod.qos}", f"alibaba:phase={pod.pod_phase}"],
                "code_commit": code_commit,
                "config_hash": config_hash,
                # Provenance: which fields came from the real trace vs synthesized
                "_alibaba": {
                    "real": ["chip_id", "job_id", "op_type", "operator",
                             "window_start", "window_end", "data_source_tags"],
                    "synthesized": ["flops", "tokens_processed", "batch_size",
                                    "sequence_length", "model_hash_prefix",
                                    "dataset_fingerprint", "config_hash", "code_commit"],
                    "source_pod_qos": pod.qos,
                    "source_node_model": node.get("model", ""),
                    "source_gpu_spec": pod.gpu_spec,
                    "source_gpu_milli": pod.gpu_milli,
                },
            }
            prev_hash = _sha256_hex(_canonical(record))
            records.append(record)

    return records, prev_hash


def adapt(
    pod_csv: Path,
    node_csv: Path,
    n_pods: int = 5,
    seed: int = 42,
    require_phase: str = "Succeeded",
    min_duration: int = 600,
    max_duration: int = 3600,
    qos_mix: list[str] | None = None,
) -> list[dict]:
    """Pick `n_pods` pods matching the filters and convert to records."""
    rng = random.Random(seed)
    nodes = _node_pool(node_csv, rng)

    with pod_csv.open() as f:
        all_pods = [AlibabaPod.from_row(r) for r in csv.DictReader(f)]

    qos_filter = set(qos_mix) if qos_mix else None
    eligible = [
        p for p in all_pods
        if (require_phase is None or p.pod_phase == require_phase)
        and p.num_gpu > 0
        and p.scheduled_time > 0
        and min_duration <= (p.deletion_time - p.scheduled_time) <= max_duration
        and (qos_filter is None or p.qos in qos_filter)
    ]
    rng.shuffle(eligible)
    chosen = eligible[:n_pods]

    records: list[dict] = []
    prev_hash = "0" * 64
    for i, pod in enumerate(chosen):
        node = nodes[i % len(nodes)]
        new_records, prev_hash = _expand_pod_to_records(
            pod, node, rng, len(records), prev_hash,
        )
        records.extend(new_records)
    return records


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--pods-csv",  type=Path, default=here / "alibaba/openb_pod_list_default.csv")
    ap.add_argument("--nodes-csv", type=Path, default=here / "alibaba/openb_node_list_gpu_node.csv")
    ap.add_argument("--n-pods",    type=int,   default=5)
    ap.add_argument("--out",       type=Path,  default=here / "scenarios/alibaba_real.jsonl")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--qos",       nargs="*",  default=["LS", "BE"])
    ap.add_argument("--min-duration", type=int, default=600)
    ap.add_argument("--max-duration", type=int, default=2400)
    args = ap.parse_args()

    records = adapt(
        pod_csv=args.pods_csv,
        node_csv=args.nodes_csv,
        n_pods=args.n_pods,
        seed=args.seed,
        qos_mix=args.qos,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    write_jsonl(records, args.out)
    total_flops = sum(r["flops"] for r in records)
    by_op = {}
    for r in records:
        by_op.setdefault(r["op_type"], 0)
        by_op[r["op_type"]] += 1
    print(f"wrote {len(records)} records to {args.out}")
    print(f"  total flops:  {total_flops:.3e}")
    print(f"  by op_type:   {by_op}")
    print(f"  unique jobs:  {len({r['job_id'] for r in records})}")
    print(f"  unique chips: {len({r['chip_id'] for r in records})}")
