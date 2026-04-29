"""Schema-compliant log record builders + hash-chain.

Two streams:

  - Summary log: 1 record per ~60s window. Fields are aggregate (sum of FLOPs
    and tokens across the window) and reproduce the canonical schema from
    inspector-agents v1's `logs/schema.md`.
  - Raw trace: 1 record per training step / per generated token. Same identity
    fields as summary, plus a nested `_real_training` or `_inference` block
    with everything the verifier might want.

Each record carries `prev_record_hash` (SHA-256 of the previous record's
canonical JSON, restart-per-run with "0" * 64). The verifier's audit can
detect any byte-level tamper of any record after the first one.

This file is the SINGLE source of truth for how records are shaped. If a
field name changes, change it here and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

WINDOW_SECONDS = 60   # one summary record per minute of wall time
GENESIS_HASH = "0" * 64


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(record: dict) -> str:
    """Sort-keyed, separator-stable JSON for hashing. Must be deterministic."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def hash_record(record: dict) -> str:
    return _sha256(_canonical(record))


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def system_fingerprint() -> dict:
    """Captured once per run to `_meta.json`. Identifies the machine + env."""
    try:
        import torch
        torch_version = torch.__version__
        mps_avail = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_avail = bool(torch.cuda.is_available())
    except Exception:
        torch_version, mps_avail, cuda_avail = "unknown", False, False
    return {
        "captured_at": utc_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "mps_available": mps_avail,
        "cuda_available": cuda_avail,
    }


def code_commit() -> str:
    """git short-SHA of the repo, or 'uncommitted' if not in a clean state."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return f"git:{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "git:uncommitted"


@dataclass
class JobIdentity:
    """All fields that stay constant across an entire run.

    Forced parity between training and inference: same chip_id, operator,
    cluster_region, dataset_fingerprint base. Only `op_type` and `job_id`
    suffix differ. The verifier must use semantic signals (FLOP/token ratio,
    presence of loss/grad_norm, etc.), not chip metadata, to discriminate.
    """
    chip_id: str = "macbook-m3-mps-0"
    operator: str = "jason-laptop"
    cluster_region: str = "local-mps"
    job_id: str = ""
    op_type: str = "training"   # or "inference"
    model_hash_prefix: str = ""
    config_hash: str = ""
    dataset_fingerprint: str = ""
    code_commit_str: str = field(default_factory=code_commit)


def model_hash_prefix(state_dict: dict) -> str:
    """SHA-256 of (sorted) state_dict tensor sums. Cheap, deterministic.

    Real-world this would hash the bytes; for a 3M-param model this is
    1000x faster and good enough for the demo (collisions essentially zero).
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode())
        try:
            h.update(str(float(v.float().sum().item())).encode())
        except Exception:
            h.update(str(v).encode())
    return f"sha256:{h.hexdigest()[:16]}"


def config_hash(config_dict: dict) -> str:
    return f"sha256:{_sha256(_canonical(config_dict))[:16]}"


def dataset_fingerprint(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"sha256:{_sha256(data.decode('utf-8') if isinstance(data, bytes) else data)[:16]}"


# ---------- record builders ----------


def build_summary_record(
    *,
    record_id: str,
    prev_hash: str,
    identity: JobIdentity,
    window_start: str,
    window_end: str,
    flops: float,
    tokens_processed: int,
    batch_size: int,
    sequence_length: int,
    steps_in_window: int,
    upstream_refs: list[str] | None = None,
    data_source_tags: list[str] | None = None,
) -> dict:
    """One record per ~60-second wall-clock window. ~50KB total per run."""
    return {
        "record_id": record_id,
        "prev_record_hash": prev_hash,
        "chip_id": identity.chip_id,
        "job_id": identity.job_id,
        "operator": identity.operator,
        "cluster_region": identity.cluster_region,
        "window_start": window_start,
        "window_end": window_end,
        "op_type": identity.op_type,
        "flops": float(flops),
        "tokens_processed": int(tokens_processed),
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "steps_in_window": int(steps_in_window),
        "model_hash_prefix": identity.model_hash_prefix,
        "dataset_fingerprint": identity.dataset_fingerprint,
        "upstream_refs": upstream_refs or [],
        "data_source_tags": data_source_tags or ["corpus:tiny-shakespeare"],
        "code_commit": identity.code_commit_str,
        "config_hash": identity.config_hash,
    }


def build_training_step_record(
    *,
    record_id: str,
    prev_hash: str,
    identity: JobIdentity,
    window_start: str,
    window_end: str,
    step: int,
    flops_this_step: float,
    tokens_this_step: int,
    batch_size: int,
    sequence_length: int,
    real_training_block: dict[str, Any],
) -> dict:
    """One record per training step. Embeds the heavy `_real_training` block."""
    base = build_summary_record(
        record_id=record_id,
        prev_hash=prev_hash,
        identity=identity,
        window_start=window_start,
        window_end=window_end,
        flops=flops_this_step,
        tokens_processed=tokens_this_step,
        batch_size=batch_size,
        sequence_length=sequence_length,
        steps_in_window=1,
    )
    base["_real_training"] = {"step": step, **real_training_block}
    return base


def build_inference_token_record(
    *,
    record_id: str,
    prev_hash: str,
    identity: JobIdentity,
    window_start: str,
    window_end: str,
    flops_this_token: float,
    inference_block: dict[str, Any],
) -> dict:
    """One record per generated token."""
    base = build_summary_record(
        record_id=record_id,
        prev_hash=prev_hash,
        identity=identity,
        window_start=window_start,
        window_end=window_end,
        flops=flops_this_token,
        tokens_processed=1,
        batch_size=1,
        sequence_length=1,
        steps_in_window=1,
    )
    base["_inference"] = inference_block
    return base


# ---------- writers ----------


class JSONLWriter:
    """Append-only JSONL writer with hash-chain tracking.

    The chain restarts at GENESIS_HASH for each writer instance — i.e.
    each run produces a fresh chain. The verifier validates the chain
    starting from GENESIS, so tampering with any record breaks every
    record after it.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")
        self._prev_hash = GENESIS_HASH
        self._n = 0

    def append(self, record_partial: dict) -> dict:
        """Stamp `prev_record_hash`, write the line, advance the chain.

        `record_partial` is everything except `prev_record_hash` (caller
        leaves that as a placeholder); we fill it in here so the writer
        owns the chain state.
        """
        record_partial["prev_record_hash"] = self._prev_hash
        line = _canonical(record_partial)
        self._fp.write(line + "\n")
        self._prev_hash = _sha256(line)
        self._n += 1
        return record_partial

    @property
    def prev_hash(self) -> str:
        return self._prev_hash

    @property
    def n_records(self) -> int:
        return self._n

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
