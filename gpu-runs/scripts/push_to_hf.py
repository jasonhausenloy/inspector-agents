"""Push captured runs to HuggingFace Datasets.

Reads HF_AUTH_TOKEN from env or gpu-runs/.env. Compresses NCCL logs in place
before upload (multi-GB raw → ~10x smaller gzipped).
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

GPU_RUNS = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def gzip_logs(run_dir: Path) -> int:
    """Gzip nccl_*.log and netdev.log in place. Returns count compressed."""
    n = 0
    for pattern in ("nccl_*.log", "netdev.log"):
        for log in run_dir.rglob(pattern):
            gz = log.with_suffix(log.suffix + ".gz")
            if gz.exists():
                continue
            with open(log, "rb") as fin, gzip.open(str(gz), "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
            log.unlink()
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=GPU_RUNS / "output" / "runs",
                    help="directory containing per-host run subdirectories")
    ap.add_argument("--repo-id", default="jasminexli/verifier-challenge-traces")
    ap.add_argument("--private", action="store_true",
                    help="create the dataset as private (default: public)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compress logs but skip the upload")
    args = ap.parse_args()

    load_env(GPU_RUNS / ".env")
    token = os.environ.get("HF_AUTH_TOKEN") or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        sys.exit("ERROR: HF_AUTH_TOKEN not set in env or gpu-runs/.env")

    if not args.run_dir.exists():
        sys.exit(f"ERROR: {args.run_dir} not found — has any host actually run?")

    host_dirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir())
    if not host_dirs:
        sys.exit(f"ERROR: no host directories in {args.run_dir}")

    if args.dry_run:
        for d in host_dirs:
            n = gzip_logs(d)
            print(f"[dry-run] {d.name}: compressed {n} log file(s)")
        print("[dry-run] would upload to:", args.repo_id)
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)

    for host_dir in host_dirs:
        n = gzip_logs(host_dir)
        print(f"{host_dir.name}: compressed {n} log file(s); uploading...")
        api.upload_folder(
            folder_path=str(host_dir),
            path_in_repo=f"runs/{host_dir.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"add run {host_dir.name}",
        )
        print(f"  uploaded {host_dir.name}")

    print(f"\ndone. https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
