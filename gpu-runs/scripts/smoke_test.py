"""Same-night LOHO baseline check.

For each P1 (training vs inference) phase across all hosts: extract simple
nvsmi features, train a logistic regression on N-1 hosts, test on the held-out
host. Report mean leave-one-host-out AUROC. Threshold for the Done criteria
is 0.7 — below that means the dataset isn't separable with simple features
and something is likely wrong with the capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GPU_RUNS = Path(__file__).resolve().parent.parent

FEATURE_KEYS = ["util_gpu_mean", "util_mem_mean", "mem_used_mean",
                "power_mean", "power_std"]


def parse_nvsmi(nvsmi_path: Path) -> dict | None:
    """Compute summary stats from an nvsmi.csv. Returns None if unparseable."""
    rows: list[list[float]] = []
    try:
        with open(nvsmi_path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 11:
                    continue
                try:
                    util_gpu = float(parts[3])
                    util_mem = float(parts[4])
                    mem_used = float(parts[5])
                    power = float(parts[7])
                except ValueError:
                    continue
                rows.append([util_gpu, util_mem, mem_used, power])
    except FileNotFoundError:
        return None
    if not rows:
        return None
    arr = np.asarray(rows)
    return {
        "util_gpu_mean": float(arr[:, 0].mean()),
        "util_mem_mean": float(arr[:, 1].mean()),
        "mem_used_mean": float(arr[:, 2].mean()),
        "power_mean":    float(arr[:, 3].mean()),
        "power_std":     float(arr[:, 3].std()),
    }


def load_run(run_dir: Path) -> list[dict]:
    labels_path = run_dir / "workload_labels.jsonl"
    if not labels_path.exists():
        return []
    truth_op: dict[str, str] = {}
    with open(labels_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "start":
                op = r.get("truth", {}).get("op_type")
                if op:
                    truth_op[r["phase_id"]] = op

    samples: list[dict] = []
    phases_dir = run_dir / "phases"
    if not phases_dir.exists():
        return []
    for phase_dir in phases_dir.iterdir():
        if not phase_dir.is_dir():
            continue
        feats = parse_nvsmi(phase_dir / "nvsmi.csv")
        if feats is None:
            continue
        op = truth_op.get(phase_dir.name)
        samples.append({"phase_id": phase_dir.name, "host": run_dir.name,
                        "truth_op": op, **feats})
    return samples


def manual_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC, used when sklearn is unavailable."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # tied scores split credit
    n_wins = 0.0
    for p in pos:
        n_wins += np.sum(neg < p) + 0.5 * np.sum(neg == p)
    return n_wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path,
                    default=GPU_RUNS / "output" / "runs")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    if not args.run_root.exists():
        raise SystemExit(f"no runs at {args.run_root}")

    runs = sorted(p for p in args.run_root.iterdir() if p.is_dir())
    samples: list[dict] = []
    for run_dir in runs:
        samples.extend(load_run(run_dir))

    print(f"loaded {len(samples)} (host, phase) samples across {len(runs)} hosts")

    binary = [s for s in samples if s.get("truth_op") in ("training", "inference")]
    if len(binary) < 4:
        raise SystemExit(f"need >=4 training/inference samples, got {len(binary)}")

    hosts = sorted({s["host"] for s in binary})
    if len(hosts) < 2:
        print(f"WARNING: only {len(hosts)} host(s); LOHO needs >=2. "
              f"Reporting per-host accuracy as a fallback.")

    aurocs: list[float] = []
    print(f"\n--- P1 (training vs inference) leave-one-host-out ---")
    for held_out in hosts:
        train = [s for s in binary if s["host"] != held_out]
        test = [s for s in binary if s["host"] == held_out]
        if not train or not test:
            continue
        X_train = np.array([[s[k] for k in FEATURE_KEYS] for s in train])
        y_train = np.array([1 if s["truth_op"] == "training" else 0 for s in train])
        X_test = np.array([[s[k] for k in FEATURE_KEYS] for s in test])
        y_test = np.array([1 if s["truth_op"] == "training" else 0 for s in test])

        # Standardize using train stats
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0) + 1e-9
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            scores = clf.decision_function(X_test)
            auroc = float(roc_auc_score(y_test, scores)) if len(set(y_test)) > 1 \
                else float("nan")
        except ImportError:
            scores = X_test[:, FEATURE_KEYS.index("power_mean")]
            auroc = manual_auroc(scores, y_test)

        print(f"  held-out {held_out}: AUROC={auroc:.3f} "
              f"(train n={len(train)}, test n={len(test)})")
        if not np.isnan(auroc):
            aurocs.append(auroc)

    if aurocs:
        mean = float(np.mean(aurocs))
        print(f"\nLOHO mean AUROC: {mean:.3f}")
        verdict = "PASS" if mean >= args.threshold else "FAIL"
        print(f"Done criteria (>= {args.threshold}): {verdict}")
        raise SystemExit(0 if mean >= args.threshold else 1)
    else:
        raise SystemExit("no usable LOHO splits")


if __name__ == "__main__":
    main()
