"""Orchestrator — runs the 8-phase schedule on this host.

Each non-idle phase is wrapped by capture/capture_run.sh, which starts/stops
collectors per-phase. Provenance is captured at start and end (so clock_offset
drift across the run can be measured). Failed phases get status=failed_rc*.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GPU_RUNS = Path(__file__).resolve().parent.parent
WORKLOAD = GPU_RUNS / "workload"
CAPTURE_RUN = GPU_RUNS / "capture" / "capture_run.sh"


def stable_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def hostid() -> str:
    return os.environ.get("RUNPOD_POD_ID") or socket.gethostname().split(".")[0]


def parse_chrony_offset_us(text: str | None) -> float | None:
    """Pull 'Last offset' from `chronyc tracking` output, return microseconds."""
    if not text:
        return None
    for line in text.splitlines():
        if "Last offset" in line:
            # e.g. "Last offset     :  +0.000423 seconds"
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                return float(val) * 1e6
            except (IndexError, ValueError):
                return None
    return None


def collect_provenance() -> dict:
    out: dict = {"python": {}, "gpu": {}, "host": {}, "code": {}}
    out["python"]["version"] = sys.version.split()[0]
    try:
        import torch
        out["python"]["torch"] = torch.__version__
        out["gpu"]["cuda_version"] = torch.version.cuda
        try:
            out["gpu"]["nccl_version"] = list(torch.cuda.nccl.version())
        except Exception as e:
            out["gpu"]["nccl_version_error"] = str(e)
        out["gpu"]["device_count"] = torch.cuda.device_count()
    except Exception as e:
        out["python"]["torch_error"] = str(e)

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        out["gpu"]["name"] = smi[0].split(",")[0].strip()
        out["gpu"]["driver_version"] = smi[0].split(",")[1].strip()
        out["gpu"]["uuids"] = [l.split(",")[2].strip() for l in smi if l.strip()]
    except Exception as e:
        out["gpu"]["nvidia_smi_error"] = str(e)

    try:
        out["host"]["chrony_tracking_raw"] = subprocess.check_output(
            ["chronyc", "tracking"], text=True, timeout=5
        )
    except Exception as e:
        out["host"]["chrony_error"] = str(e)

    out["host"]["runpod_pod_id"] = os.environ.get("RUNPOD_POD_ID", "")
    out["host"]["hostname"] = socket.gethostname()
    try:
        un = os.uname()
        out["host"]["uname"] = {"sysname": un.sysname, "release": un.release, "machine": un.machine}
    except Exception:
        pass

    try:
        os_release: dict = {}
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                os_release[k] = v.strip().strip('"').strip("'")
        out["host"]["os_release"] = os_release
    except Exception:
        pass

    try:
        out["code"]["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(GPU_RUNS), text=True, timeout=5
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=str(GPU_RUNS), timeout=5
        ).returncode
        out["code"]["git_dirty"] = bool(dirty)
    except Exception as e:
        out["code"]["git_error"] = str(e)

    return out


def build_phases(out_dir: Path) -> list[dict]:
    ckpts = out_dir / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    GPT_TINY = str(ckpts / "gpt_tiny.pt")
    GPT_SMALL = str(ckpts / "gpt_small.pt")

    cfg_tiny = "workload/configs/gpt_tiny.py"
    cfg_small = "workload/configs/gpt_small.py"

    return [
        {
            "id": "honest_pretrain_tiny", "duration_s": 30 * 60, "script": "pretrain.py",
            "args": [f"--config={cfg_tiny}", "--dataset=tinystories",
                     "--lr=6e-4", f"--ckpt_out={GPT_TINY}"],
            "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
            "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
        },
        {
            "id": "honest_pretrain_small", "duration_s": 30 * 60, "script": "pretrain.py",
            "args": [f"--config={cfg_small}", "--dataset=tinystories",
                     "--lr=6e-4", f"--ckpt_out={GPT_SMALL}"],
            "claimed": {"op_type": "training", "model": "gpt-small", "phase": "pretrain"},
            "truth":   {"op_type": "training", "model": "gpt-small", "phase": "pretrain"},
        },
        {
            "id": "honest_inference_small", "duration_s": 15 * 60, "script": "inference.py",
            "args": [f"--config={cfg_small}", "--dataset=tinystories",
                     f"--ckpt_in={GPT_SMALL}"],
            "claimed": {"op_type": "inference", "model": "gpt-small", "phase": "inference"},
            "truth":   {"op_type": "inference", "model": "gpt-small", "phase": "inference"},
        },
        {
            "id": "honest_finetune_tiny", "duration_s": 20 * 60, "script": "finetune.py",
            "args": [f"--config={cfg_tiny}", "--dataset=wikitext",
                     "--lr=3e-5", f"--ckpt_in={GPT_TINY}"],
            "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
            "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
        },
        {
            "id": "idle", "duration_s": 5 * 60, "script": None, "args": [],
            "claimed": {"op_type": "idle"},
            "truth":   {"op_type": "idle"},
        },
        {
            "id": "adv_train_as_infer", "duration_s": 15 * 60, "script": "pretrain.py",
            "args": [f"--config={cfg_small}", "--dataset=tinystories", "--lr=6e-4"],
            "claimed": {"op_type": "inference", "model": "gpt-small", "phase": "inference"},
            "truth":   {"op_type": "training", "model": "gpt-small", "phase": "pretrain"},
        },
        {
            "id": "adv_big_as_small", "duration_s": 15 * 60, "script": "pretrain.py",
            "args": [f"--config={cfg_small}", "--dataset=tinystories", "--lr=6e-4"],
            "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
            "truth":   {"op_type": "training", "model": "gpt-small", "phase": "pretrain"},
        },
        {
            "id": "adv_finetune_as_pretrain", "duration_s": 15 * 60, "script": "finetune.py",
            "args": [f"--config={cfg_tiny}", "--dataset=wikitext",
                     "--lr=3e-5", f"--ckpt_in={GPT_TINY}"],
            "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
            "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
        },
    ]


def write_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_phase(phase: dict, phase_dir: Path) -> tuple[int, str]:
    """Returns (rc, status). Idle handled by caller; this is for non-idle phases."""
    torchrun_cmd = [
        "torchrun", "--nproc_per_node=2", "--standalone",
        str(WORKLOAD / phase["script"]),
        *phase["args"],
        f"--phase_id={phase['id']}",
        f"--duration_s={phase['duration_s']}",
    ]
    cmd = [str(CAPTURE_RUN), str(phase_dir), *torchrun_cmd]
    print(f"\n=== phase {phase['id']} (duration_s={phase['duration_s']}) ===", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)

    # Hard timeout = duration_s × 1.5 + 120s — generous safety net so
    # --duration_s self-stop is the normal path; killpg only fires if a phase hangs.
    hard_timeout = phase["duration_s"] * 1.5 + 120

    try:
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid, cwd=str(GPU_RUNS))
    except Exception as e:
        return -1, f"failed_spawn:{e}"

    try:
        rc = proc.wait(timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        print(f"phase {phase['id']} exceeded hard timeout, killing process group", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        rc = proc.returncode if proc.returncode is not None else -9

    status = "ok" if rc == 0 else f"failed_rc{rc}"
    return rc, status


def main() -> None:
    load_env(GPU_RUNS / ".env")

    out_root = Path(os.environ.get("OUT_ROOT", str(GPU_RUNS / "output")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{hostid()}"
    out_dir = out_root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    phases_dir = out_dir / "phases"
    phases_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "workload_labels.jsonl"
    workload_log = out_dir / "workload.log"

    # Redirect fd 1/2 at the OS level so child processes inherit. Python's
    # existing sys.stdout/sys.stderr already wrap fd 1/2 — after dup2, their
    # writes route through to workload.log. capture_run.sh redirects the
    # wrapped command's stdout/stderr separately to per-phase log files, so
    # only its pre-command echos land here (along with orchestrator prints).
    log_fh = open(workload_log, "a", buffering=1)
    os.dup2(log_fh.fileno(), 1)
    os.dup2(log_fh.fileno(), 2)
    log_fh.close()

    print(f"=== orchestrator start, run_id={run_id} ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    prov = collect_provenance()
    prov["run_id"] = run_id
    prov["start_iso"] = datetime.now(timezone.utc).isoformat()
    chrony_start = prov.get("host", {}).pop("chrony_tracking_raw", None)
    prov["clock_tracking_start_raw"] = chrony_start
    prov["clock_offset_start_us"] = parse_chrony_offset_us(chrony_start)
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    phases = build_phases(out_dir)
    phases_meta: list[dict] = []

    for phase in phases:
        phase_dir = phases_dir / phase["id"]
        phase_dir.mkdir(parents=True, exist_ok=True)
        seed = stable_seed(phase["id"])
        write_jsonl(labels_path, {
            "ts": time.time(),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "phase_id": phase["id"], "event": "start",
            "claimed": phase["claimed"], "truth": phase["truth"], "seed": seed,
            "duration_s_target": phase["duration_s"],
        })

        t_start = time.time()
        if phase["script"] is None:  # idle
            time.sleep(phase["duration_s"])
            rc, status = 0, "ok"
        else:
            rc, status = run_phase(phase, phase_dir)
        elapsed = time.time() - t_start

        write_jsonl(labels_path, {
            "ts": time.time(),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "phase_id": phase["id"], "event": "end",
            "status": status, "rc": rc, "duration_actual_s": elapsed,
        })
        phases_meta.append({"id": phase["id"], "rc": rc, "status": status,
                            "seed": seed, "duration_actual_s": elapsed})
        print(f"phase {phase['id']}: status={status} rc={rc} actual={elapsed:.0f}s", flush=True)

    # Re-collect provenance to capture end-of-run clock tracking
    prov_end = collect_provenance()
    chrony_end = prov_end.get("host", {}).get("chrony_tracking_raw")
    prov["end_iso"] = datetime.now(timezone.utc).isoformat()
    prov["clock_tracking_end_raw"] = chrony_end
    prov["clock_offset_end_us"] = parse_chrony_offset_us(chrony_end)
    prov["phases"] = phases_meta
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    n_ok = sum(1 for p in phases_meta if p["rc"] == 0)
    print(f"\n=== orchestrator done, {n_ok}/{len(phases_meta)} phases ok ===", flush=True)


if __name__ == "__main__":
    main()
