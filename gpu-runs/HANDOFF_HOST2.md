# Host 2 handoff — Verifier Challenge overnight run

You're **host 2 of 3**. Hosts 1, 2, 3 all run the same 8-phase schedule on
their own 2×H100 pod, concurrently. They share the network volume at
`/workspace/inspector-agents`, so paths are identical to host 1 and the
tokenized datasets prepared on host 1 are already there. `RUNPOD_POD_ID`
segregates each host's output dir.

See `gpu-runs/README.md` for the schedule and rationale, and
`gpu-runs/CLAUDE.md` for the operational details host 1 already worked through
(DCGM unprivileged → NVLink TX/RX = N/A, chronyd unavailable → null
clock_offset, all gracefully handled).

## Pre-flight (~5 min)

```bash
cd /workspace/inspector-agents
git pull                                        # gets host 1's deadlock fix
apt-get update && apt-get install -y \
    datacenter-gpu-manager pigz chrony          # chrony optional
nv-hostengine -b 0.0.0.0                        # starts DCGM hostengine
pip install -r gpu-runs/requirements.txt
```

`prepare_data.py` is **idempotent** — host 1 already produced
`gpu-runs/data/{tinystories,wikitext}.npy` on the shared volume so you can skip
it. If `gpu-runs/data/*.npy` is missing, run `python gpu-runs/workload/prepare_data.py`.

`gpu-runs/.env` (with `HF_AUTH_TOKEN` and `WANDB_API_KEY`) is also on the shared
volume — no setup needed.

## Launch

```bash
cd /workspace/inspector-agents/gpu-runs
nohup python workload/orchestrator.py > /tmp/run.out 2>&1 &
echo "orchestrator pid: $!"
```

Wall ≈ **2h 26min**. Output lands in
`gpu-runs/output/runs/<utc-ts>-$RUNPOD_POD_ID/`.

## Watch

```bash
tail -f gpu-runs/output/runs/*-$RUNPOD_POD_ID/workload.log
```

Each phase prints `phase <id>: status=ok rc=0 actual=<s>` on success. Any
`failed_rcN` is non-fatal — the orchestrator continues to the next phase. The
"done criteria" needs ≥2 of 3 hosts surviving, so a partial run on host 2 is
still useful.

## Don't push to HF

`scripts/push_to_hf.py` walks `gpu-runs/output/runs/` and uploads every host's
dir, so it should run **once** after all three hosts have finished. **Don't run
it from host 2** — host 1 will run it.

## If your orchestrator crashes

Re-launch from scratch (the orchestrator doesn't natively resume):
```bash
nohup python workload/orchestrator.py > /tmp/run.out 2>&1 &
```
The new run gets a fresh `<utc-ts>-$RUNPOD_POD_ID` dir and the partial one is
left in place — `push_to_hf.py` will upload both, and the dataset card should
note which one is canonical for host 2 (use the one that's complete).
