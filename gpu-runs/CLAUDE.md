# gpu-runs — Verifier Challenge overnight 2×H100 capture

Operational notes for the overnight run. The full plan is in `README.md`; this
file is for state that's only useful while a run is in flight or being prepped.

## How to launch the real run on this host

Pre-flight is already done (deps installed, datasets tokenized, smoke 8/8 ok).
To launch the full schedule:

```bash
cd /workspace/inspector-agents/gpu-runs
nohup python workload/orchestrator.py > /tmp/run.out 2>&1 &
```

Wall time ≈ 2h 26min: 145 min workload (per `README.md` schedule) + ~11 s
teardown × 7 non-idle phases. Output lands in
`output/runs/<utc-ts>-<RUNPOD_POD_ID>/`.

After it finishes:

```bash
python scripts/push_to_hf.py --dry-run     # validates gzip + token wiring
python scripts/smoke_test.py               # LOHO needs >=2 hosts; 1-host runs warn
python scripts/push_to_hf.py               # real upload
```

## Cross-host coordination

Three hosts run the full 8-phase schedule independently and concurrently. They
share a network volume — there's no rsync/scp step; every host sees the same
files at the same path. `RUNPOD_POD_ID` segregates output dirs.

What each host needs before launching its orchestrator:

1. `apt-get install -y datacenter-gpu-manager pigz chrony` (chrony optional —
   see "what's available" below)
2. `pip install -r gpu-runs/requirements.txt`
3. `python gpu-runs/workload/prepare_data.py` (idempotent; one host can do it
   first and the other two will skip on the shared volume)
4. `gpu-runs/.env` with `HF_AUTH_TOKEN` and `WANDB_API_KEY` (already present)
5. `nohup python workload/orchestrator.py > /tmp/run.out 2>&1 &`

The HF push step runs once across all hosts — `scripts/push_to_hf.py` walks
`output/runs/` and uploads each per-host subdir. Run it after all three
orchestrators have exited.

## What's available on this host (and what isn't)

This is the **first host** (RunPod pod `j4vei3duw66yfd`).

| Component | Status | Notes |
|-----------|--------|-------|
| 2× H100 80GB | ✅ | torch 2.4.1+cu124, NCCL 2.20.5 |
| nvidia-smi @ 1 Hz | ✅ | full set of fields |
| DCGM profiling | ⚠️ partial | GR_ENGINE_ACTIVE / SM_ACTIVE / SM_OCCUPANCY / PCIE TX/RX work; **NVLink TX/RX = N/A**, TENSOR_ACTIVE / DRAM_ACTIVE = 0 (unprivileged container) |
| chronyd | ❌ | unprivileged, can't get CAP_SYS_TIME — `clock_offset_*_us` will be `null` in `provenance.json`. The raw `chronyc tracking` capture will record an error string instead |
| HF + wandb tokens | ✅ | loaded via `gpu-runs/.env` |
| Disk | ✅ | 316 TB free on shared volume |

The DCGM gap matters: NVLink TX/RX is the canonical translator-layer cross-check
for the `adv_subsetted_nccl_log` attack in `followup-adv-tests.md`. Without it,
P1 (training vs. inference) is still detectable from GR_ENGINE_ACTIVE + power
draw + the existence of `all_reduce` ops in `nccl_*.log`, but the strongest
adversarial test is weakened. Note this in the dataset card.

If the other two hosts run privileged, their NVLink TX/RX will be present —
that asymmetry across hosts is itself a useful signal but should be flagged
in the per-host `dcgm.skipped` markers / dataset card.

## Patches I made on top of the v0 scaffold

1. **DDP self-stop deadlock fix** in `workload/_train.py` — see commit message
   for full context. Without this every training phase hangs at NCCL teardown
   until the orchestrator's hard timeout (210s/phase). Caught in the first
   smoketest. Ranks now `all_reduce(MAX)` a stop flag every 5 iters.

2. **wandb logging** — rank 0 only, gated on `WANDB_API_KEY`. Project defaults
   to `verifier-challenge-traces`; group = orchestrator's `run_id` so all 8
   phases of a host land in one group and the cross-host view is one filter.

## Smoke test (already validated)

Ran the full 8-phase schedule with each phase shortened to 60s (idle = 30s) via
`/tmp/smoke_orchestrator.py` (a wrapper that monkeypatches `build_phases`).
Don't ship that wrapper to the real run — it pulls into the
`verifier-challenge-traces-smoke` wandb project so the smoke runs don't
contaminate the real charts.

Result: **8/8 phases ok**, all stopped within 11s of `duration_s`. Validates:
- DDP self-stop fix (no NCCL deadlock)
- Checkpoint dependency chain (phase 3 loads gpt_small from 2; phases 4 & 8
  load gpt_tiny from 1)
- Idle phase teardown (no leftover Python process bleeding into next phase's
  HBM baseline)
- All 4 entrypoints (pretrain.py, finetune.py, inference.py, idle/no-op)
- wandb wiring on both training and inference

Per-phase artifact sizes (60s phase): nvsmi.csv 14 KB, dcgm.csv 271 KB,
NCCL log 0.9 MB/rank. Scaling to a 30-min phase: ~8 MB nvsmi + ~135 MB dcgm +
~25 MB NCCL/rank. Whole-run footprint ≈ 1 GB pre-gzip; well under the
network-volume budget.

## Don't do this

- Don't enable `torch.use_deterministic_algorithms(True)` — it kills H100 perf
  and the README explicitly wants per-host nondeterminism (the cross-host
  signal is the point of replicating).
- Don't skip the `find_unused_parameters=True` for finetune phases — tied
  `lm_head`/`wte` + frozen `wte` makes DDP see an unused parameter on backward
  and the run will hang. (`_train.py` already handles this; if you refactor,
  preserve it.)
- Don't `git add gpu-runs/.env` — the token leaks. `.gitignore` excludes it,
  but verify before any `git add -A`.
- Don't run `scripts/push_to_hf.py` until all 3 orchestrators have exited —
  uploading a partial run-dir is fine but means a second push when the host
  finishes, doubling traffic and risking rate limits.
