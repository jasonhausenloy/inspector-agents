# What this run captures despite the unprivileged container

Quick reference for what we lose when the RunPod container is unprivileged
(no `CAP_SYS_ADMIN`, no `CAP_SYS_TIME`) and what we still get. Links the
`README.md` "Fallbacks" table to a concrete decision.

## What's missing

| Signal | Why missing | What it would have helped |
|--------|-------------|---------------------------|
| **DCGM NVLink TX/RX** (1011, 1012) | Profiling-counter access requires admin on H100 | Cross-check for `adv_subsetted_nccl_log` (translator-layer attack) — a prover that lies in the NCCL log can be caught by NVLink bytes-on-wire mismatch |
| **DCGM `TENSOR_ACTIVE`, `DRAM_ACTIVE`** (1004, 1005) | Same privilege class | Tensor-core utilization signal; DRAM bandwidth |
| **`chronyc tracking`** (clock_offset_*_us) | `chronyd` needs `CAP_SYS_TIME` | Sub-second cross-host timestamp alignment; in-run drift detection |

## What this run still captures (and a single-GPU local box can't)

The verification dataset's primary value lives at the **multi-GPU NCCL layer**,
not at NVLink TX/RX specifically.

| Signal | Captured? | Why it matters |
|--------|-----------|----------------|
| `nccl_*.log` (per-rank, `NCCL_DEBUG=INFO,COLL,INIT`) | ✅ | Every `all_reduce`, `broadcast`, init handshake, with size + timing. **This is the single most diagnostic signal for training-vs-inference.** Logged by PyTorch in-process, no admin needed. Single-GPU = empty. |
| DDP gradient-sync signature in DCGM/nvsmi | ✅ | Coordinated power/util spikes on both GPUs during backward, absent during inference. Bucket sync rhythm. Not reproducible on 1 GPU. |
| DCGM `GR_ENGINE_ACTIVE`, `SM_ACTIVE`, `SM_OCCUPANCY` (1001-1003) | ✅ | Per-GPU compute activity; works unprivileged. Anchors the P1/P2/P3 detection. |
| DCGM PCIe TX/RX (1009, 1010) | ✅ | Inter-GPU traffic (fallback path when NVLink saturates). |
| nvsmi power.draw, util, mem, clocks @ 1 Hz | ✅ | Two coordinated time series — host fingerprint + workload signature. |
| H100-specific Hopper SM counters | ✅ | Not present on consumer cards. |
| SXM topology, RunPod kernel/driver/NCCL build, tenant interference | ✅ | Datacenter realism, hard to replicate locally. |

## Net effect on the dataset

- **P1 / P2 / P3 baseline detection**: unaffected. GR_ENGINE + power + NCCL log
  is enough.
- **`adv_subsetted_nccl_log` attack** (one of three v0.2 adversarial tests):
  detection harder without NVLink TX/RX as second source. Still possible via
  PCIe TX/RX cross-check or NCCL-log-internal consistency, just less direct.
- **Cross-host temporal alignment**: in-run drift untracked. Flag the host as
  "clock_offset unknown" in the dataset card per the README convention.

## If we can get a privileged pod for any host

`--privileged` or at least `--cap-add=SYS_ADMIN` (RunPod secure-cloud, on
request). One privileged host out of three is enough to give us NVLink TX/RX
on at least one trace — that asymmetry is itself a useful axis (LOHO eval
already designed for cross-host generalization). Not a blocker for the v0
dataset, but worth asking for if a host is being respun.
