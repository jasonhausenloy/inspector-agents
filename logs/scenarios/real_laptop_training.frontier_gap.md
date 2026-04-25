# What this trace captures vs a real frontier training run

**This trace** (laptop, MPS, ~3.2M-param GPT, Tiny Shakespeare):

Per record (one per training step, 60ms cadence):
- Identity: chip_id, job_id, operator, cluster_region, hash-chained record_id
- Timing: window start/end, data-load / forward / backward / optimizer ms
- Compute: FLOPs (estimated), tokens, batch×seq, tokens/sec
- Optimization: loss, grad_norm (clipped), per-group grad norms (embed / blocks / head / ln)
- Model state: model_hash_prefix (refreshed every 50 steps), dataset_fingerprint, config_hash, code_commit
- Provenance: upstream_refs (chains), data_source_tags
- System: process RSS, CPU%, system mem%, load avg, thread count, MPS allocated bytes, net I/O counters, disk I/O counters (when permitted)
- Heavy (every 50 steps): per-group weight norms, Adam m/v second-moment norms, top-5 largest tensors with min/max/mean/std

One-time sidecar (`real_laptop_training.system_fingerprint.json`):
- Hostname, platform, machine, system, release
- Python version + implementation, torch version
- CPU logical/physical count, max freq
- RAM total
- MPS available/built, CUDA available

## What a frontier (e.g. GPT-5-scale) training trace would also have

### Per chip per step (we don't capture)
- **DCGM telemetry**: SM_ACTIVE, PIPE_TENSOR_ACTIVE (the tensor-core busy fraction — Acme paper §4 shows this is the closest signal to "what fraction of theoretical FLOPs were actually achieved"), MEM_COPY_UTIL, FB_USED/FREE
- **Per-GPU power draw** (W) — from nvidia-smi or DCGM
- **Per-GPU thermal headroom** (°C)
- **NVLink / NVSwitch link utilization** — for tensor-parallel partners
- **PCIe traffic** (bytes/s) — for CPU↔GPU offload
- **Clock frequencies** (boost throttling indicator)
- **ECC error counters** (single-bit, double-bit) — fault telemetry

### Per step, multi-host (we don't capture — single chip)
- **All-reduce latency** for gradient sync
- **All-gather / reduce-scatter** for ZeRO partitioning
- **Pipeline-parallel bubble time** (forward / backward stalls)
- **Per-rank loss / grad_norm** (we have per-process; frontier has per-rank × hundreds of ranks)
- **Network fabric counters** (InfiniBand RDMA bytes, link errors)
- **Stragglers**: which rank was slowest each iteration

### Per step, content (we partially capture)
- **Per-tensor stats**: min/max/mean/std, kurtosis, fraction-of-zero — for every parameter (we sample top-5 every 50 steps; frontier wants every tensor every step or at least every N steps)
- **Activation statistics**: per-layer activation norms, kurtosis (we don't capture activations at all — would need forward hooks)
- **Gradient histograms** per parameter (we capture norms; frontier would dump bin counts)
- **Dropout / random seeds** at each step (for replayability)
- **Loss components** (LM loss + KL + auxiliary losses for MoE / etc)

### Periodic (we don't capture)
- **Checkpoint hashes** at every save (every 1000 steps usually)
- **Validation loss** on held-out set (every 500-1000 steps)
- **Sample generations** (qualitative model probe)
- **Throughput at varying batch sizes** (calibration)
- **Failed step retries** (when grad-norm spikes too high → skip step)

### One-time start (we capture some)
- **Hardware inventory**: every GPU's serial number, PCIe slot, NVLink topology, RAM type/speed
- **Network topology**: rack assignment, switch hierarchy, RDMA queue-pair config
- **Software stack**: CUDA version, NCCL version, kernel image, container hash, all dep versions (pip freeze hash)
- **Dataset Merkle root**: per-shard hashes for the whole training corpus (we have one fingerprint of the loaded text; frontier needs per-shard verifiability)
- **Training-config hash**: hyperparameters, parallelism strategy, optimizer config (we capture)
- **Model architecture hash**: forward-pass graph (we proxy via param fingerprint)

### What requires elevated permissions / external systems
- **Power consumption (kWh per step)** — `powermetrics` on Mac needs sudo; in datacenter, Redfish API
- **Air handler / liquid-cooling stats** — facility BMS, separate stream
- **Carbon intensity** of the powering grid — joined via location + timestamp
- **Networking layer below RDMA** — switch counters, packet-loss

## So how realistic is this trace?

**On a per-chip basis** (one chip, one step), our trace captures roughly **70-80%** of what an auditor needs. The big gaps at chip level: DCGM telemetry (we'd want it but Apple Silicon doesn't expose equivalents publicly) and power draw.

**On a multi-host basis**, we capture **~10%** — we have one chip. The 90% that's missing is exactly what makes frontier training hard to verify: gradient sync, fabric topology, straggler detection, fault rates. That's the whole "it scales" question.

**On the content side** (per-tensor / per-layer detail), we capture **~30%** — we have top-5 param stats every 50 steps and per-group gradient norms every step. Frontier auditors would want all-tensor histograms and activation hooks.

## What's deliberately missing here that matters

1. **No forward hooks / activation logging.** Capturing activations would 10-100× the bytes per step. Real frontier setups sample selectively (e.g., one layer's activations every 100 steps).
2. **No checkpoint cryptographic commits.** We refresh `model_hash_prefix` every 50 steps. A real protocol would issue a Merkle commitment of the entire model state every checkpoint and require the prover to keep snapshots.
3. **No validation loss.** We log training loss only. A treaty-grade trace would alternate train/val with separate verifiable hashes.
4. **No multi-rank causal links.** Our `upstream_refs` is a simple chain. A multi-host run would have a DAG: each rank's gradient at step N depends on all ranks' gradients at step N-1 via all-reduce.

These gaps are what `~/Desktop/inspector-agents/_obsidian/rent-compute-plan.md` (the rent-compute plan) costs out: each gap is roughly $X of engineering and $Y of compute to close at a chosen scale.
