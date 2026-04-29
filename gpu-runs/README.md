# Verifier Challenge — Overnight 2×H100 Capture Plan

**Status:** v0 scaffold. Build off this.
**Hardware:** 3× independent 2×H100 SXM hosts on RunPod, run in parallel (~$5–6/hr each, ~$100 total)
**Window:** ~3.5 hours wall-clock (1h setup + ~2.5h workload + ~30min teardown), all hosts concurrent

---

## Context for picking this up cold

The Verifier Challenge (Jason Hausenloy + Jasmine Li) is a planned open competition — the "DARPA Grand Challenge" for AI verification. The motivation is that AI governance, from lab self-attestation to international treaties between the US and China, depends on being able to verify what's actually running inside an AI datacenter. That capability does not exist today. Compute-governance literature (Shavit's "Catch a Chinchilla," Heim et al. on cloud providers, FlexHEG / Aarne et al. on hardware attestation) lays out the framework but no one has built the verifier itself. The plan is to release a series of challenges, each containing real datacenter telemetry, and let teams compete to build the best open-source verifier.

Jason has already built the schema-level prototype at [jason.ml/inspector](https://jason.ml/inspector). It defines a structured "diary" format (one JSONL record per chip per minute, with FLOPs, tokens, model fingerprints, hash chain) and a treaty commitment YAML. An LLM agent reads both, runs deterministic + semantic rules, and emits one bit (compliant / violation). Jason's adversarial loop ([jason.ml/inspector/redteam](https://jason.ml/inspector/redteam)) ran 53 rounds of Sonnet generating evasions vs Haiku catching them, converging in ~14 protocol patches. **The schema layer is largely solved.**

What's missing is the layer beneath: the schema records assume a *prover-controlled translator* takes raw datacenter telemetry (DCGM counters, NCCL logs, Nsight kernel traces, power, network) and produces schema-conformant records. A dishonest prover can lie at the translator layer while every downstream check still passes. The Verifier Challenge competition's actual research contribution lives at this telemetry → schema layer. To build there, we need a real multi-GPU telemetry dataset with ground-truth labels — Jason's POC was single-GPU on Apple Silicon and explicitly lacks all the multi-GPU signals (NCCL all-reduces, NVLink traffic, fabric topology) that matter for frontier verification.

This document plans the overnight run that produces that dataset. **2×H100** is the smallest GPU configuration that captures real NCCL collectives, which is the single most diagnostic telemetry signal for training-vs-inference. We're not trying to train a useful model — we're producing a labeled telemetry trace that the V1 prototype (translator + cross-checks + agent demo) builds against. The choice of three verification properties (op-type, model-size, training-phase) drives the workload schedule: each property exercises a different telemetry channel, and each is tested both in honest and adversarial conditions where the prover lies. The run schedule is the load-bearing artifact of this plan; everything else (code layout, capture scripts) exists to execute it.

---

## Goal

Capture multi-modal telemetry from a real 2-GPU run with ground-truth-labeled workload phases. Use this dataset to demonstrate that an LLM-as-verifier can detect **three** governance-relevant properties — including under adversarial conditions where the prover lies about the workload.

This is the V1 prototype's input dataset. Everything downstream (translator, cross-checks, agent demo, redaction levels) is built on top of it.

---

## Parallelism strategy

Three RunPod 2×H100 hosts run the **full 8-phase schedule** independently and concurrently. This produces three replicas of the experiment on three physically distinct machines — different driver patches, NCCL builds, thermal envelopes, host-clock skew, neighbor-tenant interference.

The load-bearing constraint: every honest/adversarial *pair* must run on the same host. Splitting (say) phase 2 onto host A and phase 6 onto host B contaminates the verifier-relevant delta with host fingerprint, and you can no longer tell "this signal is the lie" from "this signal is just a different machine." So we don't shard the schedule across hosts — we replicate it.

What the parallel structure buys:
- **Cross-host generalization axis** in the dataset, addressing the n=1 trajectory risk. A verifier that wins on host A's traces but fails on host B's is overfit to host fingerprint, and now we can detect that.
- **Within-run consistency check.** Running phase 1 on three hosts gives the per-property variance floor. Any verifier signal smaller than this is noise.
- **Independent failures.** If host A's DCGM is unprivileged or its dataset download stalls, hosts B and C still produce a usable dataset.

Each host writes to its own `output/runs/<timestamp>-<hostid>/` directory; tarballs are merged at the end. Dataset prep (TinyStories + WikiText tokenization) can run on one host and be rsync'd to the others to save ~10 min × 2 hosts.

Scale to 4+ hosts if budget allows — each additional host is another generalization replica. 3 is the floor for variance estimates; below that, you can't tell signal from host noise.

---

## Properties we verify

Three properties. Each exercises a *different* telemetry channel — so a single-signal verifier can't catch all three at once.

| ID | Property | Telemetry signal | Adversarial form |
|----|----------|------------------|-------------------|
| **P1** | Op type: training vs inference | NCCL all-reduce events/sec, backward-pass kernel time, optimizer-step kernels | Train but log as inference |
| **P2** | Model size ≤ N parameters | HBM allocated per process, FLOP/token ratio (~6N), optimizer state size | Run big model, log as small |
| **P3** | Training phase: pretrain vs fine-tune | LR schedule, gradient norm magnitudes, dataset fingerprint | Fine-tune but log as pretrain |

---

## Models

Two architectures. Both nanoGPT-style GPT-2 variants. Same code, different config.

| Name | Params | n_layer | n_head | n_embd | block_size | batch | Notes |
|------|--------|---------|--------|--------|------------|-------|-------|
| `gpt-tiny` | ~10M | 6 | 6 | 384 | 256 | 64 | Default nanoGPT-ish config |
| `gpt-small` | ~124M | 12 | 12 | 768 | 1024 | 12 | Actual GPT-2 Small architecture |

---

## Datasets

| Name | Source | Size | Used for |
|------|--------|------|----------|
| `tinystories` | HF: `roneneldan/TinyStories` | ~1 GB tokenized | Pretraining (phases 1, 2, 3, 6, 7) |
| `wikitext` | HF: `wikitext-2-raw-v1` | ~13 MB tokenized | Fine-tuning (phases 4, 8) — deliberately different corpus |

Both BPE-tokenized via tiktoken. Prep ~5–10 min total.

---

## Run schedule

Total workload time: **~2h 25min**.

| # | Phase ID | Duration | Op (logged) | Op (true) | Model (real) | Model (logged) | Dataset | LR | Tests |
|---|----------|----------|-------------|-----------|--------------|----------------|---------|-----|-------|
| 1 | `honest_pretrain_tiny` | 30 min | training | training | gpt-tiny | gpt-tiny | tinystories | 6e-4 cosine | P1·P2·P3 baseline |
| 2 | `honest_pretrain_small` | 30 min | training | training | gpt-small | gpt-small | tinystories | 6e-4 cosine | P1·P2 baseline |
| 3 | `honest_inference_small` | 15 min | inference | inference | gpt-small | gpt-small | tinystories | — | P1 baseline |
| 4 | `honest_finetune_tiny` | 20 min | training | training | gpt-tiny (frozen embed) | gpt-tiny | wikitext | 3e-5 const | P3 baseline |
| 5 | `idle` | 5 min | idle | idle | — | — | — | — | confound check |
| 6 | `adv_train_as_infer` | 15 min | **inference** | training | gpt-small | gpt-small | tinystories | 6e-4 cosine | **P1 attack** |
| 7 | `adv_big_as_small` | 15 min | training | training | **gpt-small** | **gpt-tiny** | tinystories | 6e-4 cosine | **P2 attack** |
| 8 | `adv_finetune_as_pretrain` | 15 min | training | training | gpt-tiny (frozen embed) | gpt-tiny (full pretrain) | wikitext | 3e-5 const | **P3 attack** |

**Bold** = the lie. Phases 6–8 are where the prover misrepresents what's happening.

### Phase dependencies & checkpointing

Phases are not independent — three phases load checkpoints produced by earlier ones. The orchestrator must run them in order, on the same host, with a shared `$OUT/checkpoints/` directory.

```
phase 1 (honest_pretrain_tiny)   ── saves ──▶  $OUT/checkpoints/gpt_tiny.pt
                                                       │
                                                       ├─▶ phase 4 (honest_finetune_tiny)         [loads, runs finetune.py]
                                                       └─▶ phase 8 (adv_finetune_as_pretrain)     [loads, runs finetune.py]

phase 2 (honest_pretrain_small)  ── saves ──▶  $OUT/checkpoints/gpt_small.pt
                                                       │
                                                       └─▶ phase 3 (honest_inference_small)       [loads, runs inference.py]

phases 6, 7  ── train from random init; no checkpoint dependency
phase 5      ── idle; no model, no process (see Idle phase notes below)
```

Two consequences:

1. **Phase 4 and phase 8 must invoke `finetune.py`, not `pretrain.py`.** Without that, neither freezes embeddings nor uses the low LR — the "real" workload diverges from the schedule's *Truth* column and the P3 attack collapses to "phase 4 vs phase 8 differ in everything." This was a bug in the v0 orchestrator example; the corrected version below uses `finetune.py` for both.
2. **Each `pretrain.py` / `finetune.py` invocation must accept `--ckpt_in` and `--ckpt_out`.** Pretrain phases save a final checkpoint at the path passed by the orchestrator; finetune and inference phases load from it.

**Idle phase note:** phase 5 must run as a fresh subprocess (or no subprocess at all — just `time.sleep`), so leftover HBM allocations from phase 4's Python process don't contaminate the "idle" telemetry. If you keep the Python process alive across phases, the GPU's HBM utilization in phase 5 reflects phase 4's allocator state, not a real idle baseline.

---

## Code layout

```
inspector-agents/gpu-runs/
├── README.md                       # this doc
├── followup-adv-tests.md           # v0.2 adversarial phase sketches
├── requirements.txt                # torch, numpy, tiktoken, datasets, hf-hub, sklearn
├── .env                            # HF_AUTH_TOKEN (gitignored)
├── capture/
│   └── capture_run.sh              # wraps a workload; starts/stops nvsmi/dcgm/netdev, sets NCCL env
├── workload/
│   ├── model.py                    # GPT-2 architecture (nanoGPT-style, SDPA attention)
│   ├── data.py                     # memmap token dataset
│   ├── _train.py                   # shared train loop (DDP, LR sched, ckpt, signal handler)
│   ├── pretrain.py                 # entry: cosine LR, no freeze
│   ├── finetune.py                 # entry: constant LR, frozen wte/wpe
│   ├── inference.py                # generation loop, no_grad, no collectives
│   ├── prepare_data.py             # tokenize tinystories + wikitext to .npy
│   ├── orchestrator.py             # runs phases 1–8, writes labels + provenance
│   └── configs/
│       ├── gpt_tiny.py             # 6L 384d, ~10M non-embedding params
│       └── gpt_small.py            # 12L 768d, ~124M params (GPT-2 Small)
├── scripts/
│   ├── push_to_hf.py               # gzip nccl logs, upload to HF dataset repo
│   └── smoke_test.py               # LOHO baseline AUROC for P1; run before teardown
└── output/
    └── runs/<timestamp>-<hostid>/
        ├── provenance.json         # host/GPU/python/git/clock at start AND end
        ├── workload_labels.jsonl   # one start/end record per phase, claimed vs truth
        ├── workload.log            # orchestrator stdout/stderr
        ├── checkpoints/            # gpt_tiny.pt, gpt_small.pt (loaded by phases 3, 4, 8)
        └── phases/<phase_id>/
            ├── nvsmi.csv           # 1 Hz GPU power/util/mem/clocks/temp
            ├── dcgm.csv            # 10 Hz tensor active, NVLink TX/RX, etc. (if available)
            ├── netdev.log          # /proc/net/dev @ 1 Hz
            ├── nccl_<host>_<pid>.log  # one per rank
            ├── stdout.log          # workload subprocess stdout
            └── stderr.log          # workload subprocess stderr
```

---

## Environment setup

The plan was silent on this; here it is. Per host, before the orchestrator runs:

**Container image.** Use a RunPod PyTorch template with CUDA 12.x and PyTorch ≥ 2.4 (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). Devel image, not runtime — DCGM install needs `apt`.

**Privileged container.** DCGM requires `--privileged` or at minimum `--cap-add=SYS_ADMIN`. RunPod community pods are usually unprivileged; *secure cloud* pods can be privileged on request. **Verify before you commit the 3-host run** — see smoke test below.

**System packages.** Install DCGM (the daemon, not just the CLI) and a few helpers:

```bash
apt-get update && apt-get install -y \
    datacenter-gpu-manager \
    ca-certificates curl jq pigz
nv-hostengine -b 0.0.0.0   # starts the dcgm daemon
dcgmi dmon -e 1011 -c 1    # smoke-test: must print a real number, not 'permission denied'
```

If `dcgmi dmon` fails, the container is unprivileged. Apply the fallback (skip DCGM, raise nvidia-smi to 10 Hz) and accept losing NVLink TX/RX — and note that this *significantly* weakens the dataset for the canonical translator-layer attack (`adv_subsetted_nccl_log` in `followup-adv-tests.md`), since NVLink TX/RX is its primary cross-check.

**Python deps** (`gpu-runs/requirements.txt`):

```
torch>=2.4.0
numpy
tiktoken
datasets>=2.18
huggingface-hub
```

Install: `pip install -r gpu-runs/requirements.txt`.

PyTorch ships its own NCCL — no separate install. The debug env vars (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=COLL`, `NCCL_DEBUG_FILE=$OUT/nccl_%h_%p.log`) are set by `capture_run.sh`. The `%h_%p` is required so the two ranks on each host write to separate files.

**HF auth** (for the dataset push at the end): `huggingface-cli login` once per host with a write-scoped token. Or set `HF_TOKEN=...` in the env.

**Per-host identity.** The orchestrator reads `RUNPOD_POD_ID` (RunPod sets this automatically) for the `<hostid>` in the output path, falling back to `hostname -s`.

**NCCL log volume.** `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL` over 2.5h of training produces multi-GB logs. Pipe through `pigz` (installed above) or rotate at 100MB to keep the container's tmpfs from filling at 2am:

```bash
mkfifo /tmp/nccl.fifo
pigz -1 < /tmp/nccl.fifo > $OUT/nccl_%h_%p.log.gz &
export NCCL_DEBUG_FILE=/tmp/nccl.fifo
```

(Alternatively just point `NCCL_DEBUG_FILE` straight at a file and `pigz` at the end if you have disk to spare — H100 SXM pods usually have ≥500GB local SSD, so direct write is fine. Compress at teardown.)

**Smoke test before the real run.** Spin up *one* cheap 1×H100 instance (~$1.50/hr), run the v0 capture script for 5 minutes against `gpt-tiny`, and verify all four artifacts populate: `nvsmi.csv`, `dcgm.csv`, `nccl_*.log`, `workload_labels.jsonl`. If anything is empty or missing, debug *there* — not on three concurrent SXM pods at twice the price.

---

## Files to write tonight (priority order)

### 1. `workload/orchestrator.py` — the brain

For each phase: write a `start` marker to `workload_labels.jsonl`, spawn the workload subprocess, wait for duration, write `end` marker.

```python
# workload/orchestrator.py
import json, os, subprocess, time
from pathlib import Path

OUT = Path(os.environ["OUT"])
LABELS = OUT / "workload_labels.jsonl"
CKPTS = OUT / "checkpoints"
CKPTS.mkdir(parents=True, exist_ok=True)

GPT_TINY  = str(CKPTS / "gpt_tiny.pt")
GPT_SMALL = str(CKPTS / "gpt_small.pt")

PHASES = [
    # phase 1: pretrains gpt-tiny → saves checkpoint consumed by phases 4, 8
    {
        "id": "honest_pretrain_tiny", "duration_s": 30 * 60, "script": "pretrain.py",
        "args": ["--config=configs/gpt_tiny.py", "--dataset=tinystories",
                 "--lr=6e-4", f"--ckpt_out={GPT_TINY}"],
        "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
        "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},
    },
    # phase 2: pretrains gpt-small → saves checkpoint consumed by phase 3
    # phase 3: loads gpt_small.pt, runs inference.py
    # phase 4: HONEST FINETUNE — loads gpt_tiny.pt, runs finetune.py (freezes embeds, low LR)
    {
        "id": "honest_finetune_tiny", "duration_s": 20 * 60, "script": "finetune.py",
        "args": ["--config=configs/gpt_tiny.py", "--dataset=wikitext",
                 "--lr=3e-5", f"--ckpt_in={GPT_TINY}"],
        "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
        "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
    },
    # phase 5: idle (handled below — no subprocess)
    # phase 6: adv_train_as_infer — pretrain.py, lie in `claimed`
    # phase 7: adv_big_as_small — pretrain.py with gpt_small config, lie in `claimed.model`
    # phase 8: ADVERSARIAL — IDENTICAL workload to phase 4 (finetune.py + frozen embed +
    # wikitext + LR=3e-5) — only the `claimed` label differs. Forces verifier to detect
    # finetune-shape from telemetry, not from the orchestrator's claim.
    {
        "id": "adv_finetune_as_pretrain", "duration_s": 15 * 60, "script": "finetune.py",
        "args": ["--config=configs/gpt_tiny.py", "--dataset=wikitext",
                 "--lr=3e-5", f"--ckpt_in={GPT_TINY}"],
        "claimed": {"op_type": "training", "model": "gpt-tiny", "phase": "pretrain"},   # LIE
        "truth":   {"op_type": "training", "model": "gpt-tiny", "phase": "finetune"},
    },
]

def mark(phase_id, event, **extra):
    rec = {"ts": time.time(), "phase_id": phase_id, "event": event, **extra}
    with open(LABELS, "a") as f:
        f.write(json.dumps(rec) + "\n")

import signal
def kill_tree(proc):
    """torchrun spawns children; SIGTERM to torchrun alone leaks GPU procs into the next phase."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)

for phase in PHASES:
    mark(phase["id"], "start", claimed=phase["claimed"], truth=phase["truth"])
    if phase["id"] == "idle":
        time.sleep(phase["duration_s"])  # no subprocess; previous phase's process is fully torn down
        mark(phase["id"], "end", status="ok")
        continue
    proc = subprocess.Popen(
        ["torchrun", "--nproc_per_node=2", phase["script"], *phase["args"]],
        cwd="workload",
        preexec_fn=os.setsid,  # so we can signal the whole process group later
    )
    try:
        proc.wait(timeout=phase["duration_s"])
    except subprocess.TimeoutExpired:
        kill_tree(proc)
    status = "ok" if proc.returncode == 0 else f"failed_rc{proc.returncode}"
    mark(phase["id"], "end", status=status)
```

A `failed_rc*` status means the phase's telemetry is unusable; downstream tooling must skip it rather than treat the phase as a successful capture.

### 2. `capture/capture_run.sh` — telemetry wrapper

Already drafted; key collectors:

- `nvidia-smi --query-gpu=... -lms 1000` → `nvsmi.csv`
- `dcgmi dmon -e 1001,1002,...` → `dcgm.csv` (skip if not privileged)
- `/proc/net/dev` poll → `netdev.log`
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_DEBUG_FILE=...` → `nccl_*.log`

### 3. `workload/configs/`

```python
# configs/gpt_tiny.py
n_layer, n_head, n_embd = 6, 6, 384
block_size, batch_size = 256, 64
# ~10M params

# configs/gpt_small.py
n_layer, n_head, n_embd = 12, 12, 768
block_size, batch_size = 1024, 12
# ~124M params (GPT-2 Small)
```

### 4. `workload/prepare_data.py`

```python
from datasets import load_dataset
import tiktoken, numpy as np

enc = tiktoken.get_encoding("gpt2")

# tinystories
ds = load_dataset("roneneldan/TinyStories", split="train[:5%]")  # ~50M tokens, plenty
ids = np.concatenate([enc.encode_ordinary(x["text"]) + [enc.eot_token] for x in ds])
np.save("data/tinystories.npy", ids.astype(np.uint16))

# wikitext  
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ids = np.concatenate([enc.encode_ordinary(x["text"]) for x in ds if x["text"].strip()])
np.save("data/wikitext.npy", ids.astype(np.uint16))
```

### 5. `workload/pretrain.py`, `inference.py`, `finetune.py`

Lift from nanoGPT. Three thin variants:

| File | Differences from base |
|------|----------------------|
| `pretrain.py` | Standard. AdamW + cosine LR. Saves checkpoints. |
| `inference.py` | Loads checkpoint. Generation loop with `torch.no_grad()`. No backward, no optimizer. |
| `finetune.py` | Loads checkpoint. Freezes `transformer.wte`/`wpe`. Constant LR=3e-5. |

---

## Outputs

After the run, the tarball should contain:

| File | Contents |
|------|----------|
| `nvsmi.csv` | per-GPU power, util, memory, clocks at 1 Hz |
| `dcgm.csv` | tensor-core activity, NVLink TX/RX, PCIe TX/RX at 10 Hz *(if DCGM available)* |
| `nccl_*.log` | every collective op (all-reduce, broadcast, ...) with size + timing |
| `netdev.log` | container network counters at 1 Hz |
| `workload_labels.jsonl` | ground truth (claimed vs truth per phase) |
| `workload.log` | orchestrator stdout |
| `provenance.json` | full host / GPU / Python / NCCL / git / clock state — see below |

→ Push to HF Datasets as `your-username/verifier-challenge-traces`.

---

## Provenance & reproducibility

`provenance.json` is the most load-bearing file for benchmark use of this dataset — without it, runs are unreproducible. Write it once at run start, append `clock_offset_end_us` at run end. Schema:

```json
{
  "run_id": "2026-04-29T22:00:00Z-pod_abc123",
  "host": {
    "runpod_pod_id": "pod_abc123",
    "hostname": "ngc-host-42",
    "kernel": "Linux 6.5.0-21-generic",
    "os_release": {"NAME": "Ubuntu", "VERSION_ID": "22.04"},
    "container_image_digest": "sha256:..."
  },
  "gpu": {
    "name": "NVIDIA H100 80GB HBM3",
    "device_count": 2,
    "driver_version": "550.54.15",
    "cuda_version": "12.4",
    "nccl_version": [2, 21, 5],
    "uuids": ["GPU-...", "GPU-..."]
  },
  "python": {"version": "3.11.8", "torch": "2.4.0", "tiktoken": "0.5.2", "datasets": "2.18.0"},
  "code":   {"git_commit": "<repo hash>", "git_dirty": false},
  "clock_offset_start_us": 423,   // chronyc tracking, run start
  "clock_offset_end_us":   501,   // chronyc tracking, run end
  "phases": [
    {"id": "honest_pretrain_tiny", "argv": [...], "seed": 1337, "rc": 0, "duration_actual_s": 1801.4},
    ...
  ]
}
```

Helper (lift into `orchestrator.py`):

```python
import os, sys, json, hashlib, subprocess, torch
def collect_provenance():
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
        text=True).strip().splitlines()
    chrony = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True).stdout
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "gpu": {
            "name": smi[0].split(",")[0].strip(),
            "driver_version": smi[0].split(",")[1].strip(),
            "uuids": [l.split(",")[2].strip() for l in smi],
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()),
            "device_count": torch.cuda.device_count(),
        },
        "python": {"version": sys.version.split()[0], "torch": torch.__version__},
        "host": {"runpod_pod_id": os.environ.get("RUNPOD_POD_ID", ""),
                 "hostname": os.uname().nodename},
        "code":  {"git_commit": git},
        "chrony_tracking_raw": chrony,
    }
```

**Random seed policy.** Each phase gets a deterministic seed: `seed = hash(phase_id) % 2**31`, applied at start of `pretrain.py` / `finetune.py` / `inference.py` to `torch.manual_seed`, `numpy.random.seed`, and `random.seed`. The same phase on different hosts starts with the same RNG state. **Do NOT enable `torch.use_deterministic_algorithms(True)`** — it kills H100 perf and isn't what we want. Identical-seed runs across hosts diverge from cuDNN/cuBLAS nondeterminism, NCCL ordering, and timing jitter. *That divergence is exactly the host-fingerprint signal we're trying to isolate;* randomizing seeds per host would entangle host signal with workload randomness and the cross-host generalization test loses precision.

**Time sync.** Run `chronyc tracking` at run start AND run end; record both `clock_offset_*_us` into `provenance.json`. RunPod NTP skew can be 100ms+. End-of-run reading detects in-run drift; if it shifted >1s, flag on the dataset card.

---

## Fallbacks (decide at smoke-test, not at 3am)

| Fallback | Trigger |
|----------|---------|
| Drop **P2**, substitute run-fragmentation detection | Dual-model logging is fiddly OR DCGM unavailable |
| Drop adversarial phases 6–8 | Smoke test ate >2h; ship honest-only |
| 1×H100 PCIe instead of 2×H100 SXM | 2×H100 SXM unavailable on Community Cloud |
| Skip DCGM, raise nvidia-smi to 10 Hz | Container not privileged |
| TinyStories only, drop WikiText | Data prep takes too long |

---

## "Done" criteria

Successful night = merged tarball with **at minimum**:

- [ ] `nvsmi.csv` covering full run, both GPUs, **on at least 2 of 3 hosts**
- [ ] `nccl_*.log` from at least one training phase per surviving host (the existence-of-collectives proof)
- [ ] `workload_labels.jsonl` covering all phases that ran, per host
- [ ] `provenance.json` per host, populated per the schema above (including `clock_offset_*_us`)
- [ ] Same-night smoke test passes (see *Evaluation protocol* below)
- [ ] Pushed to HF Datasets with per-host directories preserved, link works

If only 1 of 3 hosts survives, the dataset still ships but the cross-host axis is gone — flag this in the dataset card. Anything beyond is bonus. **Don't** scope-creep at 2am.

### Evaluation protocol

The whole point of running 3 hosts in parallel is to make cross-host generalization measurable. So the dataset's primary evaluation must reflect that — **leave-one-host-out**, not pooled.

For each property P ∈ {P1, P2, P3}:

1. Train a baseline classifier on data from 2 hosts (all phases included).
2. Test on the held-out host.
3. Repeat for all 3 splits; report mean AUROC and per-split spread.

Per-host evaluation (train and test on the same host) is trivial — a verifier learns host fingerprint and wins. The cross-host floor is the actual scientific bar.

**Same-night smoke test:** dumb baseline (mean nvsmi power per phase, logistic regression) on P1, leave-one-host-out, mean AUROC ≥ 0.7. If this fails, the data isn't separable with simple features and we need to debug *before* tarballing — not the morning after.

If a verifier's pooled AUROC is much higher than its leave-one-host-out AUROC, it's overfit to host fingerprint. The gap between the two is itself a useful number to publish.

---

## Dataset card outline (for HF push)

Fill before pushing. Lives at `dataset_card.md` in the HF repo root.

- **Title:** Verifier Challenge Traces v0.1 — 2×H100 multi-host telemetry with adversarial labels
- **Motivation:** ground-truth-labeled telemetry for AI workload verification; companion to the Inspector Agents schema layer at jason.ml/inspector
- **Schema:** one directory per host (`runs/<timestamp>-<hostid>/`) containing `nvsmi.csv`, `dcgm.csv` (if available), `nccl_*.log.gz`, `netdev.log`, `workload_labels.jsonl`, `workload.log`, `provenance.json`
- **Phases:** 8 per host — 5 honest, 3 adversarial. Honest/adversarial pairs share host (load-bearing — see *Parallelism strategy* in source README). Adversarial subset is ~25% of run time; benchmark numbers should be class-balanced or report the imbalance
- **Evaluation:** primary metric is leave-one-host-out AUROC (see *Evaluation protocol* in source README). Pooled-AUROC results should be reported alongside the LOHO number, not in place of it
- **Source data not redistributed:** TinyStories (CDLA-Sharing 1.0) and WikiText-2 (CC-BY-SA 4.0) are *not* shipped — only telemetry and labels. Users re-download the corpora from HF if they want to reproduce
- **Caveats:**
  - 2×H100 single-node — no multi-node fabric, no InfiniBand. NCCL traffic is NVLink P2P, not the topology a frontier verifier will encounter
  - n=3 hosts is the variance floor; results may not generalize beyond RunPod's H100 SXM pool
  - v0.1 adversarial phases mostly perturb workload/labels; deeper translator-layer tampering is in `followup-adv-tests.md` and shipped in v0.2
  - If any host's `clock_offset_end_us` shifted >1s during the run, that host is flagged unreliable for cross-host temporal alignment
- **License:** CC-BY 4.0 for our telemetry + labels. Users assume the licensing of any source corpora they re-download.