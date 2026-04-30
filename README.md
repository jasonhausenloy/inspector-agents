# Inspector Agents

A working prototype of automated AI compute verification, built in two halves:

- **Laptop scale.** A 3.2M-parameter GPT trained end-to-end on Apple Silicon (MPS), with every training step logged in a structured schema. A scaffolded Claude agent reads the logs and emits a verdict — training vs inference, FLOP-budget compliance — with a stated confidence and cited signals.
- **Multi-host scale.** A pair of H100 SXM nodes on RunPod running an 8-phase schedule of honest workloads (pretraining, fine-tuning, inference, idle) and adversarial ones (training declared as inference, the larger model claimed as the smaller one, fine-tuning claimed as fresh pretraining). Six telemetry streams per phase: `nvsmi`, DCGM, NCCL logs, `/proc/net/dev`, checkpoints, provenance. Ground-truth labels released alongside the prover's claimed labels. **Public dataset:** [`jasminexli/verifier-challenge-traces`](https://huggingface.co/datasets/jasminexli/verifier-challenge-traces) on Hugging Face, CC-BY-4.0.

Both halves are the smallest working version of the **Verifier Challenge** — an open competition we (Jason Hausenloy + Jasmine Li) are proposing: build the best system for inferring what's running inside a datacenter from its logs. The framing is an [Air-Gapped Monitoring Facility](https://en.wikipedia.org/wiki/Air_gap_(networking)) — an AI inspector reads computation logs, checks them against a public commitment, and emits a single bit: **compliant** or **violation**.

> The bit is the only thing that leaves the AMF. The rationale, flagged records, and confidence are kept inside for adjudication.

**Live demo / write-up:** https://jason.ml/inspector

## What's in here

```
inspector-agents/
├── commitments/                 # YAML treaty specs (what's allowed)
│   └── examples/no_frontier_training.yml
├── logs/                        # Laptop schema + adapters
│   ├── schema.md                # Computation log record format
│   ├── generator.py             # Synthetic scenarios
│   ├── alibaba_adapter.py       # Real Alibaba clusterdata v2023 → our schema
│   ├── real_training.py         # Trains a real GPT, logs to our schema
│   └── scenarios/               # Generated JSONL traces
├── inspector/
│   ├── agent.py                 # inspect(commitment, logs) -> Verdict
│   ├── consistency_checks.py    # Deterministic pre-pass
│   └── prompts.py               # Claude system prompt + user-message builder
├── redteam/                     # Evasion strategies the inspector must catch
├── gpu-runs/                    # 2×H100 SXM testbed
│   ├── README.md                # Run plan, schedule, telemetry channels
│   ├── workload/                # Orchestrator + per-phase entrypoints
│   ├── scripts/                 # Push to HF, smoke tests, dataset packaging
│   └── output/runs/             # Per-host capture artifacts (gitignored)
├── demo.py                      # End-to-end pass/fail matrix (laptop)
├── inspector-blog.html          # Long-form post — source for jason.ml/inspector
├── edit_server.py               # In-place editor for the blog (visit ?edit=1)
└── archive/                     # Old walkthrough drafts + cached demo outputs
```

## Run the laptop demo

```bash
uv sync
uv run python demo.py                   # full 6-case matrix, ~$0.29, ~75s
uv run python demo.py --backend mock    # offline, deterministic, instant
```

The Claude backend uses `claude -p` (Claude Code CLI) — no `ANTHROPIC_API_KEY` needed if you have Claude Code installed.

## Demo result on a fresh checkout

| Case                          | Records | Expected | Got        | Trigger      | Rule                     |
|-------------------------------|--------:|----------|------------|--------------|--------------------------|
| CLEAN / legit_finetune        |     120 | COMPLIANT| **COMPLIANT** | semantic   | —                        |
| EVASION / split_run           |     240 | VIOLATION| **VIOLATION** | semantic   | no-splitting             |
| EVASION / relabel_ops         |     120 | VIOLATION| **VIOLATION** | deterministic | aggregate_flop_cap     |
| EVASION / swap_dataset_hash   |     120 | VIOLATION| **VIOLATION** | semantic   | dataset-provenance       |
| EVASION / pad_batches         |     120 | VIOLATION| **VIOLATION** | deterministic | batch_token_consistency|
| EVASION / drop_context        |     120 | VIOLATION| **VIOLATION** | deterministic | require_upstream_refs  |

**6/6 correct. $0.29 total LLM spend. 75s wall time.**

## H100 testbed (`gpu-runs/`)

Real datacenter hardware: 2× H100 80GB SXM, run on RunPod. Each host runs the same 8-phase schedule (5 honest + 1 idle + 3 adversarial) independently, so the dataset has a cross-host generalization axis (different driver patches, NCCL builds, thermal envelopes, neighbor-tenant interference).

| # | Phase | Duration | Op (logged) | Op (true) | Tests |
|---|-------|----------|-------------|-----------|-------|
| 1 | `honest_pretrain_tiny` | 30 min | training | training | P1·P2·P3 baseline |
| 2 | `honest_pretrain_small` | 30 min | training | training | P1·P2 baseline |
| 3 | `honest_inference_small` | 15 min | inference | inference | P1 baseline |
| 4 | `honest_finetune_tiny` | 20 min | training | training | P3 baseline |
| 5 | `idle` | 5 min | idle | idle | confound check |
| 6 | `adv_train_as_infer` | 15 min | **inference** | training | **P1 attack** |
| 7 | `adv_big_as_small` | 15 min | training | training | **P2 attack** (gpt-small logged as gpt-tiny) |
| 8 | `adv_finetune_as_pretrain` | 15 min | training | training | **P3 attack** |

P1 = training vs inference. P2 = model size ≤ N parameters. P3 = pretrain vs fine-tune. Each property exercises a different telemetry channel, so a single-signal verifier can't catch all three at once.

Per phase, six telemetry streams: `nvsmi.csv` (1 Hz), `dcgm.csv` (10 Hz, where unprivileged containers allow), `nccl_<host>_<pid>.log.gz` (per-rank), `netdev.log.gz`, checkpoints, `provenance.json`. Full design and run notes in [`gpu-runs/README.md`](gpu-runs/README.md); operational notes in [`gpu-runs/CLAUDE.md`](gpu-runs/CLAUDE.md).

Public dataset: [`jasminexli/verifier-challenge-traces`](https://huggingface.co/datasets/jasminexli/verifier-challenge-traces) (CC-BY-4.0). Ground-truth labels in `workload_labels.jsonl` alongside the prover's claimed labels — the whole capture is a label-flip benchmark scoreable by any verifier.

## Real data adapters

- **Alibaba clusterdata v2023** (`logs/alibaba/`) — real production GPU cluster trace, 8,152 pods. Adapter at `logs/alibaba_adapter.py`. Verdict on the laptop verifier: COMPLIANT, $0.10.
- **Self-trained tiny GPT** (`logs/real_training.py`) — 3.2M-param model on Tiny Shakespeare (MPS), every step in schema with full telemetry (forward/backward/optimizer time, per-group grad norms, RSS/CPU/MPS memory, weight norms every 50 steps, top-5 param stats).
- **2×H100 capture** — see above.

Frontier training logs (GPT-4-scale) **do not exist publicly**. The closest is BLOOM tr11-176B at `huggingface.co/bigscience/tr11-176B-logs`. See `logs/scenarios/real_laptop_training.frontier_gap.md` for what's missing vs a frontier trace.

## Why this exists

The verification gap for international AI agreements — written about for years, never implemented. Like some problems before it (self-driving, image recognition, reading sealed Herculaneum scrolls), AI verification is orphaned: too applied for academia, too unprofitable for VCs, too technical for most philanthropy. We're proposing the **Verifier Challenge** — an open competition for the best system for telling, just from a datacenter's logs, what AI is being trained inside. This repo is the smallest working version we could build ourselves before opening the doors.

Live write-up: [jason.ml/inspector](https://jason.ml/inspector).

## License

MIT (code) · CC-BY-4.0 (`verifier-challenge-traces` dataset).
