# Inspector Agents

A working tabletop implementation of the [Air-Gapped Monitoring Facility](https://en.wikipedia.org/wiki/Air_gap_(networking)) described in *Inspector Agents: Privacy Preserving Monitoring for Compliance with AI Agreements*. An AI inspector reads computation logs, checks them against a public commitment, and emits a single bit: **compliant** or **violation**.

> The bit is the only thing that leaves the AMF. The rationale, flagged records, and confidence are kept inside for adjudication.

**Live interactive demo:** https://jason.ml/inspector

## What's in here

```
inspector-agents/
├── commitments/                 # YAML treaty specs (what's allowed)
│   └── examples/no_frontier_training.yml
├── logs/
│   ├── schema.md                # Computation log record format
│   ├── generator.py             # Synthetic scenarios
│   ├── alibaba_adapter.py       # Real Alibaba clusterdata v2023 → our schema
│   ├── real_training.py         # Trains a real GPT, logs to our schema
│   └── scenarios/               # Generated JSONL traces
├── inspector/
│   ├── agent.py                 # inspect(commitment, logs) -> Verdict
│   ├── consistency_checks.py    # Deterministic pre-pass
│   └── prompts.py               # Claude system prompt + user-message builder
├── redteam/                     # Five evasion strategies the inspector must catch
├── demo.py                      # End-to-end pass/fail matrix
├── server.py                    # Live HTTP server for the walkthrough
└── walkthrough.html             # Single-file interactive demo
```

## Run the demo

```bash
uv sync
uv run python demo.py                   # full 6-case matrix, ~$0.29, ~75s
uv run python demo.py --backend mock    # offline, deterministic, instant
uv run python server.py                 # live walkthrough at http://localhost:8765
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

## Real data

Tested against:
- **Alibaba clusterdata v2023** (`logs/alibaba/`) — real production GPU cluster trace, 8,152 pods. Adapter at `logs/alibaba_adapter.py`. Verdict: COMPLIANT, $0.10.
- **Self-trained tiny GPT** (`logs/real_training.py`) — trains a 3.2M-param model on Tiny Shakespeare on Apple Silicon MPS, logs every step in our schema with full telemetry (forward/backward/optimizer time, per-group grad norms, RSS/CPU/MPS memory, weight norms every 50 steps, top-5 param stats).

Frontier training logs (GPT-4-scale) **do not exist publicly**. The closest is BLOOM tr11-176B at huggingface.co/bigscience/tr11-176B-logs. See `logs/scenarios/real_laptop_training.frontier_gap.md` for what's missing vs a frontier trace.

## Why this exists

The verification gap for international AI treaties — written about for years, never implemented. The hard part wasn't ever the technology; it was nobody building it. This is the working prototype.

Built in a day from the [Inspector Agents](https://example.invalid/inspector-agents-paper) and [Auto-Adjudication](https://example.invalid/auto-adjudication-paper) drafts.

## License

MIT.
