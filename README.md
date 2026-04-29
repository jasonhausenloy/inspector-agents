# Inspector MVP — Claude as AI Verifier

A working proof-of-concept for the **Verifier Challenge** — an open competition for AI verification proposed by Jason Hausenloy and Jasmine Li. A scaffolded Claude agent reads raw logs from a real GPT we trained on a MacBook M3 and infers two properties:

1. **Training vs Inference** — given a log, is this a training run or inference serving?
2. **FLOP threshold compliance** — was the model trained above a stated FLOP cap? (Scaled-down EU AI Act analog.)

**Live demo:** https://jason.ml/inspector

---

## Quick start

```bash
uv sync
bash scripts/run_full_demo.sh    # end-to-end: train → infer → adversarial → verify → render
```

Individual steps:

```bash
uv run python -m inspector_v2.train --steps 2000              # ~5–7 min on M3
uv run python -m inspector_v2.infer                           # ~30 s
uv run python -m inspector_v2.redteam                         # synth 5 adversarial traces
uv run python -m inspector_v2.verifier --property p1          # training-vs-inference
uv run python -m inspector_v2.verifier --property p2 --threshold 5e10
```

## Repo layout

```
inspector_v2/        # All code
  model.py           # TinyGPT (~3.2M params, 4 layers, 256 d_model)
  train.py           # Training loop + 2 JSONL writers + step-200 snapshot
  infer.py           # Generation with full per-token tracing
  log_writer.py      # Schema-compliant log builders + hash chain
  snapshot.py        # The "pause-and-show" HTML at step 200
  verifier.py        # Anthropic SDK + prompt caching + Pydantic Verdict
  prompts.py         # Frozen system prompt for both properties
  redteam.py         # 5 adversarial trace synthesizers
  render_results.py  # Verdict JSONs → HTML pages

traces/              # Generated artifacts (gitignored except summary.jsonl)
verdicts/            # Verifier output (committed; small JSON)
web/                 # Static HTML pages, mirrored to jason.ml/inspector-new
scripts/             # Run + deploy scripts
```

## Hardware

Built for **MacBook Pro M3 with MPS**. The training script uses `torch.device("mps")` and Apple Silicon-specific telemetry. CUDA also works but the MPS-specific telemetry fields will be `null`.

## What this proves

The Verifier Challenge thesis: given a datacenter log, infer what's running inside. This MVP shows it works end-to-end on a real training trace, with Claude as the verifier and no rule-engine fallback.

## What this doesn't claim

This is a single-chip laptop run. Frontier verification needs multi-host gradient-sync telemetry, DCGM tensor-core utilization, hardware attestation, and adversarial robustness across hundreds of evasion strategies. The Challenge funds that work. See [frontier-gap](web/frontier-gap.html) for honest limits.

## License

MIT.
