"""Render verdicts/*.json into a single static HTML page (web/agent.html).

Each verdict is presented as an expandable card: case name + verdict + a
collapsed details panel containing the rationale, primary signals,
adversarial concerns considered, and usage / cost. Designed so a non-
technical reviewer can click through every Claude-decided verdict and read
exactly what reasoning produced it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERDICTS = ROOT / "verdicts"
OUT = ROOT / "web" / "agent.html"


CASE_ORDER = [
    # filename suffix → (label, expected, description)
    ("p1_training", ("Clean training trace", "training", "The real 2,000-step training run on M3.")),
    ("p1_inference", ("Clean inference trace", "inference", "The real 320-token generation run.")),
    ("p1_adversarial_adv1_eval_with_grad", ("Adv-1 · Eval with grad enabled", "inference", "Inference workload with loss-shaped fields injected. No weight drift, no optimizer state.")),
    ("p1_adversarial_adv2_microbatch_train", ("Adv-2 · Microbatch training", "training", "Training with B=1, T=1. Looks inference-y per record.")),
    ("p1_adversarial_adv3_relabeled", ("Adv-3 · Op-type relabel", "training", "Training records with op_type set to 'inference'.")),
    ("p1_adversarial_adv4_underreported", ("Adv-4 · Under-reported FLOPs", "training", "Declared FLOPs halved, batch_size + seq_len kept honest.")),
    ("p1_adversarial_adv5_split_run", ("Adv-5 · Run-splitting", "training", "One logical run fragmented into two job_ids.")),
    ("p2_training_t5e+13", ("P2 · Training trace, T = 5×10¹³", "above_threshold", "Real run total is 1.58×10¹⁴ — above 5×10¹³.")),
    ("p2_training_t5e+14", ("P2 · Training trace, T = 5×10¹⁴", "below_threshold", "Real run total is 1.58×10¹⁴ — below 5×10¹⁴.")),
    ("p2_adversarial_adv4_underreported_t5e+13", ("P2 · Adv-4 (under-reported), T = 5×10¹³", "above_threshold", "Declared total is below T, but Kaplan cross-check exposes the under-reporting.")),
    ("p2_adversarial_adv5_split_run_t5e+13", ("P2 · Adv-5 (run-splitting), T = 5×10¹³", "above_threshold", "Aggregating across job_ids exceeds T.")),
]


CSS = """
:root {
  --bg: #f7f4ec;
  --bg-card: #ffffff;
  --ink: #1c1a17;
  --ink-soft: #5e574e;
  --rule: #e6dfd1;
  --good: #1f6b3a;
  --good-bg: #e6f1ea;
  --bad: #8a1d1d;
  --bad-bg: #f4e3e3;
  --challenge: #b83232;
  --pending: #94887a;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  --serif: "Iowan Old Style", "Source Serif 4", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 17px; line-height: 1.65; }
main { max-width: 720px; margin: 0 auto; padding: 56px 28px 96px; }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--challenge); margin-bottom: 12px; }
h1 { font-family: var(--serif); font-weight: 500; font-size: clamp(34px, 5vw, 48px); letter-spacing: -0.01em; line-height: 1.05; margin: 0 0 12px; }
h2 { font-family: var(--serif); font-weight: 500; font-size: 26px; margin: 56px 0 16px; }
p { margin: 0 0 16px; }
.lead { color: var(--ink-soft); font-size: 18px; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 56px 0; }
a { color: var(--challenge); text-decoration: none; border-bottom: 1px solid currentColor; }
a:hover { color: var(--ink); }
.back { font-family: var(--mono); font-size: 13px; margin-bottom: 32px; display: inline-block; }
details.case {
  border: 1px solid var(--rule);
  background: var(--bg-card);
  border-radius: 4px;
  margin: 12px 0;
  overflow: hidden;
}
details.case summary {
  padding: 16px 20px;
  cursor: pointer;
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 14px;
  user-select: none;
}
details.case summary::-webkit-details-marker { display: none; }
details.case summary::before {
  content: "▸";
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink-soft);
  display: inline-block;
  width: 12px;
  transition: transform 0.2s ease;
}
details.case[open] summary::before { transform: rotate(90deg); }
details.case .label { flex: 1 1 auto; font-weight: 600; color: var(--ink); }
details.case .label small { display: block; font-weight: 400; font-size: 13px; color: var(--ink-soft); margin-top: 2px; }
.verdict-tag {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 2px;
}
.verdict-tag.training, .verdict-tag.above_threshold { background: var(--bad-bg); color: var(--bad); }
.verdict-tag.inference, .verdict-tag.below_threshold { background: var(--good-bg); color: var(--good); }
.verdict-tag.miss { background: #fff4d6; color: #7a5300; }
.match-mark {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
}
.match-mark.ok { color: var(--good); }
.match-mark.bad { color: var(--bad); }
.case-body {
  padding: 0 20px 20px;
  border-top: 1px solid var(--rule);
}
.case-body h3 {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin: 18px 0 8px;
  font-weight: 600;
}
.case-body p.rationale {
  font-size: 16px;
  line-height: 1.6;
  color: var(--ink);
}
.case-body ul {
  padding-left: 18px;
  margin: 0 0 8px;
}
.case-body ul li {
  font-size: 14px;
  line-height: 1.5;
  color: var(--ink-soft);
  margin: 6px 0;
}
.case-body ul li code,
.case-body ul li b {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--ink);
  background: transparent;
}
.usage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px 24px;
  margin: 12px 0;
  font-family: var(--mono);
  font-size: 13px;
  color: var(--ink-soft);
}
.usage-grid .label { color: var(--pending); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
.usage-grid .v { color: var(--ink); display: block; margin-top: 2px; font-size: 13px; }
.summary-block {
  background: var(--bg);
  border: 1px solid var(--rule);
  padding: 14px 18px;
  margin: 32px 0;
  font-size: 15px;
}
.summary-block strong { font-weight: 700; color: var(--ink); }
.note {
  background: var(--bg-card);
  border-left: 3px solid var(--challenge);
  padding: 14px 18px;
  margin: 24px 0;
  font-size: 15px;
  color: var(--ink-soft);
}
"""


def render_case(slug: str, label: str, expected: str, description: str, verdict: dict) -> str:
    v = verdict.get("verdict", "?")
    conf = verdict.get("confidence", 0)
    rationale = verdict.get("rationale", "")
    signals = verdict.get("primary_signals", [])
    adv = verdict.get("adversarial_concerns", [])
    flagged = verdict.get("flagged_record_ids", [])
    est = verdict.get("estimated_total_flops")
    meta = verdict.get("_meta", {})
    usage = meta.get("usage", {})

    matched = (v == expected)
    mark = '<span class="match-mark ok">✓ matches</span>' if matched else '<span class="match-mark bad">✗ disagrees</span>'

    signal_lis = "\n".join(f"<li>{html.escape(s)}</li>" for s in signals)
    adv_lis = "\n".join(f"<li>{html.escape(a)}</li>" for a in adv) if adv else "<li><em>None ruled in.</em></li>"
    flagged_str = ", ".join(flagged) if flagged else "—"

    est_str = f"{est:.3e} FLOPs" if isinstance(est, (int, float)) and est else "—"

    return f"""
<details class="case">
  <summary>
    <span class="label">{html.escape(label)}<small>{html.escape(description)}</small></span>
    <span class="verdict-tag {html.escape(v)}">{html.escape(v)}</span>
    {mark}
  </summary>
  <div class="case-body">
    <h3>Rationale</h3>
    <p class="rationale">{html.escape(rationale)}</p>
    <h3>Signals used</h3>
    <ul>{signal_lis}</ul>
    <h3>Adversarial concerns considered</h3>
    <ul>{adv_lis}</ul>
    <h3>Audit fields</h3>
    <div class="usage-grid">
      <div><span class="label">Confidence</span><span class="v">{conf}</span></div>
      <div><span class="label">Estimated total FLOPs</span><span class="v">{est_str}</span></div>
      <div><span class="label">Flagged record IDs</span><span class="v">{html.escape(flagged_str)}</span></div>
      <div><span class="label">Model</span><span class="v">{html.escape(usage.get('model','?'))}</span></div>
      <div><span class="label">Wall</span><span class="v">{meta.get('wall_seconds','?')}s</span></div>
      <div><span class="label">Cost</span><span class="v">${meta.get('estimated_cost_usd','?')}</span></div>
      <div><span class="label">Cache writes</span><span class="v">{usage.get('cache_creation_input_tokens',0):,} tok</span></div>
      <div><span class="label">Cache reads</span><span class="v">{usage.get('cache_read_input_tokens',0):,} tok</span></div>
      <div><span class="label">Output</span><span class="v">{usage.get('output_tokens',0):,} tok</span></div>
    </div>
  </div>
</details>
"""


def main() -> None:
    cases_html = []
    correct = 0
    total = 0
    p1_correct = 0
    p1_total = 0
    p2_correct = 0
    p2_total = 0
    total_cost = 0.0
    for slug, (label, expected, desc) in CASE_ORDER:
        path = VERDICTS / f"{slug}.json"
        if not path.exists():
            continue
        verdict = json.loads(path.read_text())
        cases_html.append(render_case(slug, label, expected, desc, verdict))
        total += 1
        is_p2 = slug.startswith("p2_")
        if is_p2:
            p2_total += 1
        else:
            p1_total += 1
        if verdict.get("verdict") == expected:
            correct += 1
            if is_p2:
                p2_correct += 1
            else:
                p1_correct += 1
        meta = verdict.get("_meta", {})
        total_cost += float(meta.get("estimated_cost_usd", 0.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude reads. Claude decides. — Inspector MVP</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <p><a href="/inspector-new" class="back">← inspector overview</a></p>
  <p class="eyebrow">The Verifier Challenge · Agent</p>
  <h1>Claude reads. Claude decides.</h1>
  <p class="lead">Two properties. One bit out per verdict. Click any case to see the rationale, the signals Claude actually used, and the adversarial patterns it ruled in or out.</p>

  <div class="summary-block">
    <strong>Property 1 (training vs inference): {p1_correct}/{p1_total} correct.</strong>
    Property 2 (FLOP threshold): {p2_correct}/{p2_total} correct.
    Total cost across all {total} verdicts: <strong>${total_cost:.2f}</strong>.
    Default model: <code>claude-haiku-4-5</code>. Single-shot per verdict, prompt caching enabled.
  </div>

  <h2>Property 1 — Training vs Inference</h2>
  <p>Given a trace, decide whether it was produced by a training run (gradients, loss, optimizer state, large per-step token counts) or an inference workload (per-token records, attention summaries, KV-cache, no weight drift). The verifier never reads the <code>op_type</code> label — it has to derive the answer from the data.</p>

  {''.join(c for c in cases_html if 'P2' not in c.split('label">')[1].split('<')[0])}

  <h2>Property 2 — FLOP threshold compliance</h2>
  <p>Same agent, different question: was this model trained above a stated FLOP cap? We use scaled-down thresholds (5×10¹³ and 5×10¹⁴ FLOPs) to match laptop-scale runs; the same logic applies at the EU AI Act's 1×10²⁵ bar. The verifier sums declared FLOPs and cross-checks against the Kaplan estimate (6×params×tokens) to catch under-reporting.</p>

  {''.join(c for c in cases_html if 'P2' in c.split('label">')[1].split('<')[0])}

  <div class="note">
    <strong>Honest wrinkle.</strong> Haiku 4.5 occasionally inverts the verdict label on borderline below-threshold cases — its rationale is mathematically correct, but it writes "above_threshold" where the math says below. Sonnet 4.6 is the planned escalation. Both verdicts above are shown verbatim from the model output, including the inversion.
  </div>

  <hr>
  <p style="font-family: var(--mono); font-size: 12px; color: var(--ink-soft);">
    Each verdict is a single API call to Anthropic with the system prompt cached for 1 hour and the log payload cached for 5 minutes. Total LLM spend across this page: <strong>${total_cost:.2f}</strong>.
  </p>
</main>
</body>
</html>"""

    OUT.write_text(body, encoding="utf-8")
    print(f"  → wrote {OUT}  ({total} cases, {correct}/{total} correct, ${total_cost:.2f})")


if __name__ == "__main__":
    main()
