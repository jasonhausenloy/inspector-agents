"""Render the "pause-and-show" snapshot HTML at training step 200.

Single self-contained file. Inline SVG for charts (no JS frameworks). Walks a
non-technical reviewer through every field in our schema using one real record.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

FIELD_DOCS = {
    "record_id": "Unique ID for this record (monotonic across the run).",
    "prev_record_hash": "SHA-256 of the previous record's canonical JSON. Tampering with any earlier record breaks every later record's hash check.",
    "chip_id": "Identifier for the physical chip producing this record.",
    "job_id": "Logical training/inference job. The verifier groups records by job_id.",
    "operator": "The team or principal running the job.",
    "cluster_region": "Datacenter region (or local-mps for laptop).",
    "window_start": "ISO 8601 UTC timestamp at the start of the window.",
    "window_end": "ISO 8601 UTC timestamp at the end of the window.",
    "op_type": "training | inference | eval | idle. Self-declared; the verifier cross-checks.",
    "flops": "Floating-point operations performed in this window (estimated 6 × params × tokens for training).",
    "tokens_processed": "Tokens processed in this window. For training: batch × sequence.",
    "batch_size": "Number of sequences per training step.",
    "sequence_length": "Tokens per sequence (block_size).",
    "steps_in_window": "Number of training steps aggregated into this record.",
    "model_hash_prefix": "SHA-256 prefix of model parameters. Identifies the model version.",
    "dataset_fingerprint": "SHA-256 prefix of training data. Detects silent dataset substitution.",
    "upstream_refs": "IDs of records this one depends on. Forms a provenance DAG.",
    "data_source_tags": "Labels for the corpus (e.g. 'corpus:tiny-shakespeare').",
    "code_commit": "git short-SHA of the training code at run time.",
    "config_hash": "SHA-256 of the training config (hyperparameters).",
    "_real_training": "Per-step nested block (only on training records).",
}


CSS = """
:root {
  --bg: #f7f4ec;
  --bg-card: #ffffff;
  --ink: #1c1a17;
  --ink-soft: #5e574e;
  --rule: #e6dfd1;
  --good: #1f6b3a;
  --challenge: #b83232;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  --serif: "Iowan Old Style", "Source Serif 4", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 17px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
main {
  max-width: 720px;
  margin: 0 auto;
  padding: 56px 28px;
}
.eyebrow {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--challenge);
  margin-bottom: 8px;
}
h1 {
  font-family: var(--serif);
  font-weight: 500;
  font-size: clamp(30px, 4.5vw, 44px);
  letter-spacing: -0.01em;
  line-height: 1.1;
  margin: 0 0 12px;
}
h2 {
  font-family: var(--serif);
  font-weight: 500;
  font-size: 26px;
  margin: 56px 0 16px;
}
h3 {
  font-family: var(--sans);
  font-weight: 600;
  font-size: 17px;
  margin: 28px 0 8px;
}
p { margin: 0 0 16px; }
hr {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 56px 0;
}
.lead {
  color: var(--ink-soft);
  font-size: 18px;
}
.fingerprint {
  background: var(--bg-card);
  border: 1px solid var(--rule);
  padding: 18px 20px;
  font-family: var(--mono);
  font-size: 13px;
  margin: 24px 0;
}
.fingerprint dt { color: var(--ink-soft); display: inline; }
.fingerprint dd { display: inline; margin: 0 16px 0 6px; }
.fingerprint dd::after { content: ""; display: block; height: 6px; }
table.fields {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
table.fields th, table.fields td {
  border-top: 1px solid var(--rule);
  padding: 10px 0;
  vertical-align: top;
  text-align: left;
}
table.fields th {
  font-family: var(--mono);
  font-weight: 600;
  font-size: 13px;
  width: 200px;
  padding-right: 16px;
  color: var(--ink);
}
table.fields td { color: var(--ink-soft); }
pre.record {
  background: var(--bg-card);
  border: 1px solid var(--rule);
  padding: 18px 20px;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre;
}
svg.chart { display: block; margin: 8px 0 32px; }
.note {
  background: var(--bg-card);
  border-left: 3px solid var(--challenge);
  padding: 14px 18px;
  margin: 24px 0;
  font-size: 15px;
  color: var(--ink-soft);
}
.tag {
  display: inline-block;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  background: var(--bg-card);
  border: 1px solid var(--rule);
  padding: 2px 8px;
  border-radius: 2px;
  color: var(--ink-soft);
  margin-right: 6px;
}
"""


def _line_chart_svg(values: list[float], width: int = 660, height: int = 180,
                    label_x: str = "step", label_y: str = "value",
                    color: str = "#1c1a17", fill: bool = False) -> str:
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(values)
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1
    pts = []
    for i, v in enumerate(values):
        x = pad_l + (i / max(n - 1, 1)) * plot_w
        y = pad_t + (1 - (v - vmin) / (vmax - vmin)) * plot_h
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(pts)
    fill_path = ""
    if fill:
        fill_path = (f'<path d="{path} L {pts[-1].split(",")[0]},{pad_t + plot_h} '
                     f'L {pts[0].split(",")[0]},{pad_t + plot_h} Z" fill="{color}" fill-opacity="0.08"/>')
    # axes
    ax_color = "#cfc7b6"
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" width="100%"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-labelledby="t{abs(hash(label_y)) % 10**8}">
  <title id="t{abs(hash(label_y)) % 10**8}">{label_y} over {label_x}</title>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="{ax_color}"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="{ax_color}"/>
  <text x="{pad_l - 6}" y="{pad_t + 6}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{vmax:.2f}</text>
  <text x="{pad_l - 6}" y="{pad_t + plot_h + 4}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{vmin:.2f}</text>
  <text x="{pad_l + plot_w}" y="{pad_t + plot_h + 18}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{label_x}={n}</text>
  <text x="{pad_l}" y="{pad_t + plot_h + 18}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e">0</text>
  {fill_path}
  <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>
</svg>
"""


def _multiline_chart_svg(series: dict[str, list[float]], width: int = 660, height: int = 200,
                         label_x: str = "step") -> str:
    if not series:
        return ""
    pad_l, pad_r, pad_t, pad_b = 44, 130, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_vals = [v for vs in series.values() for v in vs]
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax == vmin:
        vmax = vmin + 1
    n = max(len(v) for v in series.values())
    colors = ["#1c1a17", "#b83232", "#1f6b3a", "#5e574e"]
    paths = []
    legend = []
    for idx, (k, vs) in enumerate(series.items()):
        col = colors[idx % len(colors)]
        pts = []
        for i, v in enumerate(vs):
            x = pad_l + (i / max(len(vs) - 1, 1)) * plot_w
            y = pad_t + (1 - (v - vmin) / (vmax - vmin)) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            paths.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.5"/>')
        legend.append(
            f'<g transform="translate({pad_l + plot_w + 12},{pad_t + 6 + idx * 18})">'
            f'<line x1="0" y1="0" x2="14" y2="0" stroke="{col}" stroke-width="2"/>'
            f'<text x="20" y="4" font-family="ui-monospace,monospace" font-size="11" fill="#5e574e">{k}</text>'
            f'</g>'
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" width="100%"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="grouped grad norms over {label_x}">
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#cfc7b6"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#cfc7b6"/>
  <text x="{pad_l - 6}" y="{pad_t + 6}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{vmax:.2f}</text>
  <text x="{pad_l - 6}" y="{pad_t + plot_h + 4}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{vmin:.2f}</text>
  <text x="{pad_l + plot_w}" y="{pad_t + plot_h + 18}" font-family="ui-monospace,monospace"
        font-size="10" fill="#5e574e" text-anchor="end">{label_x}={n}</text>
  {"".join(paths)}
  {"".join(legend)}
</svg>
"""


def _record_pretty(rec: dict) -> str:
    return json.dumps(rec, indent=2, default=str)


def render(*, records: list[dict], identity: Any, config: dict, fingerprint: dict, out_path: Path) -> None:
    """Render the snapshot HTML. `records` is the raw training records up to step 200."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample one mid-run record for the field-by-field walkthrough.
    sample_idx = min(100, len(records) - 1)
    sample_record = records[sample_idx]

    # Loss curve.
    losses = [r["_real_training"]["loss"] for r in records]
    grad_norms = [r["_real_training"]["grad_norm"] for r in records]
    rss = [r["_real_training"]["telemetry"]["process_rss_mb"] for r in records]

    # Per-group grad norms (last record).
    last_groups = records[-1]["_real_training"]["grad_norms_per_group"]
    grouped_series: dict[str, list[float]] = {k: [] for k in last_groups.keys()}
    for r in records:
        gnp = r["_real_training"]["grad_norms_per_group"]
        for k in grouped_series.keys():
            grouped_series[k].append(gnp.get(k, 0.0))

    fields_table = "\n".join(
        f'<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>'
        for k, v in FIELD_DOCS.items()
        if k in sample_record or k == "_real_training"
    )

    fp = fingerprint
    cfg = config
    n_records = len(records)
    sample_id = sample_record["record_id"]

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pause and Show — Inspector MVP</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <p class="eyebrow">The Verifier Challenge · Pause and Show</p>
  <h1>We trained for {n_records} steps. Here is every field captured so far.</h1>
  <p class="lead">A real GPT, trained on a real laptop. The same schema a verifier would read at scale, instrumented in full.</p>

  <dl class="fingerprint">
    <dt>host</dt><dd>{html.escape(fp.get('hostname','?'))}</dd>
    <dt>os</dt><dd>{html.escape(fp.get('platform','?'))}</dd>
    <dt>torch</dt><dd>{html.escape(fp.get('torch_version','?'))}</dd>
    <dt>mps</dt><dd>{'available' if fp.get('mps_available') else 'no'}</dd>
    <dt>ram</dt><dd>{fp.get('ram_total_gb','?')} GB</dd>
    <dt>cpu</dt><dd>{fp.get('cpu_count','?')} cores</dd>
    <dt>job</dt><dd>{html.escape(getattr(identity,'job_id',''))}</dd>
    <dt>model_hash</dt><dd>{html.escape(getattr(identity,'model_hash_prefix',''))}</dd>
    <dt>dataset</dt><dd>{html.escape(getattr(identity,'dataset_fingerprint',''))}</dd>
    <dt>config</dt><dd>n_layer={cfg['n_layer']} n_embd={cfg['n_embd']} n_head={cfg['n_head']} block={cfg['block_size']} batch={cfg['batch_size']} lr={cfg['lr']}</dd>
  </dl>

  <h2>Loss is descending</h2>
  <p>Cross-entropy loss after each training step. {len(losses)} points.</p>
  {_line_chart_svg(losses, label_y="loss", fill=True)}

  <h2>What's a record?</h2>
  <p>One record per training step. Below is record <code>{html.escape(sample_id)}</code> (step {sample_record['_real_training']['step']}). Every other record has the same shape — only the values change.</p>
  <pre class="record">{html.escape(_record_pretty(sample_record))}</pre>

  <h2>Field-by-field</h2>
  <table class="fields">
    <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
    <tbody>
{fields_table}
    </tbody>
  </table>

  <h2>Gradient norm by parameter group</h2>
  <p>How much each part of the model is learning. Embeddings, transformer blocks, output head, layer norms — tracked separately.</p>
  {_multiline_chart_svg(grouped_series)}

  <h2>Memory profile</h2>
  <p>Process RSS over the run. Training pre-allocates and stays roughly flat — this profile is one of the strongest training-vs-inference signals.</p>
  {_line_chart_svg(rss, label_y="RSS MB", color="#1f6b3a", fill=True)}

  <h2>The hash chain</h2>
  <p>Each record's <code>prev_record_hash</code> is the SHA-256 of the previous record's canonical JSON. The chain restarts at <code>{'0'*8}…{'0'*8}</code> for each run. Tampering with any record breaks every record after it. This is what lets the verifier audit the trace without re-reading the full history.</p>

  <div class="note">
    <strong>What's not captured here.</strong> Multi-host gradient sync, DCGM tensor-core utilization, per-GPU power draw, hardware attestation. The single-chip laptop run is the proof of concept; closing the gap to frontier verification is what the Verifier Challenge funds. <a href="frontier-gap.html">Read the gap</a>.
  </div>

  <hr>
  <p style="font-family: var(--mono); font-size: 12px; color: var(--ink-soft);">
    <span class="tag">live</span>
    Generated mid-training at step {n_records}. The full run continues to step {cfg.get('steps', '?')} and is then handed to a Claude verifier. <a href="agent.html">See the verifier</a>.
  </p>
</main>
</body>
</html>"""

    out_path.write_text(body, encoding="utf-8")
