"""End-to-end demo: Inspector Agent catching red-team evasions.

Runs:
  1. A clean trace through the inspector → expect VERDICT=1 (compliant).
  2. Each red-team evasion applied to the clean trace → expect VERDICT=0,
     with an appropriate triggered rule and flagged records.
  3. Prints a pass/fail matrix with cost and timing.

Usage:
  uv run python demo.py                      # real Claude backend, full matrix
  uv run python demo.py --backend mock       # offline heuristic backend
  uv run python demo.py --only split_run     # single evasion
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inspector import Verdict, inspect, load_commitment, pick_backend
from logs.generator import SCENARIOS
from redteam import EVASIONS

console = Console()


def _verdict_cell(v: Verdict, expected: int) -> Text:
    ok = v.verdict == expected
    symbol = "✓" if ok else "✗"
    label = "COMPLIANT" if v.verdict == 1 else "VIOLATION"
    style = "bold green" if ok else "bold red"
    return Text(f"{symbol} {label}", style=style)


def run_once(name: str, commitment: dict, records: list[dict], backend, expected: int):
    t0 = time.perf_counter()
    v = inspect(commitment, records, backend=backend)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "records": len(records),
        "verdict": v,
        "expected": expected,
        "elapsed": elapsed,
    }


def run_demo(backend_name: str, only: str | None, scenario_name: str):
    backend = pick_backend(backend_name)
    commitment = load_commitment("commitments/examples/no_frontier_training.yml")

    console.rule("[bold]Inspector Agent — Demo")
    console.print(Panel(
        f"[bold]Commitment:[/] {commitment['title']}\n"
        f"[bold]Scenario:[/]   {scenario_name} (clean baseline)\n"
        f"[bold]Backend:[/]    {backend_name}\n"
        f"[bold]Evasions:[/]   {len(EVASIONS)} red-team attacks",
        title="Setup",
    ))

    clean_records = SCENARIOS[scenario_name]()
    rows: list[dict] = []

    # 1. Clean trace
    console.print(f"\n[cyan]→[/] Running clean trace ({len(clean_records)} records)...")
    rows.append(run_once(f"CLEAN / {scenario_name}", commitment, clean_records, backend, expected=1))
    _print_row(rows[-1])

    # 2. Each evasion
    for ev_name, mutate in EVASIONS.items():
        if only and ev_name != only:
            continue
        evaded = mutate(clean_records)
        console.print(f"\n[cyan]→[/] Running evasion [bold yellow]{ev_name}[/] ({len(evaded)} records)...")
        rows.append(run_once(f"EVASION / {ev_name}", commitment, evaded, backend, expected=0))
        _print_row(rows[-1])

    # 3. Summary matrix
    console.rule("[bold]Verdict Matrix")
    t = Table(show_lines=True)
    t.add_column("Case")
    t.add_column("Records", justify="right")
    t.add_column("Expected", justify="center")
    t.add_column("Got", justify="center")
    t.add_column("Trigger")
    t.add_column("Rule")
    t.add_column("Flagged", justify="right")
    t.add_column("Time", justify="right")
    t.add_column("Cost", justify="right")

    total_cost = 0.0
    all_correct = True
    for row in rows:
        v = row["verdict"]
        correct = v.verdict == row["expected"]
        all_correct = all_correct and correct
        total_cost += v.llm_cost_usd
        t.add_row(
            row["name"],
            str(row["records"]),
            "COMPLIANT" if row["expected"] == 1 else "VIOLATION",
            _verdict_cell(v, row["expected"]),
            v.trigger,
            v.triggered_rule_id,
            str(len(v.flagged_record_ids)),
            f"{row['elapsed']:.1f}s",
            f"${v.llm_cost_usd:.4f}",
        )
    console.print(t)

    console.print(
        f"\n[bold]Total LLM cost:[/] ${total_cost:.4f}   "
        f"[bold]Cases correct:[/] {sum(1 for r in rows if r['verdict'].verdict == r['expected'])}/{len(rows)}"
    )
    console.print(
        "\n[dim]Reminder: in production, only the single-bit [bold]verdict[/] column would "
        "leave the Air-Gapped Monitoring Facility.[/]"
    )
    return all_correct


def _print_row(row):
    v = row["verdict"]
    correct = v.verdict == row["expected"]
    style = "green" if correct else "red"
    console.print(
        f"   [bold {style}]verdict={v.verdict}[/] "
        f"trigger={v.trigger} rule={v.triggered_rule_id} "
        f"flagged={len(v.flagged_record_ids)} "
        f"elapsed={row['elapsed']:.1f}s cost=${v.llm_cost_usd:.4f}"
    )
    if v.rationale:
        console.print(f"   [dim]rationale:[/] {v.rationale[:200]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude", "mock"], default="claude")
    ap.add_argument("--only", default=None, help="run only one evasion by name")
    ap.add_argument("--scenario", default="legit_finetune", choices=list(SCENARIOS.keys()))
    args = ap.parse_args()

    ok = run_demo(args.backend, args.only, args.scenario)
    raise SystemExit(0 if ok else 1)
