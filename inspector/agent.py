"""The Inspector Agent.

One entry point: `inspect(commitment, records) -> Verdict`.

Implements the Air-Gapped Monitoring Facility contract:
  - deterministic pre-pass (fast, no LLM) → if any rule trips, return 0
  - else build a context-rich prompt and call the LLM backend
  - LLM emits a JSON verdict; we keep confidence/flagged records locally and
    expose only the single `verdict` bit to the outside world.

Backends: `ClaudeCLIBackend` (default, uses `claude -p`; no API key required,
uses the user's existing Claude Code auth) and `MockBackend` (deterministic,
for CI and offline testing).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from inspector import consistency_checks as cc
from inspector.prompts import SYSTEM_PROMPT, build_user_message


@dataclass
class Verdict:
    commitment_id: str
    verdict: int  # 1 = compliant, 0 = violation; THIS IS THE ONLY BIT THAT LEAVES THE AMF
    confidence: float
    triggered_rule_id: str
    flagged_record_ids: list[str]
    rationale: str
    trigger: str  # "deterministic" | "semantic" | "mock"
    llm_cost_usd: float = 0.0

    # Withheld fields — kept inside the AMF, never exfiltrated
    _internal: dict = field(default_factory=dict)

    def public_bit(self) -> int:
        """The only datum the AMF emits over its 1-bit/hour channel."""
        return self.verdict

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "triggered_rule_id": self.triggered_rule_id,
            "flagged_record_ids": self.flagged_record_ids,
            "rationale": self.rationale,
            "trigger": self.trigger,
            "llm_cost_usd": self.llm_cost_usd,
        }


class LLMBackend(Protocol):
    def classify(self, system_prompt: str, user_message: str) -> tuple[dict, float]: ...


# --- Backend: Claude Code CLI --------------------------------------------

class ClaudeCLIBackend:
    """Shell out to `claude -p`. Uses Jason's existing Claude Code auth; no
    ANTHROPIC_API_KEY required. Cost ~2¢/call with prompt caching on the
    system prompt.
    """

    def __init__(self, model: str = "haiku", cwd: str = "/tmp"):
        if not shutil.which("claude"):
            raise RuntimeError("`claude` CLI not found on PATH")
        self.model = model
        self.cwd = cwd  # run from /tmp so no CLAUDE.md auto-loads

    def classify(self, system_prompt: str, user_message: str) -> tuple[dict, float]:
        proc = subprocess.run(
            [
                "claude", "-p",
                "--output-format", "json",
                "--model", self.model,
                "--system-prompt", system_prompt,
            ],
            input=user_message,
            capture_output=True,
            text=True,
            cwd=self.cwd,
            timeout=180,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or "")[:300]
            if proc.stdout:
                detail += " | stdout: " + proc.stdout[:300]
            raise RuntimeError(f"claude CLI rc={proc.returncode}: {detail}")
        envelope = json.loads(proc.stdout)
        cost = float(envelope.get("total_cost_usd", 0.0))
        result_text: str = envelope["result"]
        # Claude sometimes wraps the JSON in prose or fences despite instructions; extract.
        verdict = _extract_json(result_text)
        return verdict, cost


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    # Strip ``` fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise RuntimeError(f"no JSON object in LLM output: {text[:500]}")
        return json.loads(m.group(0))


# --- Backend: deterministic mock -----------------------------------------

class MockBackend:
    """Heuristic-only backend. Flags the same patterns the LLM is asked to
    catch, but via hand-coded rules. Used for CI and for env without LLM access.
    """

    def classify(self, system_prompt: str, user_message: str) -> tuple[dict, float]:
        # Parse the per-job summary, suspicious records, splitting pairs out of the message.
        summary = _parse_section(user_message, "PER-JOB SUMMARY")
        splits = _parse_section(user_message, "POTENTIAL RUN-SPLITTING PAIRS")
        swaps = _parse_section(user_message, "DATASET-FINGERPRINT SWAPS WITHIN A JOB")
        commitment = _parse_section(user_message, "COMMITMENT")
        if commitment is None:
            return ({
                "commitment_id": "unknown",
                "verdict": 1,
                "confidence": 0.0,
                "triggered_rule_id": "none",
                "flagged_record_ids": [],
                "rationale": "mock: could not parse commitment",
            }, 0.0)

        flagged: list[str] = []
        trigger = "none"

        # op-type-honesty
        if summary:
            for job in summary.values():
                for sus in job.get("suspicious_inference_records", []):
                    flagged.append(sus["record_id"])
                    trigger = "op-type-honesty"
                for sus in job.get("suspicious_batch_records", []):
                    flagged.append(sus["record_id"])
                    trigger = "batch-token-sanity" if trigger == "none" else trigger

        # no-splitting: check if any (shared_model, shared_commit) pair's combined
        # training FLOPs exceed the training cap from deterministic_rules.
        train_cap = None
        for rule in commitment.get("deterministic_rules", []):
            if rule["kind"] == "aggregate_flop_cap" and rule.get("op_type") == "training":
                train_cap = float(rule["threshold"])
        if splits and train_cap and summary:
            for a_id, b_id in splits:
                ja, jb = summary.get(a_id, {}), summary.get(b_id, {})
                total = ja.get("total_training_flops", 0) + jb.get("total_training_flops", 0)
                if total > train_cap:
                    # Flag the first record of the second job (earliest that pushed over)
                    trigger = "no-splitting"
                    # We don't have raw records here; the agent's summary doesn't carry
                    # first-record id. Just signal via rationale.
                    flagged.append(f"<splitting:{a_id}+{b_id}>")

        # dataset-provenance
        if swaps:
            trigger = "dataset-provenance" if trigger == "none" else trigger
            for s in swaps:
                flagged.append(s["record_id"])

        verdict = 0 if flagged else 1
        return ({
            "commitment_id": commitment.get("id", "unknown"),
            "verdict": verdict,
            "confidence": 0.9 if verdict == 0 else 0.8,
            "triggered_rule_id": trigger if verdict == 0 else "none",
            "flagged_record_ids": flagged,
            "rationale": f"mock backend: {trigger}" if verdict == 0 else "mock: no heuristic flags",
        }, 0.0)


def _parse_section(message: str, header: str):
    """Pull the JSON (or 'None detected.' sentinel) body of a '# HEADER' section."""
    marker = f"# {header}"
    start = message.find(marker)
    if start == -1:
        return None
    body_start = message.find("\n", start) + 1
    next_header = message.find("\n# ", body_start)
    body = message[body_start:next_header if next_header != -1 else len(message)].strip()
    if body.startswith("None detected") or body.startswith("All deterministic_rules passed"):
        return None
    try:
        return json.loads(body.split("\n\n", 1)[0]) if body.startswith(("{", "[")) else json.loads(body)
    except json.JSONDecodeError:
        # Section might be raw JSONL (the log itself). Return None to skip.
        return None


# --- Public entry point --------------------------------------------------

def inspect(
    commitment: dict,
    records: list[dict],
    backend: LLMBackend | None = None,
) -> Verdict:
    """Run the full AMF pipeline: deterministic pre-pass + (if clean) LLM.

    Returns a Verdict. Only `verdict.public_bit()` would be exfiltrated in
    production; the rest is held for adjudication.
    """
    # Stage 1: deterministic checks
    det_viols = cc.run_deterministic(commitment, records)
    if det_viols:
        first = det_viols[0]
        return Verdict(
            commitment_id=commitment["id"],
            verdict=0,
            confidence=1.0,
            triggered_rule_id=first.rule_id,
            flagged_record_ids=[v.record_id for v in det_viols if v.record_id],
            rationale=first.message
            + (f" (+{len(det_viols)-1} more)" if len(det_viols) > 1 else ""),
            trigger="deterministic",
            _internal={"violations": [v.to_dict() for v in det_viols]},
        )

    # Stage 2: signal extraction for the LLM
    summary = cc.summarize_for_llm(records)
    splitting_pairs = cc.detect_splitting_candidates(summary)
    dataset_swaps = cc.detect_dataset_swaps(records)
    user_msg = build_user_message(
        commitment=commitment,
        records=records,
        summary=summary,
        splitting_pairs=splitting_pairs,
        dataset_swaps=dataset_swaps,
        deterministic_violations=[],
    )

    # Stage 3: LLM classification
    if backend is None:
        backend = ClaudeCLIBackend()
    raw, cost = backend.classify(SYSTEM_PROMPT, user_msg)

    return Verdict(
        commitment_id=raw.get("commitment_id", commitment["id"]),
        verdict=int(raw["verdict"]),
        confidence=float(raw.get("confidence", 0.5)),
        triggered_rule_id=raw.get("triggered_rule_id", "none"),
        flagged_record_ids=list(raw.get("flagged_record_ids", [])),
        rationale=raw.get("rationale", ""),
        trigger="semantic",
        llm_cost_usd=cost,
        _internal={"raw": raw, "splitting_pairs": splitting_pairs, "dataset_swaps": dataset_swaps},
    )


def load_commitment(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_backend(name: str) -> LLMBackend:
    if name == "mock":
        return MockBackend()
    if name == "claude":
        return ClaudeCLIBackend(
            model=os.environ.get("INSPECTOR_MODEL", "haiku"),
        )
    raise ValueError(f"unknown backend: {name!r}")
