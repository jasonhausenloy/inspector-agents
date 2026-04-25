"""Tiny stdlib HTTP server for the live walkthrough.

Run:
    uv run python server.py
    open http://localhost:8765

Endpoints:
    GET /                        → walkthrough.html
    GET /api/audit/<case>        → live JSON verdict from inspect()
    GET /api/results             → cached results fallback

Each /api/audit/* call runs the real Claude-backed inspector and may take
15-40s per case (or near-zero if the case trips a deterministic rule).
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from inspector import inspect, load_commitment, pick_backend
from logs.generator import SCENARIOS
from redteam import EVASIONS

ROOT = Path(__file__).parent
COMMITMENT = load_commitment(ROOT / "commitments/examples/no_frontier_training.yml")
CLEAN = SCENARIOS["legit_finetune"]()
BACKEND = pick_backend("claude")

CASES = {"CLEAN": lambda: list(CLEAN)}
for name, fn in EVASIONS.items():
    CASES[name] = (lambda f=fn: f(list(CLEAN)))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/walkthrough.html"):
            html = (ROOT / "walkthrough.html").read_text()
            self._respond(200, "text/html; charset=utf-8", html.encode())
            return

        if path == "/api/results":
            data = (ROOT / "demo_results.json").read_text()
            self._respond(200, "application/json", data.encode())
            return

        if path.startswith("/api/audit/"):
            case_name = path[len("/api/audit/"):]
            if case_name not in CASES:
                self._respond(404, "application/json",
                              json.dumps({"error": f"unknown case {case_name!r}"}).encode())
                return
            t0 = time.perf_counter()
            try:
                records = CASES[case_name]()
                v = inspect(COMMITMENT, records, backend=BACKEND)
                payload = v.to_dict()
                payload["records"] = len(records)
                payload["wall_seconds"] = round(time.perf_counter() - t0, 2)
                self._respond(200, "application/json", json.dumps(payload).encode())
            except Exception as e:  # surfaced to the UI
                self._respond(500, "application/json",
                              json.dumps({"error": repr(e)}).encode())
            return

        self._respond(404, "text/plain", b"not found")

    def _respond(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # Quiet console — only show our own logs.
        return


def main(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  Inspector Agents — live demo")
    print(f"  Listening on http://localhost:{port}")
    print(f"  {len(CASES)} cases ready: {', '.join(CASES)}")
    print(f"  Each live audit calls Claude (15-40s) and costs ~$0.05-0.15.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye.")


if __name__ == "__main__":
    main()
