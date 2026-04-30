"""Local dev server for inspector-blog.html with in-place editing.

Serves repo files normally and exposes POST /save-blog, which accepts
{"html": "<article inner html>", "prev_hash": "<sha256 of last-known
article inner>"} and writes the new inner HTML back into
inspector-blog.html (between the existing <article> and </article>
tags), but only if prev_hash matches the file's current article hash.
A mismatch returns 409 Conflict, which the in-page editor uses to
detect stale tabs and stop overwriting fresher content.

To make the hash check work, GETs of inspector-blog.html inject the
current article-inner SHA-256 as a `data-prev-hash` attribute on
`<article>`. The browser-side script reads that attribute and sends it
back on every save.

Run:
    python3 edit_server.py [port]

URLs:
    http://localhost:8000/inspector-blog.html         (read-only view)
    http://localhost:8000/inspector-blog.html?edit=1  (editable view)
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import socketserver
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
BLOG_FILE = REPO_ROOT / "inspector-blog.html"
ARTICLE_RE = re.compile(r"(<article>)(.*?)(</article>)", re.DOTALL)
SAVE_LOCK = threading.Lock()


def _hash_inner(inner: str) -> str:
    return hashlib.sha256(inner.encode("utf-8")).hexdigest()


def _read_article_inner() -> tuple[str, str]:
    """Return (full file text, article inner content). Raises if not found."""
    text = BLOG_FILE.read_text(encoding="utf-8")
    match = ARTICLE_RE.search(text)
    if match is None:
        raise RuntimeError(f"no <article> tag in {BLOG_FILE}")
    return text, match.group(2)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/inspector-blog.html":
            self._serve_blog_with_hash()
            return
        super().do_GET()

    def _serve_blog_with_hash(self) -> None:
        text, inner = _read_article_inner()
        digest = _hash_inner(inner)
        injected = text.replace(
            "<article>",
            f'<article data-prev-hash="{digest}">',
            1,
        )
        body = injected.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save-blog":
            self.send_error(404, "unknown endpoint")
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self.send_error(400, "missing body")
            return

        raw = self.rfile.read(length)
        data = json.loads(raw)
        new_inner = data["html"]
        if not isinstance(new_inner, str):
            self.send_error(400, "'html' must be a string")
            return
        prev_hash = data.get("prev_hash")
        if not isinstance(prev_hash, str) or not prev_hash:
            self.send_error(400, "'prev_hash' must be a non-empty string")
            return

        with SAVE_LOCK:
            current_text, current_inner = _read_article_inner()
            current_hash = _hash_inner(current_inner)
            if current_hash != prev_hash:
                self._reply_json(
                    409,
                    {
                        "ok": False,
                        "error": "stale: file modified since this tab loaded",
                        "current_hash": current_hash,
                    },
                )
                return

            updated = ARTICLE_RE.sub(
                lambda m: m.group(1) + new_inner + m.group(3),
                current_text,
                count=1,
            )
            BLOG_FILE.write_text(updated, encoding="utf-8")
            new_hash = _hash_inner(new_inner)
            byte_count = len(updated.encode("utf-8"))

        self._reply_json(
            200,
            {"ok": True, "bytes": byte_count, "new_hash": new_hash},
        )

    def _reply_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with ReusableServer(("", port), Handler) as httpd:
        print(f"editing server on http://localhost:{port}")
        print(f"  view:  http://localhost:{port}/inspector-blog.html")
        print(f"  edit:  http://localhost:{port}/inspector-blog.html?edit=1")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
