"""Local dev server for inspector-blog.html with in-place editing.

Serves repo files normally and exposes POST /save-blog, which accepts
{"html": "<article inner html>"} and writes it back into
inspector-blog.html (between the existing <article> and </article> tags).

Run:
    python3 edit_server.py [port]

URLs:
    http://localhost:8000/inspector-blog.html         (read-only view)
    http://localhost:8000/inspector-blog.html?edit=1  (editable view)
"""

from __future__ import annotations

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


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        # Disable caching so edits are picked up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

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

        with SAVE_LOCK:
            current = BLOG_FILE.read_text(encoding="utf-8")
            updated, count = ARTICLE_RE.subn(
                lambda m: m.group(1) + new_inner + m.group(3),
                current,
                count=1,
            )
            if count != 1:
                self.send_error(500, "could not locate <article> in source")
                return
            BLOG_FILE.write_text(updated, encoding="utf-8")
            byte_count = len(updated.encode("utf-8"))

        body = json.dumps({"ok": True, "bytes": byte_count}).encode("utf-8")
        self.send_response(200)
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
