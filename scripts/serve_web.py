"""Preview the static dashboard locally with Netlify-like routing.

    python -m scripts.serve_web            # http://127.0.0.1:8080

Rules (same as netlify.toml): existing files are served as-is, missing files
under /data/ return 404, every other path returns index.html (client-side routing).
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path != "/" and not (WEB / path.lstrip("/")).exists():
            if path.startswith("/data/"):
                self.send_error(404)
                return None
            self.path = "/index.html"
        return super().send_head()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = functools.partial(SPAHandler, directory=str(WEB))
    print(f"Serving {WEB} at http://127.0.0.1:{args.port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
