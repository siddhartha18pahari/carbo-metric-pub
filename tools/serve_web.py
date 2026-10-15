"""Serve web/ the way Vercel does: static files only, /static/* rewritten to /*,
no Python engine behind it, so the page runs the browser engine.

    python tools/serve_web.py [port]      # default 8766
"""

import functools
import http.server
import os
import sys

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    # module scripts need a JavaScript MIME type whatever the OS registry says
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css"}

    def translate_path(self, path):
        if path.startswith("/static/"):
            path = path[len("/static"):]
        return super().translate_path(path)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    print(f"web build (browser engine) on http://127.0.0.1:{port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=WEB)).serve_forever()
