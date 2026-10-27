"""Start Carbo Metrics:  python run.py  [--port 8765] [--no-browser]"""

import argparse
import threading
import webbrowser

import uvicorn

from scarbon.server import app, warm_up

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    threading.Thread(target=warm_up, daemon=True).start()
    url = f"http://127.0.0.1:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Carbo Metrics on {url}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
