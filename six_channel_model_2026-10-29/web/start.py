"""Start the six-channel viewer on 127.0.0.1:8330 without touching other services."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
PORT = 8330
URL = f"http://127.0.0.1:{PORT}"
SERVICE = "pingmo-six-channel"
# the shell exports HTTP_PROXY; a localhost health check must not go to that proxy
LOCAL = build_opener(ProxyHandler({}))


def health():
    try:
        with LOCAL.open(URL + "/healthz", timeout=1) as response:
            value = json.load(response)
        return value if value.get("service") == SERVICE else None
    except Exception:
        return None


def main() -> None:
    value = health()
    if value is None:
        with (ROOT / "server.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "serve.py"), "--port", str(PORT)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        for _ in range(600):                      # loading 5,000 objects takes a few seconds
            if process.poll() is not None:
                raise RuntimeError("Server exited; inspect server.log. No existing service was stopped.")
            value = health()
            if value:
                break
            time.sleep(0.2)
        if value is None:
            raise RuntimeError("Server did not become healthy; inspect server.log.")
    state = {"service": SERVICE, "pid": value["pid"], "port": PORT, "revision": value["revision"],
             "jupyter_proxy_path": f"/proxy/{PORT}/", "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "server_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
