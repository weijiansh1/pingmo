"""Start the persistent paper server without touching other services."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PORT = 8320
URL = f'http://127.0.0.1:{PORT}'


def health():
    try:
        with urlopen(URL + '/healthz', timeout=1) as response:
            value = json.load(response)
        return value if value.get('service') == 'pingmo-paper' else None
    except Exception:
        return None


def main():
    value = health()
    if value is None:
        with (ROOT / 'server.log').open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-u', str(ROOT / 'serve.py'), '--port', str(PORT)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True,
            )
        for _ in range(20):
            if process.poll() is not None:
                raise RuntimeError('Server exited; inspect server.log. No existing service was stopped.')
            value = health()
            if value:
                break
            time.sleep(0.2)
        if value is None:
            raise RuntimeError('Server did not become healthy; inspect server.log.')
    state = {'service': 'pingmo-paper', 'pid': value['pid'], 'port': PORT,
             'paper_revision': value['revision'],
             'jupyter_proxy_path': f'/proxy/{PORT}/',
             'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'server_state.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
