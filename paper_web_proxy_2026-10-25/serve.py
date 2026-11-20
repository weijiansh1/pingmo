"""Serve the user's original delivery unchanged through the Jupyter proxy."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import json
import os

WORKSPACE = Path(__file__).resolve().parents[1]
PAPER = WORKSPACE / 'flight_control_original_restored_2026-10-25'
DISPLAY = WORKSPACE / 'paper_web_display_2026-10-25'
MATH = WORKSPACE / 'flight_control_math_primer_2026-10-24'
SUPPORT = {
    '02_品模与强化学习深查.md': WORKSPACE / 'flight_control_corrected_2026-10-24/02_品模与强化学习深查.md',
    '03_控制器路线验证与最佳实践.md': WORKSPACE / 'flight_control_corrected_2026-10-24/03_控制器路线验证与最佳实践.md',
    'rl-guide.md': WORKSPACE / 'gjb_object_library_audit_2026-10-24/强化学习逐步说明.md',
}


class PaperHandler(SimpleHTTPRequestHandler):
    server_version = 'PingmoPaper/1.0'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PAPER), **kwargs)

    def list_directory(self, path):
        self.send_error(404, 'Not found')
        return None

    def send_bytes(self, body, content_type, *, head=False):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def route(self, *, head=False):
        path = unquote(urlsplit(self.path).path)
        if path == '/':
            self.send_response(302)
            # Relative redirects preserve the /proxy/<port>/ prefix.
            self.send_header('Location', 'webpage/dhc6_report.html')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path == '/healthz':
            payload = json.dumps({'service': 'pingmo-paper', 'revision': 'original-display-fix-2026-10-25',
                                  'pid': os.getpid()}, ensure_ascii=False).encode()
            self.send_bytes(payload, 'application/json; charset=utf-8', head=head)
            return
        if path in ('/webpage/dhc6_report.html', '/webpage/dhc6_paper.html'):
            display = DISPLAY / path.lstrip('/')
            self.send_bytes((display if display.is_file() else PAPER / path.lstrip('/')).read_bytes(),
                            'text/html; charset=utf-8', head=head)
            return
        if path == '/paper.pdf':
            self.send_error(404, 'The original delivery contains HTML, not a PDF')
            return
        if path == '/math':
            self.send_response(302)
            self.send_header('Location', 'math/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path in ('/math/', '/math/数学直观入门.html'):
            text = (MATH / '数学直观入门.html').read_text()
            text = text.replace('../gjb_object_library_audit_2026-10-24/强化学习逐步说明.md',
                                '../support/rl-guide.md')
            self.send_bytes(text.encode(), 'text/html; charset=utf-8', head=head)
            return
        if path == '/math/从滚转例子读懂数学.md':
            self.send_bytes((MATH / '从滚转例子读懂数学.md').read_bytes(),
                            'text/plain; charset=utf-8', head=head)
            return
        if path.startswith('/support/') and path.removeprefix('/support/') in SUPPORT:
            self.send_bytes(SUPPORT[path.removeprefix('/support/')].read_bytes(),
                            'text/plain; charset=utf-8', head=head)
            return
        display = (DISPLAY / path.lstrip('/')).resolve()
        if display.is_relative_to((DISPLAY / 'webpage').resolve()) and display.is_file():
            self.send_bytes(display.read_bytes(), self.guess_type(str(display)), head=head)
            return
        candidate = (PAPER / path.lstrip('/')).resolve()
        allowed = (PAPER / 'webpage').resolve()
        if not candidate.is_relative_to(allowed) or not candidate.is_file():
            self.send_error(404, 'Not found')
            return
        if head:
            super().do_HEAD()
        else:
            super().do_GET()

    def do_GET(self):
        self.route()

    def do_HEAD(self):
        self.route(head=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8320)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), PaperHandler)
    print(f'Pingmo paper serving on 127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
