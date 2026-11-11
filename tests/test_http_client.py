import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli


class MockComfyHandler(BaseHTTPRequestHandler):
    uploaded = None
    submitted = None

    def log_message(self, format, *args):
        pass

    def send_payload(self, code, obj, content_type='application/json'):
        raw = json.dumps(obj).encode('utf-8') if isinstance(obj, (dict, list)) else obj
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        size = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(size)
        if self.path == '/upload/image':
            type(self).uploaded = body
            self.send_payload(200, {'name': 'uploaded.webp', 'subfolder': '', 'type': 'input'})
        elif self.path == '/prompt':
            type(self).submitted = json.loads(body)
            self.send_payload(200, {'prompt_id': 'test-id'})
        else:
            self.send_payload(404, {'error': 'not found'})

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == '/history/test-id':
            self.send_payload(200, {'test-id': {'status': {'status_str': 'success'}, 'outputs': {
                '8': {'images': [{'filename': 'result.png', 'subfolder': '', 'type': 'output'}]}
            }}})
        elif parsed.path == '/view':
            self.send_payload(200, b'\x89PNG\r\n\x1a\nmockpng', 'image/png')
        else:
            self.send_payload(404, {'error': 'not found'})


class HTTPTests(unittest.TestCase):
    def test_complete_local_api_roundtrip(self):
        MockComfyHandler.uploaded = None
        MockComfyHandler.submitted = None
        server = ThreadingHTTPServer(('127.0.0.1', 0), MockComfyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                inp = Path(tmp) / 'cat.webp'
                out = Path(tmp) / 'cat.png'
                inp.write_bytes(b'fake-webp-content')
                args = cli.parse_args(['-i', str(inp), '-o', str(out),
                                       '--server', f'http://127.0.0.1:{server.server_port}',
                                       '--steps', '4', '--seed', '42', '--poll', '0.01'])
                self.assertEqual(cli.run(args), 0)
                self.assertTrue(out.read_bytes().startswith(b'\x89PNG'))
                self.assertIn(b'fake-webp-content', MockComfyHandler.uploaded)
                prompt = MockComfyHandler.submitted['prompt']
                self.assertEqual(prompt['1']['inputs']['image'], 'uploaded.webp')
                self.assertEqual(prompt['13']['inputs']['steps'], 4)
                self.assertEqual(prompt['13']['inputs']['seed'], 42)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
