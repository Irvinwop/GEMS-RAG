from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "openai_endpoint_router.py"
    spec = importlib.util.spec_from_file_location("openai_endpoint_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


router = _load_module()


class _UpstreamHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        payload = json.dumps(
            {"upstream": self.server.label, "path": self.path, "body": body}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


class OpenAIEndpointRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat = cls._start_upstream("chat")
        cls.embedding = cls._start_upstream("embedding")
        cls.proxy = router.build_server(
            "127.0.0.1",
            0,
            chat_url=f"http://127.0.0.1:{cls.chat.server_port}",
            embedding_url=f"http://127.0.0.1:{cls.embedding.server_port}",
            upstream_timeout_s=5,
        )
        cls.proxy_thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.proxy_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.proxy.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        for server in [cls.proxy, cls.chat, cls.embedding]:
            server.shutdown()
            server.server_close()

    @classmethod
    def _start_upstream(cls, label: str) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        server.label = label
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_routes_embedding_paths_to_embedding_server(self) -> None:
        request = Request(
            f"{self.base_url}/v1/embeddings",
            data=b'{"input":"test"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.load(response)
        self.assertEqual(payload["upstream"], "embedding")
        self.assertEqual(payload["path"], "/v1/embeddings")
        self.assertEqual(payload["body"], '{"input":"test"}')

    def test_routes_models_and_completions_to_chat_server(self) -> None:
        with urlopen(f"{self.base_url}/v1/models") as response:
            models = json.load(response)
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            completion = json.load(response)
        self.assertEqual(models["upstream"], "chat")
        self.assertEqual(completion["upstream"], "chat")

    def test_health_reports_both_upstreams(self) -> None:
        with urlopen(f"{self.base_url}/healthz") as response:
            payload = json.load(response)
        self.assertTrue(payload["ok"])
        self.assertIn(str(self.chat.server_port), payload["chat_url"])
        self.assertIn(str(self.embedding.server_port), payload["embedding_url"])


if __name__ == "__main__":
    unittest.main()
