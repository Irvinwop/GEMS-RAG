#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
EMBEDDING_PATHS = {"/embeddings", "/api/embed", "/api/embeddings"}


class OpenAIEndpointRouter(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        chat_url: str,
        embedding_url: str,
        upstream_timeout_s: float,
    ) -> None:
        super().__init__(address, OpenAIEndpointRouterHandler)
        self.chat_url = chat_url.rstrip("/")
        self.embedding_url = embedding_url.rstrip("/")
        self.upstream_timeout_s = upstream_timeout_s


class OpenAIEndpointRouterHandler(BaseHTTPRequestHandler):
    server: OpenAIEndpointRouter

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _handle(self) -> None:
        if self.path.split("?", 1)[0] == "/healthz":
            self._send_json(
                200,
                {
                    "ok": True,
                    "chat_url": self.server.chat_url,
                    "embedding_url": self.server.embedding_url,
                },
            )
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        target = self._target_url()
        request = Request(
            target,
            data=body,
            headers={
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
            },
            method=self.command,
        )
        try:
            with urlopen(request, timeout=self.server.upstream_timeout_s) as response:
                self._relay(response.status, response.headers, response.read())
        except HTTPError as exc:
            self._relay(exc.code, exc.headers, exc.read())
        except (URLError, TimeoutError, OSError) as exc:
            self._send_json(502, {"error": "upstream_unavailable", "detail": repr(exc), "target": target})

    def _target_url(self) -> str:
        path = self.path.split("?", 1)[0].rstrip("/")
        origin = self.server.embedding_url if any(path.endswith(suffix) for suffix in EMBEDDING_PATHS) else self.server.chat_url
        return f"{origin}{self.path}"

    def _relay(self, status: int, headers: Any, body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_server(
    host: str,
    port: int,
    *,
    chat_url: str,
    embedding_url: str,
    upstream_timeout_s: float = 3600,
) -> OpenAIEndpointRouter:
    return OpenAIEndpointRouter(
        (host, port),
        chat_url=chat_url,
        embedding_url=embedding_url,
        upstream_timeout_s=upstream_timeout_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route one OpenAI-compatible URL to separate chat/vision and embedding servers."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--chat-url", default="http://127.0.0.1:11435")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:11436")
    parser.add_argument("--upstream-timeout-s", type=float, default=3600)
    args = parser.parse_args()

    server = build_server(
        args.host,
        args.port,
        chat_url=args.chat_url,
        embedding_url=args.embedding_url,
        upstream_timeout_s=args.upstream_timeout_s,
    )
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{server.server_port}",
                "chat_url": server.chat_url,
                "embedding_url": server.embedding_url,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
