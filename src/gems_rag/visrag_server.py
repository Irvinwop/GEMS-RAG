from __future__ import annotations

import copy
import json
import os
import socket
import socketserver
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 64 * 1_048_576


class VisragServerError(RuntimeError):
    pass


@dataclass
class VisragServerState:
    query_func: Callable[[str, int], dict[str, Any]]
    fingerprint: str
    manifest: Path
    embeddings: Path
    model_name_or_path: str
    model_revision: str
    max_cache_entries: int = 2048
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    cache: OrderedDict[tuple[str, int], dict[str, Any]] = field(default_factory=OrderedDict)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ready",
            "pid": os.getpid(),
            "fingerprint": self.fingerprint,
            "manifest": str(self.manifest.resolve()),
            "embeddings": str(self.embeddings.resolve()),
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "started_at": self.started_at,
            "cache_entries": len(self.cache),
        }

    def query(self, question: str, top_k: int) -> tuple[dict[str, Any], bool]:
        key = (question, top_k)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return copy.deepcopy(cached), True
        result = self.query_func(question, top_k)
        self.cache[key] = copy.deepcopy(result)
        self.cache.move_to_end(key)
        while len(self.cache) > max(0, self.max_cache_entries):
            self.cache.popitem(last=False)
        return result, False


class _VisragUnixServer(socketserver.UnixStreamServer):
    def __init__(self, socket_path: str, state: VisragServerState):
        self.state = state
        self.stop_requested = False
        self.last_activity = time.monotonic()
        super().__init__(socket_path, _VisragRequestHandler)


class _VisragRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.server.last_activity = time.monotonic()
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            self._write({"ok": False, "error": "request_too_large"})
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            response = self._dispatch(request)
        except Exception as exc:
            traceback.print_exc()
            response = {
                "ok": False,
                "error": "request_failed",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        self._write(response)

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        action = str(request.get("action") or "")
        if action == "health":
            return self.server.state.health()
        if action == "stop":
            self.server.stop_requested = True
            return {"ok": True, "status": "stopping"}
        if action != "query":
            raise ValueError(f"unsupported action: {action!r}")
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        top_k = int(request.get("top_k", 6))
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        result, cache_hit = self.server.state.query(question, top_k)
        return {"ok": True, "result": result, "cache_hit": cache_hit}

    def _write(self, payload: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")


def serve_visrag_socket(
    socket_path: Path,
    state: VisragServerState,
    *,
    idle_timeout_s: float,
) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = _VisragUnixServer(str(socket_path), state)
    server.timeout = min(1.0, max(0.05, idle_timeout_s))
    try:
        while not server.stop_requested:
            server.handle_request()
            if time.monotonic() - server.last_activity >= idle_timeout_s:
                break
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)


def request_visrag_socket(
    socket_path: Path,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    if len(encoded) > MAX_REQUEST_BYTES:
        raise VisragServerError("request exceeds the VisRAG server limit")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_s)
            client.connect(str(socket_path))
            client.sendall(encoded)
            with client.makefile("rb") as response:
                raw = response.readline(MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        raise VisragServerError(f"VisRAG server request failed: {exc}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise VisragServerError("response exceeds the VisRAG server limit")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisragServerError("VisRAG server returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise VisragServerError("VisRAG server returned a non-object response")
    return decoded
