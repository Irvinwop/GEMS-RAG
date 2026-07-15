from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from gems_rag.visrag_server import VisragServerState, request_visrag_socket, serve_visrag_socket


class TestVisragServer(unittest.TestCase):
    def test_server_reuses_exact_query_and_stops_cleanly(self) -> None:
        calls = []

        def query(question: str, top_k: int):
            calls.append((question, top_k))
            return {"question": question, "contexts": [{"name": "page:0001", "score": 1.0}]}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            socket_path = root / "visrag.sock"
            state = VisragServerState(
                query_func=query,
                fingerprint="fingerprint",
                manifest=root / "manifest.jsonl",
                embeddings=root / "embeddings.npy",
                model_name_or_path="openbmb/VisRAG-Ret",
                model_revision="revision",
                max_cache_entries=2,
            )
            thread = threading.Thread(
                target=serve_visrag_socket,
                args=(socket_path, state),
                kwargs={"idle_timeout_s": 5.0},
                daemon=True,
            )
            thread.start()
            deadline = time.monotonic() + 2.0
            while not socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)

            health = request_visrag_socket(socket_path, {"action": "health"}, timeout_s=1.0)
            first = request_visrag_socket(
                socket_path,
                {"action": "query", "question": "What is required?", "top_k": 3},
                timeout_s=1.0,
            )
            second = request_visrag_socket(
                socket_path,
                {"action": "query", "question": "What is required?", "top_k": 3},
                timeout_s=1.0,
            )
            stopped = request_visrag_socket(socket_path, {"action": "stop"}, timeout_s=1.0)
            thread.join(timeout=2.0)

        self.assertEqual(health["fingerprint"], "fingerprint")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(calls, [("What is required?", 3)])
        self.assertEqual(stopped["status"], "stopping")
        self.assertFalse(thread.is_alive())

    def test_state_cache_respects_entry_limit(self) -> None:
        state = VisragServerState(
            query_func=lambda question, top_k: {"question": question, "top_k": top_k},
            fingerprint="fingerprint",
            manifest=Path("manifest.jsonl"),
            embeddings=Path("embeddings.npy"),
            model_name_or_path="model",
            model_revision="revision",
            max_cache_entries=1,
        )

        state.query("first", 1)
        state.query("second", 1)

        self.assertEqual(list(state.cache), [("second", 1)])


if __name__ == "__main__":
    unittest.main()
