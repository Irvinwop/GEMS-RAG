from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from gems_rag.mrag_reference_server import (
    CachedReranker,
    MragReferenceServerState,
    request_reference_socket,
    serve_reference_socket,
)


class _Vector(list):
    def tolist(self):
        return list(self)


class TestMragReferenceServer(unittest.TestCase):
    def test_cached_reranker_only_scores_new_query_document_pairs(self) -> None:
        calls = []

        class Reranker:
            def rank(self, query, documents, top_k):
                calls.append(list(documents))
                return [(index, float(len(document))) for index, document in enumerate(documents)]

        reranker = CachedReranker(Reranker())

        first = reranker.rank("query", ["a", "long"], top_k=2)
        second = reranker.rank("query", ["long", "medium"], top_k=2)

        self.assertEqual(first, [(1, 4.0), (0, 1.0)])
        self.assertEqual(second, [(1, 6.0), (0, 4.0)])
        self.assertEqual(calls, [["a", "long"], ["medium"]])

    def test_server_reuses_cached_mode_query_and_stops_cleanly(self) -> None:
        calls = []

        class TextEmbedder:
            def encode_dense(self, texts):
                calls.append(list(texts))
                return [_Vector([0.1, 0.2])]

        class Client:
            def query_points(self, **kwargs):
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id=1,
                            score=0.8,
                            payload={"chunk_id": "chunk-1", "text": "Evidence"},
                        )
                    ]
                )

        pipeline = SimpleNamespace(
            text=TextEmbedder(),
            store=SimpleNamespace(_client=Client()),
            image=None,
            kg=None,
            rerank=None,
        )
        with tempfile.TemporaryDirectory() as td:
            socket_path = Path(td) / "mrag.sock"
            state = MragReferenceServerState(
                pipeline=pipeline,
                chunks=[],
                mrag_dir=Path(td),
                fingerprint="test-fingerprint",
            )
            thread = threading.Thread(
                target=serve_reference_socket,
                args=(socket_path, state),
                kwargs={"idle_timeout_s": 10},
                daemon=True,
            )
            thread.start()
            deadline = time.monotonic() + 2
            while not socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)

            health = request_reference_socket(socket_path, {"action": "health"}, timeout_s=1)
            first = request_reference_socket(
                socket_path,
                {"action": "retrieve", "mode": "dense", "question": "Question", "top_k": 1},
                timeout_s=1,
            )
            second = request_reference_socket(
                socket_path,
                {"action": "retrieve", "mode": "dense", "question": "Question", "top_k": 1},
                timeout_s=1,
            )
            stopped = request_reference_socket(socket_path, {"action": "stop"}, timeout_s=1)
            thread.join(timeout=2)

        self.assertEqual(health["fingerprint"], "test-fingerprint")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["result"]["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(calls, [["Question"]])
        self.assertEqual(stopped["status"], "stopping")
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
