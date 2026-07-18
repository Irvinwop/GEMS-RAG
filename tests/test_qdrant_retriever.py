from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gems_rag.retrieval import QdrantHashVectorRetriever
from gems_rag.types import QAItem


def _item() -> QAItem:
    return QAItem(
        qa_id="qdrant-test",
        question="standard sign legend",
        question_type=None,
        expected_refusal=False,
        gold_answer={},
        references=[],
    )


class TestQdrantRetriever(unittest.TestCase):
    def test_retrieve_releases_file_backed_client(self) -> None:
        hit = SimpleNamespace(
            payload={"chunk_id": "chunk-1", "text": "A standard sign legend."},
            score=0.9,
        )
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[hit])
        retriever = QdrantHashVectorRetriever(
            name="qdrant",
            chunks=[],
            qdrant_path=Path("unused"),
        )
        retriever._client = client

        result = retriever.retrieve(_item())

        self.assertEqual([evidence.evidence_id for evidence in result.evidence], ["chunk-1"])
        client.close.assert_called_once_with()
        self.assertIsNone(retriever._client)

    def test_retrieve_releases_client_after_query_failure(self) -> None:
        client = Mock()
        client.query_points.side_effect = RuntimeError("query failed")
        retriever = QdrantHashVectorRetriever(
            name="qdrant",
            chunks=[],
            qdrant_path=Path("unused"),
        )
        retriever._client = client

        with self.assertRaisesRegex(RuntimeError, "query failed"):
            retriever.retrieve(_item())

        client.close.assert_called_once_with()
        self.assertIsNone(retriever._client)


if __name__ == "__main__":
    unittest.main()
