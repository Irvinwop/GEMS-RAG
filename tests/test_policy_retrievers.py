from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.config import RetrieverConfig
from gem_rags.retrieval import build_retriever
from gem_rags.types import QAItem


def _fixture_mrag(root: Path) -> Path:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    chunk = {
        "chunk_id": "chunk-1",
        "section_id": "2A.01",
        "section_title": "Function and Purpose",
        "content_type": "Standard",
        "ordinal": 1,
        "page_printed": "2",
        "part": "Part 2",
        "text": "A customterm permit sign shall fulfill a demonstrated need.",
    }
    (cache / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    return mrag_dir


def _qa(question: str = "customterm") -> QAItem:
    return QAItem(
        qa_id="qa_policy",
        question=question,
        question_type=None,
        expected_refusal=False,
        gold_answer={},
        references=[],
    )


class TestPolicyRetrievers(unittest.TestCase):
    def test_self_rag_policy_honors_nested_base_retriever_and_keyword_options(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mrag_dir = _fixture_mrag(Path(td))
            retriever = build_retriever(
                RetrieverConfig(
                    name="self_nested",
                    kind="self_rag_policy",
                    top_k=8,
                    options={
                        "mode": "adaptive_retrieval",
                        "threshold": 0.4,
                        "retrieval_keywords": ["customterm"],
                        "base_retriever": {
                            "name": "tiny_hash",
                            "kind": "hash_vector",
                            "top_k": 1,
                            "options": {"dims": 16},
                        },
                    },
                ),
                mrag_dir,
            )
            result = retriever.retrieve(_qa())

        self.assertEqual(result.debug["decision"], "retrieve")
        self.assertEqual(result.debug["base_adapter"], "tiny_hash")
        self.assertEqual(result.debug["vector_dims"], 16)
        self.assertEqual(result.debug["retrieval_keywords"], ["customterm"])
        self.assertEqual(len(result.evidence), 1)

    def test_crag_policy_honors_nested_retrievers_and_threshold_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mrag_dir = _fixture_mrag(Path(td))
            retriever = build_retriever(
                RetrieverConfig(
                    name="crag_nested",
                    kind="crag_policy",
                    top_k=8,
                    options={
                        "primary_retriever": {"name": "primary_bm25", "kind": "bm25", "top_k": 1},
                        "fallback_retriever": {
                            "name": "fallback_hash",
                            "kind": "hash_vector",
                            "top_k": 1,
                            "options": {"dims": 32},
                        },
                        "confidence_threshold": 0.99,
                        "fallback_threshold": 0.95,
                    },
                ),
                mrag_dir,
            )
            result = retriever.retrieve(_qa("customterm permit sign"))

        self.assertEqual(result.debug["action"], "fallback")
        self.assertEqual(result.debug["primary_adapter"], "primary_bm25")
        self.assertEqual(result.debug["fallback_adapter"], "fallback_hash")
        self.assertEqual(result.debug["fallback_debug"]["vector_dims"], 32)
        self.assertEqual(result.debug["accept_threshold"], 0.99)
        self.assertEqual(result.debug["reject_threshold"], 0.95)


if __name__ == "__main__":
    unittest.main()
