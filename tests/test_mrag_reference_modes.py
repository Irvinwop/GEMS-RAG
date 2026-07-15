from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gems_rag.mrag_reference_modes import retrieve_reference_mode


class _Vector(list):
    def tolist(self):
        return list(self)


class TestMragReferenceModes(unittest.TestCase):
    def test_reference_modes_replace_colliding_qdrant_payloads_with_canonical_chunks(self) -> None:
        class TextEmbedder:
            def encode_dense(self, texts):
                return [_Vector([0.1, 0.2])]

        class Client:
            def query_points(self, **kwargs):
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id=1,
                            score=0.75,
                            payload={"chunk_id": "collision", "text": "4.5"},
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
        canonical = {"chunk_id": "collision", "text": "The complete canonical provision."}

        result = retrieve_reference_mode(
            pipeline,
            "query",
            mode="dense",
            top_k=1,
            chunks=[canonical],
        )

        self.assertEqual(result["chunks"][0]["text"], canonical["text"])

    def test_dense_mode_deduplicates_colliding_qdrant_points(self) -> None:
        calls = []

        class TextEmbedder:
            def encode_dense(self, texts):
                return [_Vector([0.1, 0.2])]

        class Client:
            def query_points(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(id=1, score=0.9, payload={"chunk_id": "collision"}),
                        SimpleNamespace(id=2, score=0.8, payload={"chunk_id": "collision"}),
                        SimpleNamespace(id=3, score=0.7, payload={"chunk_id": "second"}),
                    ]
                )

        pipeline = SimpleNamespace(
            text=TextEmbedder(),
            store=SimpleNamespace(_client=Client()),
            image=None,
            kg=None,
            rerank=None,
        )
        chunks = [
            {"chunk_id": "collision", "text": "Canonical collision"},
            {"chunk_id": "second", "text": "Second result"},
        ]

        result = retrieve_reference_mode(
            pipeline,
            "query",
            mode="dense",
            top_k=2,
            chunks=chunks,
        )

        self.assertEqual([chunk["chunk_id"] for chunk in result["chunks"]], ["collision", "second"])
        self.assertEqual(calls[0]["limit"], 4)

    def test_full_mode_returns_empty_evidence_without_calling_reranker(self) -> None:
        class TextEmbedder:
            def encode_both(self, texts):
                return [_Vector([0.2, 0.4])], [{}]

        class Store:
            def search_chunks_hybrid(self, collection, dense, sparse, top_k):
                return []

        class KG:
            def query_entities(self, query):
                return set()

            def neighbors(self, node, n_hops=1):
                return set()

            def proximity_score(self, entities, chunk_id):
                return 0.0

        class Reranker:
            def rank(self, query, documents, top_k):
                raise AssertionError("empty candidates should not be reranked")

        pipeline = SimpleNamespace(
            text=TextEmbedder(),
            store=Store(),
            image=None,
            kg=KG(),
            rerank=Reranker(),
        )

        result = retrieve_reference_mode(
            pipeline,
            "no matching evidence",
            mode="no_visual",
            top_k=2,
            chunks=[],
        )

        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["debug"]["ranked_candidates"], 0)

    def test_dense_mode_queries_only_the_named_dense_vector(self) -> None:
        calls = []

        class TextEmbedder:
            def encode_dense(self, texts):
                self.texts = texts
                return [_Vector([0.1, 0.2])]

        class Client:
            def query_points(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id=1,
                            score=0.75,
                            payload={"chunk_id": "chunk-1", "text": "Dense evidence"},
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

        result = retrieve_reference_mode(
            pipeline,
            "dense query",
            mode="dense",
            top_k=3,
            chunks=[],
        )

        self.assertEqual(result["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(result["figures"], [])
        self.assertEqual(result["pages"], [])
        self.assertEqual(calls[0]["using"], "dense")
        self.assertEqual(calls[0]["limit"], 6)
        self.assertEqual(result["debug"]["components"], ["dense"])

    def test_hybrid_mode_uses_dense_sparse_fusion_without_graph_or_visuals(self) -> None:
        calls = []

        class TextEmbedder:
            def encode_both(self, texts):
                self.texts = texts
                return [_Vector([0.2, 0.4])], [{11: 0.7}]

        class Store:
            def search_chunks_hybrid(self, collection, dense, sparse, top_k):
                calls.append((collection, dense, sparse, top_k))
                return [
                    {
                        "id": 2,
                        "score": 0.03,
                        "payload": {"chunk_id": "chunk-2", "text": "Hybrid evidence"},
                    }
                ]

        pipeline = SimpleNamespace(
            text=TextEmbedder(),
            store=Store(),
            image=None,
            kg=None,
            rerank=None,
        )

        result = retrieve_reference_mode(
            pipeline,
            "hybrid query",
            mode="hybrid",
            top_k=4,
            chunks=[],
        )

        self.assertEqual(result["chunks"][0]["chunk_id"], "chunk-2")
        self.assertEqual(calls[0][0], "mutcd_chunks")
        self.assertEqual(calls[0][2], {11: 0.7})
        self.assertEqual(calls[0][3], 8)
        self.assertEqual(result["debug"]["components"], ["dense", "sparse"])

    def test_multimodal_mode_adds_direct_figure_and_page_retrieval_without_graph(self) -> None:
        class TextEmbedder:
            def encode_both(self, texts):
                return [_Vector([0.2, 0.4])], [{11: 0.7}]

        class ImageEmbedder:
            def encode_queries(self, texts):
                return [_Vector([[0.5, 0.6]])]

        class Store:
            def search_chunks_hybrid(self, collection, dense, sparse, top_k):
                return [{"id": 1, "score": 0.03, "payload": {"chunk_id": "seed", "text": "Seed"}}]

            def search_figures(self, collection, dense, top_k):
                return [SimpleNamespace(score=0.6, payload={"figure_id": "Figure 1A-1", "caption": "Caption"})]

            def search_figures_visual(self, collection, query, top_k):
                return [
                    SimpleNamespace(
                        score=0.9,
                        payload={
                            "figure_id": "Figure 2A-1",
                            "caption": "Visual",
                            "image_path": "/content/drive/MyDrive/MRAG/figures/figure_2A-1.png",
                        },
                    )
                ]

            def search_pages(self, collection, query, top_k):
                return [
                    SimpleNamespace(
                        score=0.8,
                        payload={"page_pdf": 12, "image_path": "/content/drive/MyDrive/MRAG/page_images/page_0012.png"},
                    )
                ]

        with tempfile.TemporaryDirectory() as td:
            mrag_dir = Path(td)
            figure_path = mrag_dir / "figures" / "figure_2A-1.png"
            page_path = mrag_dir / "page_images" / "page_0012.png"
            figure_path.parent.mkdir()
            page_path.parent.mkdir()
            figure_path.write_bytes(b"figure")
            page_path.write_bytes(b"page")
            pipeline = SimpleNamespace(
                text=TextEmbedder(),
                store=Store(),
                image=ImageEmbedder(),
                kg=None,
                rerank=None,
                mrag_dir=mrag_dir,
            )

            result = retrieve_reference_mode(
                pipeline,
                "visual query",
                mode="multimodal",
                top_k=4,
                chunks=[],
            )

        self.assertEqual([figure["figure_id"] for figure in result["figures"]], ["Figure 2A-1", "Figure 1A-1"])
        self.assertEqual(result["figures"][0]["image_path"], str(figure_path.resolve()))
        self.assertEqual(
            result["figures"][0]["upstream_image_path"],
            "/content/drive/MyDrive/MRAG/figures/figure_2A-1.png",
        )
        self.assertEqual(result["pages"][0]["page_pdf"], 12)
        self.assertEqual(result["pages"][0]["image_path"], str(page_path.resolve()))
        self.assertEqual(result["debug"]["components"], ["dense", "sparse", "visual"])
        self.assertNotIn("graph", result["debug"]["components"])

    def test_full_mode_adds_graph_neighbor_chunks_before_reranking(self) -> None:
        class TextEmbedder:
            def encode_both(self, texts):
                return [_Vector([0.2, 0.4])], [{11: 0.7}]

        class ImageEmbedder:
            def encode_queries(self, texts):
                return [_Vector([[0.5, 0.6]])]

        class Store:
            def search_chunks_hybrid(self, collection, dense, sparse, top_k):
                return [
                    {
                        "id": 1,
                        "score": 0.03,
                        "payload": {"chunk_id": "seed", "text": "Seed evidence", "content_type": "Support"},
                    }
                ]

            def search_figures(self, collection, dense, top_k):
                return []

            def search_figures_visual(self, collection, query, top_k):
                return []

            def search_pages(self, collection, query, top_k):
                return []

        class KG:
            def __init__(self):
                self.g = SimpleNamespace(
                    nodes={
                        "figure:Figure 9A-1": {
                            "id": "Figure 9A-1",
                            "caption": "Linked figure",
                            "image_path": "figure.png",
                        }
                    }
                )

            def query_entities(self, query):
                return {"section:9A.01"}

            def neighbors(self, node, n_hops=1):
                if node in {"chunk:seed", "section:9A.01"}:
                    return {node, "chunk:seed", "chunk:expanded"}
                return {node}

            def proximity_score(self, entities, chunk_id):
                return 1.0 if chunk_id == "expanded" else 0.5

            def figures_for_chunk(self, chunk_id):
                return ["Figure 9A-1"] if chunk_id == "expanded" else []

            def figure(self, figure_id):
                return f"figure:{figure_id}"

        class Reranker:
            def rank(self, query, documents, top_k):
                expanded_index = next(index for index, text in enumerate(documents) if "Expanded" in text)
                return [(expanded_index, 0.95)]

        pipeline = SimpleNamespace(
            text=TextEmbedder(),
            store=Store(),
            image=ImageEmbedder(),
            kg=KG(),
            rerank=Reranker(),
        )
        chunks = [
            {
                "chunk_id": "expanded",
                "text": "Expanded graph evidence",
                "content_type": "Standard",
                "part": "Part 9",
                "chapter": "Chapter 9A",
            }
        ]

        result = retrieve_reference_mode(
            pipeline,
            "What does Section 9A.01 require in Part 9?",
            mode="full",
            top_k=2,
            chunks=chunks,
        )

        self.assertEqual(result["chunks"][0]["chunk_id"], "expanded")
        self.assertEqual(result["figures"][0]["figure_id"], "Figure 9A-1")
        self.assertEqual(result["debug"]["graph_expanded_chunks"], 1)
        self.assertIn("graph", result["debug"]["components"])
        self.assertIn("rule_ranking", result["debug"]["components"])


if __name__ == "__main__":
    unittest.main()
