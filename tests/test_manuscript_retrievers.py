from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from gem_rags.manuscript_retrievers import (
    KG2RAGRetriever,
    M3KGRAGRetriever,
    MultimodalCandidateRetriever,
    OKHRAGRetriever,
    SAMRAGRetriever,
)
from gem_rags.config import RetrieverConfig
from gem_rags.retrieval import BM25Retriever, build_retriever
from gem_rags.types import Evidence, QAItem, RetrievalResult


def _item(question: str) -> QAItem:
    return QAItem(
        qa_id="qa-1",
        question=question,
        question_type=None,
        expected_refusal=False,
        gold_answer={},
        references=[],
    )


class TestManuscriptRetrievers(unittest.TestCase):
    def test_build_retriever_exposes_all_local_manuscript_algorithms(self) -> None:
        chunk = {
            "chunk_id": "seed",
            "section_id": "2B.01",
            "section_title": "STOP sign",
            "content_type": "Standard",
            "ordinal": 1,
            "page_pdf": 1,
            "page_printed": "1",
            "part": "Part 2",
            "text": "A complete stop is required.",
        }
        figure = {"figure_id": "Figure 2B-1", "caption": "STOP sign", "image_path": "stop.png"}
        graph = nx.MultiDiGraph()
        graph.add_edge("chunk:seed", "section:2B.01", label="belongs_to")
        graph.add_edge("section:2B.01", "figure:Figure 2B-1", label="illustrated_by")

        with tempfile.TemporaryDirectory() as td:
            mrag_dir = Path(td)
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir()
            (cache / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
            (cache / "figures.jsonl").write_text(json.dumps(figure) + "\n", encoding="utf-8")
            with (cache / "graph.gpickle").open("wb") as handle:
                pickle.dump(graph, handle)

            expected = {
                "kg2rag": "KG2RAGRetriever",
                "m3kg_rag": "M3KGRAGRetriever",
                "okh_rag": "OKHRAGRetriever",
                "sam_rag": "SAMRAGRetriever",
            }
            built = {
                kind: build_retriever(RetrieverConfig(name=kind, kind=kind, top_k=3), mrag_dir)
                for kind in expected
            }

        self.assertEqual({kind: type(retriever).__name__ for kind, retriever in built.items()}, expected)

    def test_multimodal_candidates_include_query_aligned_figures(self) -> None:
        chunks = [
            {
                "chunk_id": "text",
                "section_id": "2B.01",
                "section_title": "STOP sign",
                "content_type": "Standard",
                "ordinal": 1,
                "page_pdf": 1,
                "page_printed": "1",
                "part": "Part 2",
                "text": "A stop is required.",
            }
        ]
        figures = [
            {"figure_id": "Figure 2B-1", "caption": "STOP sign visual", "image_path": "stop.png"},
            {"figure_id": "Figure 8A-1", "caption": "Unrelated freeway map", "image_path": "map.png"},
        ]
        text = BM25Retriever("text", chunks, top_k=2)
        retriever = MultimodalCandidateRetriever("mm", text, figures, top_k=3)

        result = retriever.retrieve(_item("What does a STOP sign require?"))

        self.assertIn("Figure 2B-1", [evidence.evidence_id for evidence in result.evidence])
        self.assertNotIn("Figure 8A-1", [evidence.evidence_id for evidence in result.evidence])
        self.assertEqual(result.debug["modalities"], ["text", "visual"])

    def test_okhrag_returns_graph_evidence_as_an_ordered_section_trajectory(self) -> None:
        chunks = [
            {
                "chunk_id": "before",
                "section_id": "4D.01",
                "section_title": "Signal sequence",
                "content_type": "Support",
                "ordinal": 1,
                "page_pdf": 10,
                "page_printed": "10",
                "part": "Part 4",
                "text": "First establish the applicable condition.",
            },
            {
                "chunk_id": "seed",
                "section_id": "4D.01",
                "section_title": "Signal sequence",
                "content_type": "Standard",
                "ordinal": 2,
                "page_pdf": 10,
                "page_printed": "10",
                "part": "Part 4",
                "text": "The signal sequence shall display red next.",
            },
            {
                "chunk_id": "after",
                "section_id": "4D.01",
                "section_title": "Signal sequence",
                "content_type": "Guidance",
                "ordinal": 3,
                "page_pdf": 10,
                "page_printed": "10",
                "part": "Part 4",
                "text": "Finally apply the clearance interval.",
            },
        ]
        graph = nx.MultiDiGraph()
        for chunk_id in ["before", "seed", "after"]:
            graph.add_edge(f"chunk:{chunk_id}", "section:4D.01", label="belongs_to")
        seed = BM25Retriever("seed", chunks, top_k=1)
        retriever = OKHRAGRetriever("okh-rag", seed, chunks, graph, top_k=3, graph_hops=2)

        result = retriever.retrieve(_item("What red signal sequence shall be displayed?"))

        self.assertEqual([evidence.evidence_id for evidence in result.evidence], ["before", "seed", "after"])
        self.assertEqual([evidence.metadata["trajectory_index"] for evidence in result.evidence], [0, 1, 2])
        self.assertEqual(result.evidence[1].metadata["preceded_by"], "before")
        self.assertEqual(result.debug["precedence_source"], "document_order")
        self.assertEqual(result.debug["implementation"], "paper_spec_no_public_code")

    def test_okhrag_centers_a_bounded_trajectory_on_the_relevant_seed(self) -> None:
        chunks = []
        graph = nx.MultiDiGraph()
        for ordinal in range(1, 7):
            chunk_id = f"chunk-{ordinal}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section_id": "4D.02",
                    "section_title": "Long sequence",
                    "content_type": "Standard",
                    "ordinal": ordinal,
                    "page_pdf": 20,
                    "page_printed": "20",
                    "part": "Part 4",
                    "text": "unique amber transition" if ordinal == 4 else f"generic provision {ordinal}",
                }
            )
            graph.add_edge(f"chunk:{chunk_id}", "section:4D.02", label="belongs_to")
        seed = BM25Retriever("seed", chunks, top_k=1)
        retriever = OKHRAGRetriever("okh-rag", seed, chunks, graph, top_k=3, graph_hops=2)

        result = retriever.retrieve(_item("What is the unique amber transition?"))

        self.assertEqual(
            [evidence.evidence_id for evidence in result.evidence],
            ["chunk-3", "chunk-4", "chunk-5"],
        )

    def test_m3kgrag_lifts_modality_seeds_and_prunes_off_topic_graph_evidence(self) -> None:
        chunks = [
            {
                "chunk_id": "seed",
                "section_id": "2B.01",
                "section_title": "STOP sign",
                "content_type": "Standard",
                "ordinal": 1,
                "page_pdf": 1,
                "page_printed": "1",
                "part": "Part 2",
                "text": "The STOP sign is a regulatory sign.",
            },
            {
                "chunk_id": "relevant-neighbor",
                "section_id": "2B.01",
                "section_title": "STOP sign",
                "content_type": "Standard",
                "ordinal": 2,
                "page_pdf": 2,
                "page_printed": "2",
                "part": "Part 2",
                "text": "A complete stop is required before proceeding.",
            },
            {
                "chunk_id": "noise-neighbor",
                "section_id": "2B.01",
                "section_title": "History",
                "content_type": "Support",
                "ordinal": 3,
                "page_pdf": 3,
                "page_printed": "3",
                "part": "Part 2",
                "text": "Unrelated decorative color history.",
            },
        ]
        figures = [
            {
                "figure_id": "Figure 2B-1",
                "caption": "STOP sign visual",
                "image_path": "stop.png",
                "page_pdf": 2,
            }
        ]
        graph = nx.MultiDiGraph()
        for chunk_id in ["seed", "relevant-neighbor", "noise-neighbor"]:
            graph.add_edge(f"chunk:{chunk_id}", "section:2B.01", label="belongs_to")
        graph.add_edge("section:2B.01", "figure:Figure 2B-1", label="illustrated_by")
        seed = BM25Retriever("text-seed", chunks, top_k=1)
        retriever = M3KGRAGRetriever(
            "m3kg-rag",
            seed,
            chunks,
            figures,
            graph,
            top_k=4,
            graph_hops=2,
            presence_threshold=0.2,
        )

        result = retriever.retrieve(_item("What complete stop does the STOP sign require?"))
        ids = [evidence.evidence_id for evidence in result.evidence]

        self.assertIn("seed", ids)
        self.assertIn("relevant-neighbor", ids)
        self.assertIn("Figure 2B-1", ids)
        self.assertNotIn("noise-neighbor", ids)
        self.assertIn("noise-neighbor", result.debug["grasp_pruned_ids"])
        self.assertEqual(result.debug["implementation"], "paper_spec_no_public_code")

    def test_samrag_scans_batches_until_relevance_verification_succeeds(self) -> None:
        class CandidateRetriever:
            name = "multimodal-candidates"

            def retrieve(self, item):
                return RetrievalResult(
                    adapter=self.name,
                    query=item.question,
                    evidence=[
                        Evidence("noise-1", "chunk", "Unrelated bicycle history.", score=0.9),
                        Evidence("noise-2", "figure", "Decorative illustration.", score=0.8),
                        Evidence("relevant", "chunk", "A complete stop is required by the STOP sign.", score=0.7),
                        Evidence("unused", "chunk", "This later batch should not be inspected.", score=0.6),
                    ],
                )

        retriever = SAMRAGRetriever(
            "sam-rag",
            CandidateRetriever(),
            top_k=3,
            batch_size=2,
            relevance_threshold=0.2,
        )

        result = retriever.retrieve(_item("What does a STOP sign require?"))

        self.assertEqual([evidence.evidence_id for evidence in result.evidence], ["relevant"])
        self.assertEqual(result.debug["batches_scanned"], 2)
        self.assertTrue(result.debug["stopped_after_relevant_batch"])
        self.assertEqual(result.debug["verification"]["relevant"]["isRel"], True)

    def test_kg2rag_expands_semantic_seeds_to_graph_neighbor_chunks(self) -> None:
        chunks = [
            {
                "chunk_id": "seed",
                "section_id": "1A.01",
                "section_title": "Stop requirements",
                "content_type": "Standard",
                "ordinal": 1,
                "page_pdf": 1,
                "page_printed": "1",
                "part": "Part 1",
                "text": "A stop sign requires a complete stop.",
            },
            {
                "chunk_id": "neighbor",
                "section_id": "1A.01",
                "section_title": "Stop requirements",
                "content_type": "Guidance",
                "ordinal": 2,
                "page_pdf": 2,
                "page_printed": "2",
                "part": "Part 1",
                "text": "The related placement provision has no shared query terms.",
            },
        ]
        graph = nx.MultiDiGraph()
        graph.add_edge("chunk:seed", "section:1A.01", label="belongs_to")
        graph.add_edge("chunk:neighbor", "section:1A.01", label="belongs_to")
        seed = BM25Retriever("seed", chunks, top_k=1)
        retriever = KG2RAGRetriever("kg2rag", seed, chunks, graph, top_k=2, graph_hops=2)

        result = retriever.retrieve(_item("What does the stop sign require?"))

        self.assertEqual([evidence.evidence_id for evidence in result.evidence], ["seed", "neighbor"])
        self.assertEqual(result.debug["seed_chunk_ids"], ["seed"])
        self.assertEqual(result.debug["expanded_chunk_ids"], ["neighbor"])
        self.assertEqual(result.debug["implementation"], "mutcd_adaptation_of_official_algorithm")


if __name__ == "__main__":
    unittest.main()
