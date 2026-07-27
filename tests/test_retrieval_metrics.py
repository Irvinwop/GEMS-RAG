from __future__ import annotations

import math
import unittest

from gems_rag.retrieval_metrics import (
    binary_ranking_metrics,
    build_chunk_qrels,
    expand_section_annotations,
    score_canonical_rankings,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_section_ranges_expand_against_known_corpus_sections(self) -> None:
        sections = {"2B.11", "2B.12", "2B.13", "2B.17", "2B.18"}

        expanded = expand_section_annotations(
            ["2B.12-2B.17", "2B.18"],
            sections,
        )

        self.assertEqual(expanded, {"2B.12", "2B.13", "2B.17", "2B.18"})

    def test_qrels_apply_page_section_table_and_figure_or_rule(self) -> None:
        gold = [
            {
                "benchmark_version": "test",
                "question_id": "Q1",
                "question": "Question",
                "answerable": True,
                "source_pdf_pages": [10],
                "sections": ["2B.12-2B.13"],
                "tables": ["2B-1"],
                "figures": ["2B-2"],
            }
        ]
        corpus = [
            _chunk("page", page=10, section="1A.01"),
            _chunk("section", page=20, section="2B.12"),
            _chunk("table", page=30, section="3A.01", tables=["2B-1"]),
            _chunk("figure", page=40, section="4A.01", figures=["2B-2"]),
            _chunk("irrelevant", page=50, section="5A.01"),
        ]

        qrels, report = build_chunk_qrels(gold, corpus)

        self.assertEqual(
            qrels[0]["relevant_chunk_ids"],
            ["page", "section", "table", "figure"],
        )
        reasons = {
            row["chunk_id"]: row["matching_reasons"][0]["field"]
            for row in qrels[0]["relevant_chunks"]
        }
        self.assertEqual(
            reasons,
            {
                "page": "source_pdf_pages",
                "section": "sections",
                "table": "tables",
                "figure": "figures",
            },
        )
        self.assertEqual(report["qrels_min"], 4)

    def test_binary_metrics_are_true_set_recall_mrr_and_ndcg_at_10(self) -> None:
        metrics = binary_ranking_metrics(
            ["x", "r2", "y", "r1"],
            {"r1", "r2", "r3"},
            depth=10,
        )

        expected_dcg = (1 / math.log2(3)) + (1 / math.log2(5))
        expected_idcg = 1 + (1 / math.log2(3)) + (1 / math.log2(4))
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr_at_10"], 1 / 2)
        self.assertAlmostEqual(metrics["ndcg_at_10"], expected_dcg / expected_idcg)
        self.assertEqual(metrics["first_relevant_rank"], 2)

    def test_scores_and_aggregates_canonical_rankings(self) -> None:
        qrels = [
            {
                "question_id": "Q1",
                "answerable": True,
                "primary_modality": "text",
                "split": "test",
                "relevant_chunk_ids": ["r1", "r2"],
                "relevant_chunks": [
                    {
                        "chunk_id": "r1",
                        "matching_reasons": [{"field": "sections", "value": "1A.01"}],
                    },
                    {
                        "chunk_id": "r2",
                        "matching_reasons": [{"field": "source_pdf_pages", "value": 42}],
                    },
                ],
            }
        ]
        rankings = [
            {
                "question_id": "Q1",
                "method": "bm25",
                "display_name": "BM25",
                "canonical_chunk_ranking_top10": [
                    {"canonical_chunk_id": "x"},
                    {"canonical_chunk_id": "r1"},
                ],
                "native_context_ranking": [
                    {
                        "evidence_id": "native-x",
                        "canonical_chunk_ids": ["x"],
                    },
                    {
                        "evidence_id": "native-r",
                        "canonical_chunk_ids": ["r1", "r2"],
                    },
                ],
            }
        ]

        scores, summary, sensitivity = score_canonical_rankings(rankings, qrels)

        self.assertEqual(scores[0]["hit_at_10"], 1.0)
        self.assertEqual(scores[0]["recall_at_10"], 0.5)
        self.assertEqual(summary["methods"][0]["recall_at_10"], 0.5)
        self.assertEqual(sensitivity[0]["chunk_coverage_recall_at_10"], 1.0)
        self.assertNotIn("ndcg_at_10", sensitivity[0])


def _chunk(
    chunk_id: str,
    *,
    page: int,
    section: str,
    tables: list[str] | None = None,
    figures: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "page_pdf": page,
        "page_printed": str(page),
        "section_id": section,
        "table_refs": tables or [],
        "figure_refs": figures or [],
        "text": chunk_id,
    }


if __name__ == "__main__":
    unittest.main()
