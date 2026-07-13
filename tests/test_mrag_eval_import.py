from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.analysis import summarize_rows
from gems_rag.mrag_eval_import import import_mrag_eval


class TestMragEvalImport(unittest.TestCase):
    def test_import_mrag_eval_joins_runs_scores_and_enriches_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            eval_dir = mrag_dir / "eval"
            figures_dir = mrag_dir / "figures"
            pages_dir = mrag_dir / "page_images"
            cache.mkdir(parents=True)
            eval_dir.mkdir()
            figures_dir.mkdir()
            pages_dir.mkdir()
            (figures_dir / "figure_2A-1_p0001.png").write_bytes(b"fake")
            (pages_dir / "page_0001.png").write_bytes(b"fake")
            (cache / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "MUTCD11e_2A01_Standard_01",
                        "section_id": "2A.01",
                        "section_title": "Function and Purpose",
                        "content_type": "Standard",
                        "ordinal": 1,
                        "page_printed": "1",
                        "text": "A standard sign shall be used.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (cache / "figures.jsonl").write_text(
                json.dumps(
                    {
                        "figure_id": "Figure 2A-1",
                        "kind": "Figure",
                        "canonical_id": "2A-1",
                        "page_pdf": 1,
                        "caption": "Figure 2A-1",
                        "image_path": "/content/drive/MyDrive/MRAG/figures/figure_2A-1_p0001.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (eval_dir / "gold_qa.jsonl").write_text(
                json.dumps(
                    {
                        "qa_id": "qa_0001",
                        "question": "What sign is required?",
                        "gold_answer": {"direct_answer": "A standard sign shall be used."},
                        "references": [{"section_id": "2A.01", "content_type": "Standard", "ordinal": 1}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {"prompt_style": "fewshot", "vlm_alias": "flash", "vlm_model_id": "qwen3-vl-flash"}
            (eval_dir / "runs.jsonl").write_text(
                json.dumps(
                    {
                        "qa_id": "qa_0001",
                        "question": "What sign is required?",
                        "config": config,
                        "answer": "A standard sign shall be used.",
                        "chunks_used": [{"section_id": "2A.01", "content_type": "Standard", "ordinal": 1, "score": 9.0}],
                        "figures_used": [{"figure_id": "Figure 2A-1", "source": "visual"}],
                        "pages_used": ["1"],
                        "debug": {"query": "What sign is required?"},
                        "latency_s": 1.23456,
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (eval_dir / "scored.jsonl").write_text(
                json.dumps(
                    {
                        "qa_id": "qa_0001",
                        "question": "What sign is required?",
                        "config": config,
                        "judge_scores": {"factual_accuracy": {"score": 5, "note": "ok"}},
                        "judge_confidence": 0.9,
                        "judge_explanation": "grounded",
                        "figure_metrics": {"figure_recall": 1.0},
                        "system_confidence_breakdown": {"system_confidence": 8.0},
                        "judge_error": None,
                        "rag_answer_length_chars": 31,
                        "rag_n_figures": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "runs" / "mrag-prior" / "runs.jsonl"
            stats = import_mrag_eval(mrag_dir, output, overwrite=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            summary = summarize_rows(rows)

        self.assertTrue(stats["ok"])
        self.assertEqual(stats["rows_written"], 1)
        self.assertEqual(stats["chunk_evidence"], 1)
        self.assertEqual(stats["figure_evidence"], 1)
        self.assertEqual(stats["page_evidence"], 1)
        self.assertEqual(rows[0]["config"]["model_provider"], "qwen")
        self.assertEqual(rows[0]["config"]["model"], "qwen3-vl-flash")
        self.assertTrue(rows[0]["model_raw"]["imported_mrag_eval"])
        self.assertEqual(rows[0]["model_raw"]["source_model"], "qwen3-vl-flash")
        self.assertEqual(rows[0]["evidence"][0]["metadata"]["chunk_id"], "MUTCD11e_2A01_Standard_01")
        self.assertTrue(rows[0]["evidence"][1]["metadata"]["image_path"].endswith("figure_2A-1_p0001.png"))
        self.assertEqual(rows[0]["grader_raw"]["diagnostics"]["gold_reference_recall"], 1.0)
        self.assertEqual(summary[0]["mean_factual_accuracy"], 5.0)


if __name__ == "__main__":
    unittest.main()
