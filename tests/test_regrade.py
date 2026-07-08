from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.regrade import regrade_row, regrade_run
from gem_rags.types import QAItem


def _qa() -> QAItem:
    return QAItem(
        qa_id="qa_regrade",
        question="What does Section 2A.04 require?",
        question_type=None,
        expected_refusal=False,
        gold_answer={"direct_answer": "Use the standard sign message."},
        references=[{"section_id": "2A.04", "content_type": "Standard", "ordinal": 13}],
    )


def _row() -> dict:
    return {
        "qa_id": "qa_regrade",
        "question": "What does Section 2A.04 require?",
        "config": {
            "experiment": "regrade",
            "retriever": "unit",
            "context_mode": "injected",
            "model_provider": "dry_run",
            "model": "dry-run",
            "grader": "old-grader",
        },
        "answer": "Use the standard sign message.",
        "model_error": None,
        "retrieval_error": None,
        "evidence": [
            {
                "evidence_id": "chunk-1",
                "kind": "chunk",
                "text": "Section 2A.04 Standard 13\nUse the standard sign message.",
                "metadata": {"section_id": "2A.04", "content_type": "Standard", "ordinal": 13},
                "score": 1.0,
            }
        ],
        "judge_scores": {"factual_accuracy": {"score": 0, "note": "old"}},
        "judge_error": "old-error",
    }


class TestRegrade(unittest.TestCase):
    def test_regrade_row_replaces_judge_fields(self) -> None:
        updated = regrade_row(_row(), _qa(), GraderConfig(provider="heuristic", model="heuristic"), regraded_at="now")

        self.assertEqual(updated["config"]["grader"], "heuristic")
        self.assertIsNone(updated["judge_error"])
        self.assertIn("completeness", updated["judge_scores"])
        self.assertEqual(updated["regrade_debug"]["regraded_at"], "now")
        self.assertEqual(updated["regrade_debug"]["evidence_count"], 1)

    def test_regrade_run_writes_output_and_only_missing_copies_clean_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text(
                json.dumps(
                    {
                        "qa_id": "qa_regrade",
                        "question": "What does Section 2A.04 require?",
                        "gold_answer": {"direct_answer": "Use the standard sign message."},
                        "references": [{"section_id": "2A.04", "content_type": "Standard", "ordinal": 13}],
                        "gold_figures": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                name="regrade",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="unit", kind="external_placeholder")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            output_path = root / "regraded.jsonl"
            stats = regrade_run(config, runs_path=runs_path, output_path=output_path)
            clean_stats = regrade_run(config, runs_path=output_path, output_path=root / "copied.jsonl", only_missing=True)
            updated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(stats["ok"])
        self.assertEqual(stats["rows_regraded"], 1)
        self.assertEqual(updated["config"]["grader"], "heuristic")
        self.assertEqual(clean_stats["rows_copied"], 1)
        self.assertEqual(clean_stats["rows_regraded"], 0)

    def test_regrade_run_rejects_in_place_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text(
                json.dumps(
                    {
                        "qa_id": "qa_regrade",
                        "question": "What does Section 2A.04 require?",
                        "gold_answer": {"direct_answer": "Use the standard sign message."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                name="regrade",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="unit", kind="external_placeholder")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )

            with self.assertRaisesRegex(ValueError, "output path must differ"):
                regrade_run(config, runs_path=runs_path, output_path=runs_path)

    def test_regrade_run_records_grader_exception_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text(
                json.dumps(
                    {
                        "qa_id": "qa_regrade",
                        "question": "What does Section 2A.04 require?",
                        "gold_answer": {"direct_answer": "Use the standard sign message."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                name="regrade",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="unit", kind="external_placeholder")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="unknown", model="bad-judge"),
                output_dir=root / "runs",
            )
            output_path = root / "regraded.jsonl"
            stats = regrade_run(config, runs_path=runs_path, output_path=output_path)
            updated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(stats["ok"])
        self.assertEqual(stats["judge_errors"], 1)
        self.assertIn("regrade_failed: ValueError", updated["judge_error"])
        self.assertEqual(updated["regrade_debug"]["grader_model"], "bad-judge")


if __name__ == "__main__":
    unittest.main()
