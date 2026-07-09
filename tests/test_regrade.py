from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.regrade import regrade_row, regrade_run
from gem_rags.types import GradingResult, ModelResult, QAItem


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
        "model_raw": {"answer_call_id": "answer-123", "api": "responses"},
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
        self.assertEqual(updated["config"]["grader_provider"], "heuristic")
        self.assertEqual(updated["model_raw"]["answer_call_id"], "answer-123")
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

    def test_regrade_only_missing_regrades_partial_judge_score_rows(self) -> None:
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
            partial = _row()
            partial["judge_scores"] = {"factual_accuracy": {"score": 1, "note": "partial prior score"}}
            partial["judge_error"] = None
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")
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
            stats = regrade_run(config, runs_path=runs_path, output_path=output_path, only_missing=True)
            updated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(stats["ok"])
        self.assertEqual(stats["rows_copied"], 0)
        self.assertEqual(stats["rows_regraded"], 1)
        self.assertIn("completeness", updated["judge_scores"])
        self.assertIsNone(updated["judge_error"])

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

    def test_regrade_run_reuses_llm_grader_client_across_rows(self) -> None:
        class FakeJudge:
            def __init__(self, config: ModelConfig) -> None:
                self.config = config
                self.calls = 0

            def generate(self, _prompt: str) -> ModelResult:
                self.calls += 1
                return ModelResult(
                    provider=self.config.provider,
                    model=self.config.model,
                    output='{"judge_scores": {"factual_accuracy": 5}, "judge_confidence": 0.9}',
                    raw={"fake_judge": True},
                )

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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                name="regrade",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="unit", kind="external_placeholder")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="openai", model="target-judge-model"),
                output_dir=root / "runs",
            )
            fake_judge = FakeJudge(ModelConfig(provider="openai", model="target-judge-model"))
            build_calls = []

            def fake_build_model(model_config: ModelConfig):
                build_calls.append((model_config.provider, model_config.model))
                return fake_judge

            output_path = root / "regraded.jsonl"
            with (
                patch("gem_rags.regrade.build_model", side_effect=fake_build_model),
                patch("gem_rags.grading.build_model", side_effect=AssertionError("regrade should reuse the grader")),
            ):
                stats = regrade_run(config, runs_path=runs_path, output_path=output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertTrue(stats["ok"])
        self.assertEqual(stats["rows_regraded"], 2)
        self.assertEqual(build_calls, [("openai", "target-judge-model")])
        self.assertEqual(fake_judge.calls, 2)
        self.assertTrue(all(row["judge_scores"]["factual_accuracy"]["score"] == 5 for row in rows))
        self.assertTrue(all(row["judge_error"] is None for row in rows))

    def test_regrade_uses_existing_model_raw_when_reconstructing_answer_result(self) -> None:
        captured = {}

        def fake_grade(_grader, _item, model_result, _retrieval, *, model_client=None):
            captured["raw"] = model_result.raw
            return GradingResult(grader="heuristic", scores={})

        with patch("gem_rags.regrade.grade_answer", side_effect=fake_grade):
            regrade_row(_row(), _qa(), GraderConfig(provider="heuristic", model="heuristic"), regraded_at="now")

        self.assertEqual(captured["raw"]["answer_call_id"], "answer-123")
        self.assertTrue(captured["raw"]["regraded_from_row"])


if __name__ == "__main__":
    unittest.main()
