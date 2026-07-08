from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.analysis import analyze_run, compare_conditions, metric_value, parse_filter, validate_run
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig


def _row(qa_id: str, context_mode: str, retriever: str, factual: int, evidence_count: int) -> dict:
    return {
        "qa_id": qa_id,
        "config": {
            "experiment": "unit",
            "retriever": retriever,
            "context_mode": context_mode,
            "model_provider": "dry_run",
            "model": "dry-run",
            "grader": "heuristic",
        },
        "judge_scores": {
            "factual_accuracy": {"score": factual, "note": ""},
            "completeness": {"score": factual + 1, "note": ""},
        },
        "grader_raw": {"diagnostics": {"gold_section_recall": factual / 5}},
        "evidence": [{"id": idx} for idx in range(evidence_count)],
        "latency_s": 0.1,
    }


class TestAnalysis(unittest.TestCase):
    def test_compare_conditions_matches_unchanged_fields(self) -> None:
        rows = [
            _row("qa1", "injected", "bm25", 2, 6),
            _row("qa1", "tool_explore", "bm25", 4, 3),
            _row("qa2", "injected", "bm25", 3, 5),
            _row("qa2", "tool_explore", "bm25", 3, 4),
            _row("qa1", "tool_explore", "hash_vector", 5, 2),
        ]
        result = compare_conditions(
            rows,
            baseline_filter={"context_mode": "injected"},
            candidate_filter={"context_mode": "tool_explore"},
            metrics=["factual_accuracy", "evidence_count"],
        )

        self.assertEqual(result["matched_pairs"], 2)
        factual = next(metric for metric in result["metrics"] if metric["metric"] == "factual_accuracy")
        self.assertEqual(factual["baseline_mean"], 2.5)
        self.assertEqual(factual["candidate_mean"], 3.5)
        self.assertEqual(factual["wins"], 1)
        self.assertEqual(factual["ties"], 1)
        evidence = next(metric for metric in result["metrics"] if metric["metric"] == "evidence_count")
        self.assertEqual(evidence["mean_delta"], -2.0)

    def test_metric_value_supports_diagnostics(self) -> None:
        self.assertEqual(metric_value(_row("qa1", "injected", "bm25", 4, 6), "gold_section_recall"), 0.8)

    def test_analyze_run_writes_summary_and_axis_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_path = root / "runs.jsonl"
            qa_path = root / "qa.jsonl"
            qa_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "qa_id": "qa1",
                                "question": "Q1?",
                                "question_type": "figure lookup",
                                "expected_refusal": False,
                                "gold_answer": {},
                                "references": [{"section_id": "2A.01", "content_type": "Standard"}],
                                "gold_figures": ["Figure 2A-1"],
                            }
                        ),
                        json.dumps(
                            {
                                "qa_id": "qa2",
                                "question": "Q2?",
                                "question_type": "text lookup",
                                "expected_refusal": True,
                                "gold_answer": {},
                                "references": [],
                                "gold_figures": [],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                _row("qa1", "injected", "bm25", 2, 6),
                _row("qa1", "tool_explore", "bm25", 4, 3),
                _row("qa2", "injected", "bm25", 3, 5),
                _row("qa2", "tool_explore", "bm25", 3, 4),
                _row("qa1", "tool_explore", "hash_vector", 5, 2),
            ]
            runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output_dir = root / "analysis"

            report = analyze_run(
                runs_path,
                output_dir=output_dir,
                qa_path=qa_path,
                axis="context_mode",
                baseline="injected",
                metrics=["factual_accuracy"],
            )

            comparison = report["comparisons"][0]
            self.assertEqual(report["filtered_rows"], 5)
            self.assertEqual(report["candidate_values"], ["tool_explore"])
            self.assertEqual(comparison["matched_pairs"], 2)
            self.assertTrue((output_dir / "analysis.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue(Path(comparison["comparison_json"]).exists())
            self.assertTrue(Path(comparison["metrics_csv"]).exists())
            self.assertTrue(Path(comparison["pairs_csv"]).exists())
            self.assertTrue(Path(report["strata_summary_csv"]).exists())
            self.assertTrue(Path(report["strata_comparisons_csv"]).exists())
            with Path(report["strata_comparisons_csv"]).open(encoding="utf-8") as handle:
                strata_rows = list(csv.DictReader(handle))
            figure_stratum = [
                row
                for row in strata_rows
                if row["facet"] == "has_gold_figures" and row["value"] == "true" and row["metric"] == "factual_accuracy"
            ]
            self.assertEqual(figure_stratum[0]["matched_pairs"], "1")
            self.assertEqual(figure_stratum[0]["mean_delta"], "2.0")

    def test_parse_filter_requires_equals(self) -> None:
        self.assertEqual(parse_filter(["context_mode=injected"]), {"context_mode": "injected"})
        with self.assertRaises(ValueError):
            parse_filter(["context_mode"])

    def test_validate_run_reports_missing_duplicate_unexpected_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected", "tool_explore"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = root / "runs.jsonl"
            rows = [
                _row("qa1", "injected", "bm25", 2, 1),
                _row("qa1", "injected", "bm25", 2, 1),
                _row("qa1", "injected", "unexpected", 2, 1),
            ]
            rows[0]["retrieval_error"] = "boom"
            runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = validate_run(config, runs_path)

        self.assertFalse(report["ok"])
        self.assertEqual(report["expected_rows"], 2)
        self.assertEqual(report["actual_rows"], 3)
        self.assertEqual(report["missing_rows"], 1)
        self.assertEqual(report["unexpected_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["retrieval_errors"], 1)


if __name__ == "__main__":
    unittest.main()
