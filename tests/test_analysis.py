from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.analysis import RUBRIC_KEYS, analyze_run, compare_conditions, leaderboard_rows, metric_value, parse_filter, summarize_rows, validate_run
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig


def _row(
    qa_id: str,
    context_mode: str,
    retriever: str,
    factual: int,
    evidence_count: int,
    context_debug: dict | None = None,
    model_usage: dict | None = None,
    judge_usage: dict | None = None,
) -> dict:
    row = {
        "qa_id": qa_id,
        "config": {
            "experiment": "unit",
            "retriever": retriever,
            "context_mode": context_mode,
            "model_provider": "dry_run",
            "model": "dry-run",
            "grader_provider": "heuristic",
            "grader": "heuristic",
        },
        "model_raw": {"usage": model_usage} if model_usage else {},
        "judge_scores": {
            "factual_accuracy": {"score": factual, "note": ""},
            "completeness": {"score": factual + 1, "note": ""},
        },
        "grader_raw": {
            "diagnostics": {"gold_section_recall": factual / 5},
            **({"model_raw": {"usage": judge_usage}} if judge_usage else {}),
        },
        "evidence": [{"id": idx} for idx in range(evidence_count)],
        "latency_s": 0.1,
    }
    if context_debug is not None:
        row["retrieval_debug"] = {"context_debug": context_debug}
    return row


def _complete_judge_scores(score: int = 3) -> dict[str, dict[str, object]]:
    return {key: {"score": score, "note": ""} for key in RUBRIC_KEYS}


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

    def test_tool_operational_metrics_are_available_for_comparisons_and_summaries(self) -> None:
        row = _row(
            "qa1",
            "tool_search",
            "bm25",
            4,
            2,
            context_debug={
                "selected_ids": ["hit-a", "hit-b", "missing"],
                "opened_ids": ["hit-a", "hit-b"],
                "selection_parse_failed": True,
                "search_queries": [{"query": "Section 2A.04", "top_k": 2}, {"query": "warning signs", "top_k": 3}],
                "search_results": [
                    {"result_ids": ["hit-a", "hit-b"]},
                    {"result_ids": ["hit-b", "hit-c"]},
                ],
                "search_errors": ["adapter timeout"],
                "search_parse_failed": False,
            },
        )
        injected = _row("qa1", "injected", "bm25", 4, 6)

        self.assertEqual(metric_value(row, "tool_selected_count"), 3.0)
        self.assertEqual(metric_value(row, "tool_opened_count"), 2.0)
        self.assertEqual(metric_value(row, "tool_selection_parse_failed"), 1.0)
        self.assertEqual(metric_value(row, "tool_search_query_count"), 2.0)
        self.assertEqual(metric_value(row, "tool_search_result_count"), 3.0)
        self.assertEqual(metric_value(row, "tool_search_error_count"), 1.0)
        self.assertEqual(metric_value(row, "tool_search_parse_failed"), 0.0)
        self.assertEqual(metric_value(injected, "tool_opened_count"), 0.0)

        comparison = compare_conditions(
            [injected, row],
            baseline_filter={"context_mode": "injected"},
            candidate_filter={"context_mode": "tool_search"},
            metrics=["tool_opened_count", "tool_search_query_count"],
        )
        opened = next(metric for metric in comparison["metrics"] if metric["metric"] == "tool_opened_count")
        queries = next(metric for metric in comparison["metrics"] if metric["metric"] == "tool_search_query_count")
        self.assertEqual(opened["mean_delta"], 2.0)
        self.assertEqual(queries["mean_delta"], 2.0)

        summary = summarize_rows([row])[0]
        self.assertEqual(summary["mean_tool_selected"], 3.0)
        self.assertEqual(summary["mean_tool_opened"], 2.0)
        self.assertEqual(summary["tool_selection_parse_failures"], 1)
        self.assertEqual(summary["mean_tool_search_queries"], 2.0)
        self.assertEqual(summary["mean_tool_search_results"], 3.0)
        self.assertEqual(summary["mean_tool_search_errors"], 1.0)
        self.assertEqual(summary["tool_search_parse_failures"], 0)

    def test_token_usage_metrics_are_available_for_answer_and_judge_calls(self) -> None:
        row = _row(
            "qa1",
            "injected",
            "bm25",
            4,
            2,
            model_usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            judge_usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
        baseline = _row(
            "qa1",
            "tool_explore",
            "bm25",
            4,
            2,
            model_usage={"input_tokens": 70, "output_tokens": 10, "total_tokens": 80},
            judge_usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        )

        self.assertEqual(metric_value(row, "answer_input_tokens"), 100.0)
        self.assertEqual(metric_value(row, "answer_output_tokens"), 30.0)
        self.assertEqual(metric_value(row, "answer_total_tokens"), 130.0)
        self.assertEqual(metric_value(row, "judge_input_tokens"), 80.0)
        self.assertEqual(metric_value(row, "judge_output_tokens"), 20.0)
        self.assertEqual(metric_value(row, "judge_total_tokens"), 100.0)
        self.assertEqual(metric_value(row, "total_tokens"), 230.0)

        comparison = compare_conditions(
            [baseline, row],
            baseline_filter={"context_mode": "tool_explore"},
            candidate_filter={"context_mode": "injected"},
            metrics=["answer_total_tokens", "judge_total_tokens", "total_tokens"],
        )
        total = next(metric for metric in comparison["metrics"] if metric["metric"] == "total_tokens")
        self.assertEqual(total["mean_delta"], 100.0)

        summary = summarize_rows([row])[0]
        self.assertEqual(summary["mean_answer_input_tokens"], 100.0)
        self.assertEqual(summary["mean_answer_total_tokens"], 130.0)
        self.assertEqual(summary["mean_judge_total_tokens"], 100.0)
        self.assertEqual(summary["mean_total_tokens"], 230.0)

    def test_model_pricing_metrics_are_available_for_answer_and_judge_calls(self) -> None:
        row = _row(
            "qa1",
            "injected",
            "bm25",
            4,
            2,
            model_usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            judge_usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
        row["config"]["grader_provider"] = "openai"
        row["config"]["grader"] = "judge"
        priced = summarize_rows(
            [row],
            model_pricing={
                "dry_run:dry-run": {"input_per_1m": 1.0, "output_per_1m": 2.0},
                "openai:judge": {"input_per_1m": 10.0, "output_per_1m": 20.0},
            },
        )[0]

        self.assertEqual(priced["mean_answer_cost_usd"], 0.00016)
        self.assertEqual(priced["mean_judge_cost_usd"], 0.0012)
        self.assertEqual(priced["mean_total_cost_usd"], 0.00136)
        self.assertEqual(priced["total_cost_usd"], 0.00136)

    def test_leaderboard_ranks_by_score_errors_and_cost(self) -> None:
        good = _row(
            "qa1",
            "injected",
            "bm25",
            4,
            2,
            model_usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        )
        good["judge_scores"] = _complete_judge_scores(4)
        errored = _row(
            "qa1",
            "injected",
            "dense",
            4,
            2,
            model_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        errored["judge_scores"] = _complete_judge_scores(4)
        errored["retrieval_error"] = "adapter failed"
        weak = _row(
            "qa1",
            "injected",
            "oracle",
            2,
            2,
            model_usage={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        )
        weak["judge_scores"] = _complete_judge_scores(2)

        summary = summarize_rows(
            [errored, weak, good],
            model_pricing={"dry_run:dry-run": {"input_per_1m": 1.0, "output_per_1m": 2.0}},
        )
        leaderboard = leaderboard_rows(summary)

        self.assertEqual([row["retriever"] for row in leaderboard], ["bm25", "dense", "oracle"])
        self.assertEqual([row["rank"] for row in leaderboard], [1, 2, 3])
        self.assertEqual(leaderboard[0]["mean_judge_score"], 4.0)
        self.assertEqual(leaderboard[1]["row_error_rate"], 1.0)
        self.assertEqual(leaderboard[2]["mean_judge_score"], 2.0)

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
            self.assertTrue((output_dir / "leaderboard.json").exists())
            self.assertTrue((output_dir / "leaderboard.csv").exists())
            self.assertTrue(Path(report["leaderboard_csv"]).exists())
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

    def test_validate_run_reports_partial_judge_scores_unless_errors_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = root / "runs.jsonl"
            partial = _row("qa1", "injected", "bm25", 2, 1)
            partial["judge_scores"] = {"factual_accuracy": {"score": 2, "note": "partial"}}
            runs_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")

            report = validate_run(config, runs_path)
            allowed = validate_run(config, runs_path, allow_errors=True)

        self.assertFalse(report["ok"])
        self.assertEqual(report["incomplete_judge_scores"], 1)
        self.assertIn("incomplete_judge_scores=1", report["problems"][-1])
        self.assertEqual(report["incomplete_judge_scores_sample"][0]["missing_score_keys"][0], "category_correctness")
        self.assertTrue(allowed["ok"])
        self.assertEqual(allowed["incomplete_judge_scores"], 1)

    def test_validate_run_reports_grader_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="new-judge"),
                output_dir=root / "runs",
            )
            row = _row("qa1", "injected", "bm25", 2, 1)
            row["config"]["grader"] = "old-judge"
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = validate_run(config, runs_path)
            allowed = validate_run(config, runs_path, allow_errors=True)

        self.assertFalse(report["ok"])
        self.assertEqual(report["grader_mismatches"], 1)
        self.assertIn("grader_mismatches=1", report["problems"][-1])
        self.assertEqual(report["grader_mismatches_sample"][0]["expected_grader"], "new-judge")
        self.assertEqual(report["grader_mismatches_sample"][0]["actual_grader"], "old-judge")
        self.assertTrue(allowed["ok"])
        self.assertEqual(allowed["grader_mismatches"], 1)

    def test_validate_run_reports_and_enforces_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            row = _row(
                "qa1",
                "injected",
                "bm25",
                2,
                1,
                model_usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
                judge_usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
            )
            row["judge_scores"] = _complete_judge_scores()
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            within = validate_run(config, runs_path, max_total_tokens=300)
            exceeded = validate_run(config, runs_path, max_total_tokens=200)

        self.assertTrue(within["ok"])
        self.assertTrue(within["budget_ok"])
        self.assertEqual(within["token_usage"]["answer_total_tokens"], 130)
        self.assertEqual(within["token_usage"]["judge_total_tokens"], 100)
        self.assertEqual(within["token_usage"]["total_tokens"], 230)
        self.assertEqual(within["token_usage"]["rows_with_any_token_usage"], 1)
        self.assertFalse(exceeded["ok"])
        self.assertFalse(exceeded["budget_ok"])
        self.assertEqual(exceeded["budget_checks"][0]["name"], "total_tokens")
        self.assertEqual(exceeded["budget_checks"][0]["actual"], 230)
        self.assertEqual(exceeded["budget_checks"][0]["limit"], 200)
        self.assertIn("budget exceeded: total_tokens=230 limit=200", exceeded["problems"])

    def test_validate_run_enforces_cost_budget_and_requires_complete_cost_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate-cost",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="openai", model="answer-model")],
                grader=GraderConfig(provider="openai", model="judge-model"),
                output_dir=root / "runs",
            )
            row = _row(
                "qa1",
                "injected",
                "bm25",
                2,
                1,
                model_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                judge_usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
            )
            row["config"].update(
                {
                    "model_provider": "openai",
                    "model": "answer-model",
                    "grader_provider": "openai",
                    "grader": "judge-model",
                }
            )
            row["judge_scores"] = _complete_judge_scores()
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            pricing = {
                "openai:answer-model": {"input_per_1m": 1.0, "output_per_1m": 2.0},
                "openai:judge-model": {"input_per_1m": 10.0, "output_per_1m": 20.0},
            }

            within = validate_run(
                config,
                runs_path,
                max_total_cost_usd=0.002,
                model_pricing=pricing,
                pricing_source="catalog.json",
            )
            exceeded = validate_run(config, runs_path, max_total_cost_usd=0.001, model_pricing=pricing)
            incomplete = validate_run(
                config,
                runs_path,
                max_total_cost_usd=1.0,
                model_pricing={"openai:answer-model": pricing["openai:answer-model"]},
            )

        self.assertTrue(within["ok"])
        self.assertEqual(within["pricing_source"], "catalog.json")
        self.assertTrue(within["cost"]["coverage_ok"])
        self.assertEqual(within["cost"]["total_cost_usd"], 0.0014)
        self.assertEqual(within["budget_checks"][0]["name"], "total_cost_usd")
        self.assertFalse(exceeded["ok"])
        self.assertEqual(exceeded["budget_checks"][0]["actual"], 0.0014)
        self.assertIn("budget exceeded: total_cost_usd=0.0014 limit=0.001", exceeded["problems"])
        self.assertFalse(incomplete["ok"])
        self.assertFalse(incomplete["cost"]["coverage_ok"])
        self.assertIsNone(incomplete["cost"]["total_cost_usd"])
        self.assertEqual(incomplete["cost"]["known_total_cost_usd"], 0.0002)
        self.assertEqual(incomplete["budget_checks"][0]["reason"], "incomplete_cost_coverage")
        self.assertIn("missing pricing or usage for 1 paid model calls", incomplete["problems"][-1])

    def test_validate_run_rejects_partial_multi_call_usage_for_cost_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate-partial-cost",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["tool_explore"],
                models=[ModelConfig(provider="openai", model="answer-model")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            row = _row(
                "qa1",
                "tool_explore",
                "bm25",
                2,
                1,
                model_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
            row["config"].update({"model_provider": "openai", "model": "answer-model"})
            row["model_raw"]["usage_coverage"] = {"expected_calls": 2, "observed_calls": 1, "complete": False}
            row["judge_scores"] = _complete_judge_scores()
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = validate_run(
                config,
                runs_path,
                max_total_cost_usd=1.0,
                model_pricing={"openai:answer-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}},
            )

        self.assertFalse(report["ok"])
        self.assertFalse(report["cost"]["coverage_ok"])
        self.assertEqual(report["cost"]["missing_cost_components"], 1)
        self.assertEqual(report["budget_checks"][0]["reason"], "incomplete_cost_coverage")

    def test_validate_run_accepts_explicit_zero_cost_without_usage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate-local-cost",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["tool_search"],
                models=[ModelConfig(provider="local_openai", model="local-model")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            row = _row("qa1", "tool_search", "bm25", 2, 1)
            row["config"].update({"model_provider": "local_openai", "model": "local-model"})
            row["judge_scores"] = _complete_judge_scores()
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = validate_run(
                config,
                runs_path,
                max_total_cost_usd=0,
                model_pricing={"local_openai:local-model": {"input_per_1m": 0.0, "output_per_1m": 0.0}},
            )

        self.assertTrue(report["ok"])
        self.assertTrue(report["cost"]["coverage_ok"])
        self.assertEqual(report["cost"]["total_cost_usd"], 0.0)
        self.assertEqual(report["cost"]["expected_answer_calls"], 3)
        self.assertEqual(report["cost"]["priced_answer_calls"], 3)

    def test_validate_run_allows_nonheuristic_grader_dry_run_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            qa_path.write_text('{"qa_id":"qa1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            config = ExperimentConfig(
                name="validate",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="openai", model="answer-model")],
                grader=GraderConfig(provider="openai", model="judge-model"),
                output_dir=root / "runs",
                dry_run=True,
            )
            row = _row("qa1", "injected", "bm25", 2, 1)
            row["config"]["model_provider"] = "openai"
            row["config"]["model"] = "answer-model"
            row["config"]["grader"] = "judge-model"
            row["judge_scores"] = {}
            row["grader_raw"] = {"dry_run": True}
            runs_path = root / "runs.jsonl"
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = validate_run(config, runs_path)

        self.assertTrue(report["ok"])
        self.assertEqual(report["incomplete_judge_scores"], 0)


if __name__ == "__main__":
    unittest.main()
