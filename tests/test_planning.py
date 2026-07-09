from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gem_rags.cli import main
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig, write_experiment_config
from gem_rags.planning import evaluate_plan_budget, plan_experiment


def _write_qa(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"qa_id": "qa_1", "question": "Question one?", "gold_answer": {}, "references": []},
        {"qa_id": "qa_2", "question": "Question two?", "gold_answer": {}, "references": []},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestPlanning(unittest.TestCase):
    def test_plan_experiment_counts_rows_and_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "gold_qa.jsonl"
            _write_qa(qa_path)
            config = ExperimentConfig(
                name="plan-test",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected", "tool_explore"],
                models=[
                    ModelConfig(provider="dry_run", model="a"),
                    ModelConfig(provider="dry_run", model="b"),
                ],
                grader=GraderConfig(provider="openai", model="judge"),
            )
            report = plan_experiment(config)

        self.assertEqual(report["dataset"]["qa_count"], 2)
        self.assertEqual(report["dimensions"]["conditions"], 4)
        self.assertEqual(report["estimates"]["rows"], 8)
        self.assertEqual(report["estimates"]["answer_model_calls"], 12)
        self.assertEqual(report["estimates"]["judge_model_calls"], 8)
        self.assertEqual(report["estimates"]["total_model_calls"], 20)
        self.assertEqual(report["estimates"]["paid_model_calls"], 8)

    def test_cli_plan_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "gold_qa.jsonl"
            _write_qa(qa_path)
            config = ExperimentConfig(
                name="plan-cli",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
            )
            config_path = root / "config.json"
            json_output = root / "plan.json"
            csv_output = root / "plan.csv"
            write_experiment_config(config, config_path)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["plan", str(config_path), "--output", str(json_output), "--csv", str(csv_output)])
            payload = json.loads(stdout.getvalue())
            json_payload = json.loads(json_output.read_text(encoding="utf-8"))
            csv_header = csv_output.read_text(encoding="utf-8").splitlines()[0]

            self.assertEqual(code, 0)
            self.assertEqual(payload["estimates"]["rows"], 2)
            self.assertEqual(json_payload["estimates"]["rows"], 2)
            self.assertIn("retriever", csv_header)

    def test_tool_search_counts_three_answer_model_calls_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "gold_qa.jsonl"
            _write_qa(qa_path)
            config = ExperimentConfig(
                name="plan-tool-search",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["tool_search"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
            )
            report = plan_experiment(config)

        self.assertEqual(report["estimates"]["rows"], 2)
        self.assertEqual(report["estimates"]["answer_model_calls"], 6)
        self.assertEqual(report["estimates"]["total_model_calls"], 6)

    def test_dry_run_plan_reports_zero_paid_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "gold_qa.jsonl"
            _write_qa(qa_path)
            config = ExperimentConfig(
                name="plan-dry-run",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["tool_search"],
                models=[ModelConfig(provider="openai", model="target")],
                grader=GraderConfig(provider="openai", model="judge"),
                dry_run=True,
            )
            report = plan_experiment(config)

        self.assertTrue(report["dry_run"])
        self.assertEqual(report["estimates"]["total_model_calls"], 8)
        self.assertEqual(report["estimates"]["paid_model_calls"], 0)

    def test_budget_check_reports_exceeded_limits(self) -> None:
        plan = {"estimates": {"rows": 12, "total_model_calls": 30, "paid_model_calls": 8}}

        budget = evaluate_plan_budget(plan, max_rows=10, max_total_model_calls=30, max_paid_model_calls=7)

        self.assertIsNotNone(budget)
        assert budget is not None
        self.assertFalse(budget["ok"])
        self.assertEqual(
            [(item["name"], item["actual"], item["limit"]) for item in budget["exceeded"]],
            [("rows", 12, 10), ("paid_model_calls", 8, 7)],
        )


if __name__ == "__main__":
    unittest.main()
