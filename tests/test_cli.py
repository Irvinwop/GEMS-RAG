from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gem_rags.cli import main
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig, load_experiment_config, write_experiment_config
from gem_rags.qa_sets import write_qa_split


def _write_fixture_config(root: Path) -> Path:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa_1",
                "question": "What standard applies to signs?",
                "gold_answer": {"direct_answer": "Use the standard sign."},
                "references": [],
                "gold_figures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"placeholder")
    config = ExperimentConfig(
        name="sweep-mini",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
        retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
        context_modes=["injected", "tool_explore"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
        grader=GraderConfig(provider="heuristic", model="heuristic"),
        output_dir=root / "runs",
        max_evidence_chars=500,
    )
    config_path = root / "base.json"
    write_experiment_config(config, config_path)
    return config_path


class TestCli(unittest.TestCase):
    def test_sweep_writes_run_summary_and_context_compare(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["sweep", str(config_path), "--overwrite"])
            payload = json.loads(stdout.getvalue())
            run_dir = root / "runs" / "sweep-mini"

            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["validation_ok"])
            self.assertEqual(payload["rows"], 2)
            self.assertEqual(payload["matched_context_pairs"], 1)
            self.assertTrue((run_dir / "materialized_config.json").exists())
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "runs.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "validation.json").exists())
            self.assertTrue((run_dir / "context-compare.json").exists())
            self.assertTrue((run_dir / "context-pairs.csv").exists())

    def test_analyze_writes_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["sweep", str(config_path), "--overwrite"]), 0)
            run_dir = root / "runs" / "sweep-mini"
            output_dir = run_dir / "analysis"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "analyze",
                        str(run_dir / "runs.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--qa-path",
                        str(root / "MRAG" / "eval" / "gold_qa.jsonl"),
                        "--axis",
                        "context_mode",
                        "--baseline",
                        "injected",
                    ]
                )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(payload["candidate_values"], ["tool_explore"])
            self.assertEqual(payload["comparisons"][0]["matched_pairs"], 1)
            self.assertTrue((output_dir / "analysis.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "strata-summary.csv").exists())
            self.assertTrue((output_dir / "strata-comparisons.csv").exists())

    def test_sweep_writes_tool_search_context_compare(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            base = load_experiment_config(config_path)
            config = ExperimentConfig(
                name="tool-search-compare",
                dataset=base.dataset,
                retrievers=base.retrievers,
                context_modes=["injected", "tool_search"],
                models=base.models,
                grader=base.grader,
                output_dir=base.output_dir,
                max_evidence_chars=base.max_evidence_chars,
            )
            write_experiment_config(config, config_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["sweep", str(config_path), "--overwrite"])
            payload = json.loads(stdout.getvalue())
            run_dir = root / "runs" / "tool-search-compare"

            self.assertEqual(code, 0)
            self.assertEqual(payload["rows"], 2)
            self.assertIn("tool_search", payload["context_comparisons"])
            self.assertEqual(payload["context_comparisons"]["tool_search"]["matched_pairs"], 1)
            self.assertTrue((run_dir / "context-tool-search-compare.json").exists())
            self.assertTrue((run_dir / "context-tool-search-pairs.csv").exists())

    def test_materialize_accepts_qa_ids_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            split_path = root / "split.json"
            write_qa_split(split_path, {"qa_ids": ["qa_1"]})
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["materialize", str(config_path), "--qa-ids-file", str(split_path)])
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(payload["dataset"]["qa_ids"], ["qa_1"])

    def test_materialize_accepts_models_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            models_path = root / "models.txt"
            models_path.write_text(
                """
                openai:gpt-4.1-mini,max_tokens=300
                local_openai:llama-3.1-8b,base_url=http://localhost:8000/v1,max_tokens=700
                """,
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["materialize", str(config_path), "--models-file", str(models_path)])
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(
                [(item["provider"], item["model"]) for item in payload["models"]],
                [("openai", "gpt-4.1-mini"), ("local_openai", "llama-3.1-8b")],
            )
            self.assertEqual(payload["models"][1]["options"]["base_url"], "http://localhost:8000/v1")

    def test_run_retry_errors_replaces_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            base = load_experiment_config(config_path)
            config = ExperimentConfig(
                name="retry-cli",
                dataset=base.dataset,
                retrievers=[
                    RetrieverConfig(
                        name="external_slot",
                        kind="external_command",
                        options={"command": [sys.executable, "-c", "import sys; sys.exit(9)"]},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            write_experiment_config(config, config_path)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(config_path), "--overwrite"]), 0)

            fixed = ExperimentConfig(
                name=config.name,
                dataset=config.dataset,
                retrievers=[
                    RetrieverConfig(
                        name="external_slot",
                        kind="external_command",
                        options={"command": [sys.executable, "-c", "print('{{\"contexts\": []}}')"]},
                    )
                ],
                context_modes=config.context_modes,
                models=config.models,
                grader=config.grader,
                output_dir=config.output_dir,
            )
            write_experiment_config(fixed, config_path)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(config_path), "--retry-errors"]), 0)

            runs_path = root / "runs" / "retry-cli" / "runs.jsonl"
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["retrieval_error"])


if __name__ == "__main__":
    unittest.main()
