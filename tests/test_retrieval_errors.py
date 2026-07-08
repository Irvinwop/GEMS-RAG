from __future__ import annotations

import json
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.analysis import metric_value, summarize_rows
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.runner import run_experiment
from gem_rags.types import RetrievalResult


def _fixture_mrag(root: Path) -> tuple[Path, Path]:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"placeholder")
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa_fail",
                "question": "What does the adapter return?",
                "gold_answer": {},
                "references": [],
                "gold_figures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return mrag_dir, qa_path


class TestRetrievalErrors(unittest.TestCase):
    def test_retriever_build_failure_is_recorded_as_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="build-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bad_retriever", kind="unknown_external")],
                context_modes=["injected", "tool_explore"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["retrieval_error"].startswith("retriever_build_failed: ValueError") for row in rows))
        self.assertTrue(all(row["config"]["retriever"] == "bad_retriever" for row in rows))
        self.assertTrue(all(row["retrieval_debug"]["retriever_build_error"] == row["retrieval_error"] for row in rows))

    def test_retriever_runtime_exception_is_recorded_and_run_continues(self) -> None:
        class RaisingRetriever:
            def retrieve(self, _item):
                raise RuntimeError("index missing")

        class EmptyRetriever:
            def retrieve(self, item):
                return RetrievalResult(adapter="ok_retriever", query=item.question, evidence=[], debug={"ok": True})

        def fake_build(config, _mrag_dir):
            return RaisingRetriever() if config.name == "bad_retriever" else EmptyRetriever()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="runtime-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(name="bad_retriever", kind="external_command"),
                    RetrieverConfig(name="ok_retriever", kind="external_command"),
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            with patch("gem_rags.runner.build_retriever", side_effect=fake_build):
                runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        by_retriever = {row["config"]["retriever"]: row for row in rows}
        self.assertEqual(by_retriever["bad_retriever"]["retrieval_error"], "retriever_failed: RuntimeError: index missing")
        self.assertIsNone(by_retriever["ok_retriever"]["retrieval_error"])

    def test_retry_errors_prunes_failed_rows_and_reruns_only_those_keys(self) -> None:
        class RaisingRetriever:
            def retrieve(self, _item):
                raise RuntimeError("index missing")

        class EmptyRetriever:
            def retrieve(self, item):
                return RetrievalResult(adapter="flaky_retriever", query=item.question, evidence=[], debug={"fixed": True})

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="retry-errors",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="flaky_retriever", kind="external_command")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            with patch("gem_rags.runner.build_retriever", return_value=RaisingRetriever()):
                runs_path = run_experiment(config, overwrite=True)
            failed_row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(failed_row["retrieval_error"], "retriever_failed: RuntimeError: index missing")

            with patch("gem_rags.runner.build_retriever", return_value=EmptyRetriever()):
                rerun_path = run_experiment(config, retry_errors=True)
            rows = [json.loads(line) for line in rerun_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest = json.loads((root / "runs" / "retry-errors" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["retrieval_error"])
        self.assertTrue(rows[0]["retrieval_debug"]["fixed"])
        self.assertEqual(manifest["summary"]["mode"], "retry_errors")
        self.assertEqual(manifest["summary"]["rows_pruned_for_retry"], 1)
        self.assertEqual(manifest["summary"]["rows_kept_for_retry"], 0)
        self.assertEqual(manifest["summary"]["rows_written"], 1)

    def test_model_and_grader_build_failures_are_recorded_as_row_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="model-grader-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="unknown_provider", model="bad-model")],
                grader=GraderConfig(provider="unknown_grader", model="bad-judge"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["model_error"], "model_build_failed: ValueError: unknown model provider: unknown_provider")
        self.assertEqual(row["judge_error"], "grade_failed: ValueError: unknown grader provider: unknown_grader")

    def test_external_command_failure_is_recorded_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="retrieval-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="failing_external",
                        kind="external_command",
                        options={
                            "command": [
                                sys.executable,
                                "-c",
                                "import sys; sys.stderr.write('adapter exploded'); sys.exit(7)",
                            ]
                        },
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["retrieval_error"], "external command exited with return code 7")
        self.assertEqual(rows[0]["retrieval_debug"]["returncode"], 7)
        self.assertIn("adapter exploded", rows[0]["retrieval_debug"]["stderr"])
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["retrieval_errors"], 1)
        self.assertEqual(metric_value(rows[0], "retrieval_failed"), 1.0)

    def test_external_command_context_preserves_visual_metadata(self) -> None:
        payload = {
            "contexts": [
                {
                    "name": "visrag:page:0001",
                    "kind": "page",
                    "text": "MUTCD document page image 1. Sections: 2A.01.",
                    "score": 0.87,
                    "image_path": "/tmp/page_0001.png",
                    "metadata": {"page_pdf": 1, "section_ids": ["2A.01"]},
                }
            ]
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        command = [sys.executable, "-c", f"import base64; print(base64.b64decode('{encoded}').decode())"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="visual-context",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="visual_external",
                        kind="external_command",
                        options={"command": command},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        evidence = row["evidence"][0]
        self.assertEqual(evidence["kind"], "page")
        self.assertEqual(evidence["metadata"]["image_path"], "/tmp/page_0001.png")
        self.assertEqual(evidence["metadata"]["page_pdf"], 1)
        self.assertEqual(evidence["metadata"]["section_ids"], ["2A.01"])


if __name__ == "__main__":
    unittest.main()
