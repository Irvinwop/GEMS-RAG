from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from gems_rag.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gems_rag.models import DryRunModel
from gems_rag.runner import run_experiment
from gems_rag.types import ModelResult


def _fixture(root: Path, *, qa_count: int = 1) -> tuple[Path, Path]:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"placeholder")
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    rows = [
        {
            "qa_id": f"qa_dry_{index}",
            "question": f"What is dry run answer {index}?",
            "gold_answer": {"direct_answer": "No model should be called."},
            "references": [],
        }
        for index in range(qa_count)
    ]
    qa_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return mrag_dir, qa_path


class FakeJudgeModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.calls = 0

    def generate(self, _prompt: str) -> ModelResult:
        self.calls += 1
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output='{"judge_scores": {"factual_accuracy": {"score": 5, "note": "ok"}}, "judge_confidence": 0.9}',
            raw={"fake_judge": True},
        )


class TestRunnerDryRun(unittest.TestCase):
    def test_runner_rejects_incompatible_rag_context_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            config = ExperimentConfig(
                name="invalid-context",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[
                    RetrieverConfig(
                        name="fixed-question",
                        kind="bm25",
                        context_modes=("injected", "tool_explore"),
                        interaction="fixed_question",
                    )
                ],
                context_modes=["tool_native"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                output_dir=root / "runs",
            )

            with self.assertRaisesRegex(ValueError, "fixed-question: tool_native"):
                run_experiment(config, overwrite=True)

            self.assertFalse((config.output_dir / config.name).exists())

    def test_run_lock_blocks_a_second_writer_and_recovers_when_stale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            config = ExperimentConfig(
                name="locked-run",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run-answer")],
                output_dir=root / "runs",
            )
            run_dir = config.output_dir / config.name
            run_dir.mkdir(parents=True)
            lock_path = run_dir / ".run.lock"
            lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "already active"):
                run_experiment(config, overwrite=True)

            lock_path.write_text("{incomplete", encoding="utf-8")
            stale_time = lock_path.stat().st_mtime - 20
            os.utime(lock_path, (stale_time, stale_time))
            runs_path = run_experiment(config, overwrite=True)
            self.assertTrue(runs_path.is_file())
            self.assertFalse(lock_path.exists())

    def test_resume_repairs_truncated_tail_and_keeps_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            config = ExperimentConfig(
                name="durable-resume",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run-answer")],
                output_dir=root / "runs",
            )

            runs_path = run_experiment(config, overwrite=True)
            with runs_path.open("ab") as handle:
                handle.write(b'{"qa_id":"partial"')
            resumed_path = run_experiment(config, resume=True)
            rows = [json.loads(line) for line in resumed_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads((resumed_path.parent / "manifest.json").read_text(encoding="utf-8"))
            snapshot = json.loads((resumed_path.parent / "materialized_config.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest["summary"]["rows_skipped"], 1)
        self.assertTrue(manifest["summary"]["truncated_tail_repaired"])
        self.assertEqual(snapshot["name"], "durable-resume")

    def test_resume_rejects_changed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            config = ExperimentConfig(
                name="config-guard",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run-answer")],
                output_dir=root / "runs",
            )

            run_experiment(config, overwrite=True)
            changed = replace(config, max_evidence_chars=config.max_evidence_chars + 1)

            with self.assertRaisesRegex(ValueError, "does not match"):
                run_experiment(changed, resume=True)

    def test_tool_native_dry_run_writes_search_open_trace_and_opened_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            chunk = {
                "chunk_id": "native-hit",
                "section_id": "2A.04",
                "section_title": "Traffic Control Device Principles",
                "content_type": "Standard",
                "ordinal": 1,
                "page_printed": "42",
                "text": "The dry run answer uses this native tool evidence.",
            }
            (mrag_dir / "mmrag_cache_v3" / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                name="native-tool-dry-run",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["tool_native"],
                models=[ModelConfig(provider="openai", model="target-answer-model")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
                dry_run=True,
            )

            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["config"]["context_mode"], "tool_native")
        self.assertEqual(row["config"]["model_provider"], "openai")
        self.assertEqual(row["config"]["model"], "target-answer-model")
        self.assertIsNone(row["retrieval_error"])
        self.assertIsNone(row["model_error"])
        self.assertTrue(row["model_raw"]["native_tool_calls"])
        self.assertEqual([call["name"] for call in row["model_raw"]["tool_calls"]], ["search", "open"])
        self.assertEqual(row["model_raw"]["tool_native"]["opened_ids"], ["native-hit"])
        self.assertEqual([item["evidence_id"] for item in row["evidence"]], ["native-hit"])
        self.assertTrue(row["retrieval_debug"]["deferred_retrieval"])

    def test_config_dry_run_preserves_target_model_labels_without_calling_models_or_llm_grader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root)
            config = ExperimentConfig(
                name="dry-run-preview",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="openai", model="target-answer-model")],
                grader=GraderConfig(provider="openai", model="target-judge-model"),
                output_dir=root / "runs",
                dry_run=True,
            )

            with (
                patch("gems_rag.runner.build_model", side_effect=AssertionError("answer model should not be built")),
                patch("gems_rag.runner.grade_answer", side_effect=AssertionError("grader should not be called")),
            ):
                runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads((root / "runs" / "dry-run-preview" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(row["config"]["model_provider"], "openai")
        self.assertEqual(row["config"]["model"], "target-answer-model")
        self.assertEqual(row["config"]["grader_provider"], "openai")
        self.assertEqual(row["config"]["grader"], "target-judge-model")
        self.assertIn("DRY RUN", row["answer"])
        self.assertTrue(row["model_raw"]["dry_run"])
        self.assertGreater(row["model_raw"]["prompt_chars"], 0)
        self.assertIsNone(row["model_error"])
        self.assertIsNone(row["judge_error"])
        self.assertTrue(row["grader_raw"]["dry_run"])
        self.assertTrue(manifest["summary"]["dry_run"])

    def test_llm_grader_client_is_built_once_and_reused_across_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture(root, qa_count=2)
            fake_judge = FakeJudgeModel(ModelConfig(provider="openai", model="target-judge-model"))
            build_calls = []

            def fake_build_model(model_config: ModelConfig):
                build_calls.append((model_config.provider, model_config.model))
                if model_config.provider == "dry_run":
                    return DryRunModel(model_config)
                if model_config.provider == "openai":
                    return fake_judge
                raise AssertionError(f"unexpected model build: {model_config}")

            config = ExperimentConfig(
                name="reused-grader",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run-answer")],
                grader=GraderConfig(provider="openai", model="target-judge-model"),
                output_dir=root / "runs",
            )

            with (
                patch("gems_rag.runner.build_model", side_effect=fake_build_model),
                patch("gems_rag.grading.build_model", side_effect=AssertionError("grader should be prebuilt")),
            ):
                runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads((root / "runs" / "reused-grader" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(build_calls.count(("openai", "target-judge-model")), 1)
        self.assertEqual(fake_judge.calls, 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["model_raw"]["dry_run"] for row in rows))
        self.assertTrue(all(row["judge_scores"]["factual_accuracy"]["score"] == 5 for row in rows))
        self.assertTrue(all(row["judge_error"] is None for row in rows))
        self.assertIsNone(manifest["summary"]["grader_build_error"])


if __name__ == "__main__":
    unittest.main()
