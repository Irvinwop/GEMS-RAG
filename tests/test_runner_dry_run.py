from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.models import DryRunModel
from gem_rags.runner import run_experiment
from gem_rags.types import ModelResult


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
                patch("gem_rags.runner.build_model", side_effect=AssertionError("answer model should not be built")),
                patch("gem_rags.runner.grade_answer", side_effect=AssertionError("grader should not be called")),
            ):
                runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads((root / "runs" / "dry-run-preview" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(row["config"]["model_provider"], "openai")
        self.assertEqual(row["config"]["model"], "target-answer-model")
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
                patch("gem_rags.runner.build_model", side_effect=fake_build_model),
                patch("gem_rags.grading.build_model", side_effect=AssertionError("grader should be prebuilt")),
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
