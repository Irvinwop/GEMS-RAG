from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.runner import run_experiment


def _fixture(root: Path) -> tuple[Path, Path]:
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
                "qa_id": "qa_dry",
                "question": "What is the dry run answer?",
                "gold_answer": {"direct_answer": "No model should be called."},
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return mrag_dir, qa_path


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
        self.assertIsNone(row["model_error"])
        self.assertIsNone(row["judge_error"])
        self.assertTrue(row["grader_raw"]["dry_run"])
        self.assertTrue(manifest["summary"]["dry_run"])


if __name__ == "__main__":
    unittest.main()
