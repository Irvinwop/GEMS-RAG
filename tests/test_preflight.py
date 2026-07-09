from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gem_rags.preflight import _external_command_check, preflight_config


def _write_mrag_dataset(root: Path) -> tuple[Path, Path]:
    qa_path = root / "gold_qa.jsonl"
    qa_path.write_text(json.dumps({"qa_id": "qa_1", "question": "Question?", "gold_answer": {}}) + "\n", encoding="utf-8")
    mrag_dir = root / "MRAG"
    cache_dir = mrag_dir / "mmrag_cache_v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache_dir / "figures.jsonl").write_text("", encoding="utf-8")
    (cache_dir / "graph.gpickle").write_bytes(b"")
    return qa_path, mrag_dir


class TestPreflightExternalCommand(unittest.TestCase):
    def test_external_check_command_override_is_used(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python", "adapter.py", "check", "--allow-missing-api-key"],
            returncode=0,
            stdout='{"runnable": true, "api_key_present": false, "api_key_usable": true}',
            stderr="",
        )
        with patch("gem_rags.preflight.subprocess.run", return_value=completed) as run:
            result = _external_command_check(
                ["python", "adapter.py", "query", "--question", "{question}"],
                check_external=True,
                timeout_s=5,
                check_command=["python", "adapter.py", "check", "--allow-missing-api-key"],
            )

        run.assert_called_once()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["check_command"], ["python", "adapter.py", "check", "--allow-missing-api-key"])
        self.assertEqual(result["problems"], [])

    def test_missing_key_without_usable_flag_blocks_credentials(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python", "adapter.py", "check"],
            returncode=2,
            stdout='{"runnable": false, "api_key_env": "OPENAI_API_KEY", "api_key_present": false}',
            stderr="",
        )
        with patch("gem_rags.preflight.subprocess.run", return_value=completed):
            result = _external_command_check(
                ["python", "adapter.py", "query", "--question", "{question}"],
                check_external=True,
                timeout_s=5,
                check_command=["python", "adapter.py", "check"],
            )

        self.assertEqual(result["status"], "blocked_by_credentials")
        self.assertEqual(result["problems"], ["missing API key env var: OPENAI_API_KEY"])

    def test_missing_external_index_is_reported_as_blocked(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python", "adapter.py", "check"],
            returncode=2,
            stdout='{"runnable": false, "environment_ready": true, "api_key_usable": true, "index_ready": false, "working_dir": "/tmp/index"}',
            stderr="",
        )
        with patch("gem_rags.preflight.subprocess.run", return_value=completed):
            result = _external_command_check(
                ["python", "adapter.py", "query", "--question", "{question}"],
                check_external=True,
                timeout_s=5,
                check_command=["python", "adapter.py", "check"],
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["problems"], ["index not ready: /tmp/index"])

    def test_vector_db_script_check_is_inferred(self) -> None:
        result = _external_command_check(
            [".venv/bin/python", "scripts/query_vector_db.py", "search", "--question", "{question}"],
            check_external=False,
            timeout_s=5,
        )

        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["check_command"], [".venv/bin/python", "scripts/query_vector_db.py", "check"])


class TestPreflightConfig(unittest.TestCase):
    def test_dry_run_skips_live_model_and_grader_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_mrag_dataset(root)
            config = ExperimentConfig(
                name="dry-preflight",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="openai", model="target")],
                grader=GraderConfig(provider="openai", model="judge"),
                dry_run=True,
            )

            report = preflight_config(config, check_external=False)

        self.assertTrue(report["ok"])
        model_report = report["sections"]["models"][0]
        grader_report = report["sections"]["grader"]
        self.assertEqual(model_report["status"], "ready")
        self.assertEqual(model_report["backend"], "openai_compatible")
        self.assertTrue(model_report["dry_run"])
        self.assertEqual(model_report["missing_api_key_envs"], [])
        self.assertEqual(grader_report["status"], "ready")
        self.assertTrue(grader_report["dry_run"])
        self.assertEqual(grader_report["missing_api_key_envs"], [])
        self.assertEqual(report["blocking"], [])

    def test_dry_run_still_blocks_unknown_model_and_grader_providers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_mrag_dataset(root)
            config = ExperimentConfig(
                name="dry-preflight-unknowns",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="not-a-provider", model="target")],
                grader=GraderConfig(provider="not-a-grader", model="judge"),
                dry_run=True,
            )

            report = preflight_config(config, check_external=False)

        self.assertFalse(report["ok"])
        self.assertEqual(report["sections"]["models"][0]["status"], "blocked")
        self.assertEqual(report["sections"]["grader"]["status"], "blocked")
        self.assertEqual(
            [item["path"] for item in report["blocking"]],
            ["models[0].target", "grader"],
        )

    def test_preflight_blocks_unresolved_model_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_mrag_dataset(root)
            config = ExperimentConfig(
                name="placeholder-models",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="openai", model="replace-with-openai-small")],
                grader=GraderConfig(provider="openai", model="replace-with-final-judge"),
                dry_run=True,
            )

            report = preflight_config(config, check_external=False)

        self.assertFalse(report["ok"])
        self.assertEqual(report["sections"]["models"][0]["status"], "blocked")
        self.assertEqual(report["sections"]["models"][0]["problems"], ["unresolved model placeholder: replace-with-openai-small"])
        self.assertEqual(report["sections"]["grader"]["status"], "blocked")
        self.assertEqual(report["sections"]["grader"]["problems"], ["unresolved model placeholder: replace-with-final-judge"])


if __name__ == "__main__":
    unittest.main()
