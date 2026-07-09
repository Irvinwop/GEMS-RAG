from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from gem_rags import external_setup
from gem_rags.config import ExperimentConfig, RetrieverConfig, write_experiment_config


def _args(**overrides):
    values = {
        "only": None,
        "config": None,
        "skip": None,
        "dry_run": False,
        "force": False,
        "no_precheck": False,
        "allow_failures": False,
        "strict_skips": False,
        "timeout_s": 3600,
        "check_timeout_s": 60,
        "allow_missing_api_key": False,
        "local_openai_base_url": "http://localhost:8000/v1",
        "graphrag_method": "standard",
        "visrag_scope": "pages",
        "visrag_limit": None,
        "visrag_batch_size": 4,
        "hipporag_limit": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeRunner:
    def __init__(self, responses: list[subprocess.CompletedProcess[str]]) -> None:
        self.responses = list(responses)
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if not self.responses:
            raise AssertionError(f"unexpected command: {command}")
        return self.responses.pop(0)


def _completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=json.dumps(payload), stderr="")


class TestBuildExternalIndexes(unittest.TestCase):
    def test_local_openai_command_ordering_matches_adapter_parsers(self) -> None:
        plans = external_setup._adapter_plans(_args(allow_missing_api_key=True, force=True))

        self.assertEqual(
            plans["graphrag"].check_command,
            [".venv/bin/python", "scripts/query_graphrag_index.py", "--allow-missing-api-key", "check"],
        )
        self.assertEqual(
            plans["lightrag"].build_commands[0],
            [
                ".venv/bin/python",
                "scripts/query_lightrag_index.py",
                "index",
                "--force",
                "--base-url",
                "http://localhost:8000/v1",
                "--allow-missing-api-key",
            ],
        )
        self.assertEqual(
            plans["paperqa2"].check_command,
            [
                ".venv/bin/python",
                "scripts/query_paperqa_index.py",
                "--base-url",
                "http://localhost:8000/v1",
                "--allow-missing-api-key",
                "check",
            ],
        )

    def test_dry_run_reports_would_run_without_build_commands(self) -> None:
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2)])

        report = external_setup.build_external_indexes(_args(only="lightrag", dry_run=True), runner=runner)

        self.assertEqual(report["would_run"], ["lightrag"])
        self.assertEqual(report["needs_index"], ["lightrag"])
        self.assertEqual(report["query_ready"], [])
        self.assertEqual(report["setup_plan"][0]["action"], "run_build_commands")
        self.assertEqual(report["setup_plan"][0]["commands"], report["results"][0]["build_commands"])
        self.assertEqual(report["built"], [])
        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(runner.commands[0], [".venv/bin/python", "scripts/query_lightrag_index.py", "check"])

    def test_skips_when_environment_is_not_ready(self) -> None:
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": False}, returncode=2)])

        report = external_setup.build_external_indexes(_args(only="hipporag"), runner=runner)

        self.assertEqual(report["skipped"], ["hipporag"])
        self.assertEqual(report["needs_environment"], ["hipporag"])
        self.assertEqual(report["setup_plan"][0]["action"], "install_environment")
        self.assertEqual(report["setup_plan"][0]["commands"], [])
        self.assertEqual(report["results"][0]["status"], "skipped_not_environment_ready")
        self.assertEqual(len(runner.commands), 1)

    def test_runs_build_commands_and_final_check(self) -> None:
        runner = FakeRunner(
            [
                _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                _completed({"indexed": True}),
                _completed({"runnable": True, "environment_ready": True, "index_ready": True}),
            ]
        )

        report = external_setup.build_external_indexes(_args(only="paperqa2"), runner=runner)

        self.assertEqual(report["built"], ["paperqa2"])
        self.assertEqual(report["query_ready"], ["paperqa2"])
        self.assertEqual(report["failed"], [])
        self.assertEqual(report["setup_plan"][0]["action"], "none")
        self.assertEqual(
            runner.commands,
            [
                [".venv/bin/python", "scripts/query_paperqa_index.py", "check"],
                [".venv/bin/python", "scripts/query_paperqa_index.py", "index", "--defer-embedding"],
                [".venv/bin/python", "scripts/query_paperqa_index.py", "check"],
            ],
        )

    def test_check_only_not_ready_is_separated_from_index_builds(self) -> None:
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": False}, returncode=2)])

        report = external_setup.build_external_indexes(_args(only="mrag_reference", dry_run=True), runner=runner)

        self.assertEqual(report["check_only"], ["mrag_reference"])
        self.assertEqual(report["check_only_not_ready"], ["mrag_reference"])
        self.assertEqual(report["needs_index"], [])
        self.assertEqual(report["needs_environment"], [])
        self.assertEqual(report["setup_plan"][0]["action"], "install_environment_or_credentials")

    def test_config_restricts_setup_to_referenced_external_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = ExperimentConfig(
                name="from-config",
                retrievers=[
                    RetrieverConfig(
                        name="qdrant_command",
                        kind="external_command",
                        options={"command": [".venv/bin/python", "scripts/query_vector_db.py", "search", "--question", "{question}"]},
                    ),
                    RetrieverConfig(
                        name="lightrag",
                        kind="external_command",
                        options={"command": [".venv/bin/python", str(root / "scripts/query_lightrag_index.py"), "query", "--question", "{question}"]},
                    ),
                    RetrieverConfig(name="bm25", kind="bm25"),
                ],
            )
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            runner = FakeRunner(
                [
                    _completed({"runnable": True, "environment_ready": True}),
                    _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                ]
            )

            report = external_setup.build_external_indexes(_args(config=config_path, dry_run=True), runner=runner)

        self.assertEqual(report["selected"], ["qdrant_hash_vector_command", "lightrag"])
        self.assertEqual(report["query_ready"], ["qdrant_hash_vector_command"])
        self.assertEqual(report["needs_index"], ["lightrag"])
        self.assertEqual(
            [cmd[1] for cmd in runner.commands],
            ["scripts/query_vector_db.py", "scripts/query_lightrag_index.py"],
        )

    def test_config_and_only_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_path.write_text('{"name":"empty"}\n', encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "--only and --config"):
                external_setup.build_external_indexes(_args(config=config_path, only="lightrag"), runner=FakeRunner([]))


if __name__ == "__main__":
    unittest.main()
