from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "build_external_indexes.py"
    spec = importlib.util.spec_from_file_location("build_external_indexes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "only": None,
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
        mod = _load_script()
        plans = mod._adapter_plans(_args(allow_missing_api_key=True, force=True))

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
        mod = _load_script()
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2)])

        report = mod.build_external_indexes(_args(only="lightrag", dry_run=True), runner=runner)

        self.assertEqual(report["would_run"], ["lightrag"])
        self.assertEqual(report["built"], [])
        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(runner.commands[0], [".venv/bin/python", "scripts/query_lightrag_index.py", "check"])

    def test_skips_when_environment_is_not_ready(self) -> None:
        mod = _load_script()
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": False}, returncode=2)])

        report = mod.build_external_indexes(_args(only="hipporag"), runner=runner)

        self.assertEqual(report["skipped"], ["hipporag"])
        self.assertEqual(report["results"][0]["status"], "skipped_not_environment_ready")
        self.assertEqual(len(runner.commands), 1)

    def test_runs_build_commands_and_final_check(self) -> None:
        mod = _load_script()
        runner = FakeRunner(
            [
                _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                _completed({"indexed": True}),
                _completed({"runnable": True, "environment_ready": True, "index_ready": True}),
            ]
        )

        report = mod.build_external_indexes(_args(only="paperqa2"), runner=runner)

        self.assertEqual(report["built"], ["paperqa2"])
        self.assertEqual(report["failed"], [])
        self.assertEqual(
            runner.commands,
            [
                [".venv/bin/python", "scripts/query_paperqa_index.py", "check"],
                [".venv/bin/python", "scripts/query_paperqa_index.py", "index", "--defer-embedding"],
                [".venv/bin/python", "scripts/query_paperqa_index.py", "check"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
