from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from gems_rag import external_setup
from gems_rag.config import ExperimentConfig, RagBackendConfig, RetrieverConfig, write_experiment_config


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
        "megarag_limit": None,
        "ingestion_mode": "shared_corpus",
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
            [
                ".venv/bin/python",
                "scripts/query_graphrag_index.py",
                "--base-url",
                "http://localhost:8000/v1",
                "--allow-missing-api-key",
                "check",
            ],
        )
        self.assertEqual(
            plans["lightrag"].build_commands[0],
            [".venv/bin/python", "scripts/export_mrag_corpus.py"],
        )
        self.assertEqual(
            plans["lightrag"].build_commands[1],
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
        for adapter in ["dpr", "graphrag", "lightrag", "raganything", "hipporag", "paperqa2"]:
            self.assertEqual(plans[adapter].build_commands[0], [".venv/bin/python", "scripts/export_mrag_corpus.py"])
        self.assertEqual(
            plans["dpr"].build_commands[1],
            [".venv/bin/python", "scripts/query_dpr_index.py", "index", "--force"],
        )
        self.assertEqual(
            plans["gfmrag"].build_commands,
            [
                [".venv/bin/python", "scripts/query_gfmrag_index.py", "prepare", "--force"],
                [".venv/bin/python", "scripts/query_gfmrag_index.py", "index", "--force"],
            ],
        )
        self.assertEqual(
            plans["megarag"].build_commands,
            [
                [".venv/bin/python", "scripts/query_megarag_index.py", "prepare"],
                [
                    ".venv/bin/python",
                    "scripts/query_megarag_index.py",
                    "--base-url",
                    "http://localhost:8000/v1",
                    "--allow-missing-api-key",
                    "index",
                    "--force",
                ],
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
        self.assertEqual(
            plans["hipporag"].check_command,
            [
                ".venv/bin/python",
                "scripts/query_hipporag_index.py",
                "--base-url",
                "http://localhost:8000/v1",
                "--allow-missing-api-key",
                "check",
            ],
        )
        self.assertEqual(
            plans["hipporag"].build_commands[1],
            [
                ".venv/bin/python",
                "scripts/query_hipporag_index.py",
                "--base-url",
                "http://localhost:8000/v1",
                "--allow-missing-api-key",
                "index",
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

    def test_native_pdf_mode_uses_upstream_pdf_parsers_without_shared_export(self) -> None:
        plans = external_setup._adapter_plans(_args(ingestion_mode="native_pdf", force=True))

        self.assertEqual(
            plans["raganything"].check_command[-2:],
            ["--ingestion-mode", "native_pdf"],
        )
        self.assertEqual(
            plans["raganything"].build_commands,
            [[
                ".venv/bin/python",
                "scripts/query_raganything_index.py",
                "index",
                "--force",
                "--ingestion-mode",
                "native_pdf",
            ]],
        )
        self.assertEqual(
            plans["paperqa2"].build_commands,
            [[
                ".venv/bin/python",
                "scripts/query_paperqa_index.py",
                "index",
                "--ingestion-mode",
                "native_pdf",
            ]],
        )
        self.assertEqual(
            plans["paperqa2"].check_command[-2:],
            ["--ingestion-mode", "native_pdf"],
        )

    def test_skips_when_environment_is_not_ready(self) -> None:
        runner = FakeRunner([_completed({"runnable": False, "environment_ready": False}, returncode=2)])

        report = external_setup.build_external_indexes(_args(only="hipporag"), runner=runner)

        self.assertEqual(report["skipped"], ["hipporag"])
        self.assertEqual(report["needs_environment"], ["hipporag"])
        self.assertEqual(report["setup_plan"][0]["action"], "install_environment")
        self.assertEqual(report["setup_plan"][0]["commands"], [])
        self.assertEqual(report["results"][0]["status"], "skipped_not_environment_ready")
        self.assertEqual(len(runner.commands), 1)

    def test_skips_when_model_service_is_not_ready(self) -> None:
        runner = FakeRunner(
            [
                _completed(
                    {
                        "runnable": False,
                        "environment_ready": True,
                        "model_service_ready": False,
                        "endpoint_reachable": False,
                    },
                    returncode=2,
                )
            ]
        )

        report = external_setup.build_external_indexes(
            _args(only="lightrag", dry_run=True, allow_missing_api_key=True),
            runner=runner,
        )

        self.assertEqual(report["needs_environment"], [])
        self.assertEqual(report["needs_model_service"], ["lightrag"])
        self.assertEqual(report["needs_index"], [])
        self.assertEqual(report["setup_plan"][0]["action"], "start_model_service_or_fix_credentials")
        self.assertEqual(report["results"][0]["status"], "skipped_model_service_unavailable")

    def test_runs_build_commands_and_final_check(self) -> None:
        runner = FakeRunner(
            [
                _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                _completed({"chunks": 2}),
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
                [".venv/bin/python", "scripts/export_mrag_corpus.py"],
                [".venv/bin/python", "scripts/query_paperqa_index.py", "index"],
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

    def test_config_infers_local_openai_setup_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = ExperimentConfig(
                name="local-openai-from-config",
                retrievers=[
                    RetrieverConfig(
                        name="lightrag_local",
                        kind="external_command",
                        options={
                            "command": [
                                ".venv/bin/python",
                                "scripts/query_lightrag_index.py",
                                "query",
                                "--base-url",
                                "http://localhost:9000/v1",
                                "--allow-missing-api-key",
                                "--question",
                                "{question}",
                            ],
                            "check_command": [
                                ".venv/bin/python",
                                "scripts/query_lightrag_index.py",
                                "check",
                                "--base-url",
                                "http://localhost:9000/v1",
                                "--allow-missing-api-key",
                            ],
                        },
                    )
                ],
            )
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            runner = FakeRunner(
                [_completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2)]
            )

            report = external_setup.build_external_indexes(_args(config=config_path, dry_run=True), runner=runner)

        self.assertTrue(report["allow_missing_api_key"])
        self.assertEqual(
            report["config_setup_options"],
            {"allow_missing_api_key": True, "local_openai_base_url": "http://localhost:9000/v1"},
        )
        self.assertEqual(
            runner.commands,
            [
                [
                    ".venv/bin/python",
                    "scripts/query_lightrag_index.py",
                    "check",
                    "--base-url",
                    "http://localhost:9000/v1",
                    "--allow-missing-api-key",
                ]
            ],
        )
        self.assertEqual(
            report["results"][0]["build_commands"][0],
            [".venv/bin/python", "scripts/export_mrag_corpus.py"],
        )
        self.assertEqual(
            report["results"][0]["build_commands"][1],
            [
                ".venv/bin/python",
                "scripts/query_lightrag_index.py",
                "index",
                "--base-url",
                "http://localhost:9000/v1",
                "--allow-missing-api-key",
            ],
        )

    def test_explicit_rag_backend_controls_index_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = ExperimentConfig(
                name="controlled-rag-backend",
                rag_backend=RagBackendConfig(
                    provider="local_openai",
                    api_key_env="LOCAL_OPENAI_API_KEY",
                    base_url="http://localhost:9100/v1",
                    allow_missing_api_key=True,
                    chat_model="qwen3:14b",
                    embedding_model="bge-m3",
                    embedding_dim=1024,
                    vision_model="qwen2.5-vl:7b",
                ),
                retrievers=[
                    RetrieverConfig(
                        name="graph",
                        kind="external_command",
                        options={
                            "command": [
                                ".venv/bin/python",
                                "scripts/query_graphrag_index.py",
                                "query",
                                "--question",
                                "{question}",
                            ]
                        },
                    ),
                    RetrieverConfig(
                        name="light",
                        kind="external_command",
                        options={
                            "command": [
                                ".venv/bin/python",
                                "scripts/query_lightrag_index.py",
                                "query",
                                "--question",
                                "{question}",
                            ]
                        },
                    ),
                ],
            )
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            runner = FakeRunner(
                [
                    _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                    _completed({"runnable": False, "environment_ready": True, "index_ready": False}, returncode=2),
                ]
            )

            report = external_setup.build_external_indexes(_args(config=config_path, dry_run=True), runner=runner)

        graph_plan = next(row for row in report["results"] if row["name"] == "graphrag")
        light_plan = next(row for row in report["results"] if row["name"] == "lightrag")
        graph_init = next(command for command in graph_plan["build_commands"] if "init" in command)
        light_index = light_plan["build_commands"][-1]
        self.assertEqual(graph_init[graph_init.index("--llm-model") + 1], "qwen3:14b")
        self.assertEqual(graph_init[graph_init.index("--embedding-model") + 1], "bge-m3")
        self.assertEqual(light_index[light_index.index("--embedding-dim") + 1], "1024")
        self.assertIn("--allow-missing-api-key", graph_plan["check_command"])
        self.assertEqual(report["config_setup_options"]["rag_backend"]["vision_model"], "qwen2.5-vl:7b")

    def test_config_and_only_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_path.write_text('{"name":"empty"}\n', encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "--only and --config"):
                external_setup.build_external_indexes(_args(config=config_path, only="lightrag"), runner=FakeRunner([]))


if __name__ == "__main__":
    unittest.main()
