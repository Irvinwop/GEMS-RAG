from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.cli import main
from gems_rag.config import ExperimentConfig, RetrieverConfig, load_experiment_config, write_experiment_config
from gems_rag.retriever_profiles import apply_retriever_profile, load_retriever_profile


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="profile-test",
        retrievers=[
            RetrieverConfig(
                name="graph",
                kind="external_command",
                options={
                    "command": [
                        "python",
                        "query_graphrag_index.py",
                        "--base-url",
                        "http://localhost:11434/v1",
                        "query",
                        "--question",
                        "{question}",
                        "--limit",
                        "99",
                    ],
                    "check_command": [
                        "python",
                        "query_graphrag_index.py",
                        "check",
                        "--limit=99",
                    ],
                },
            ),
            RetrieverConfig(name="untouched", kind="bm25"),
        ],
    )


def _profile() -> dict:
    return {
        "schema_version": 1,
        "name": "test-scope",
        "rules": [
            {
                "retrievers": ["graph"],
                "global_options": {"--working-dir": "scope/graph"},
                "subcommand_options": {"--limit": 8},
            }
        ],
    }


class TestRetrieverProfiles(unittest.TestCase):
    def test_profile_rewrites_query_and_check_idempotently(self) -> None:
        profiled, report = apply_retriever_profile(_config(), _profile())
        repeated, _ = apply_retriever_profile(profiled, _profile())

        query = profiled.retrievers[0].options["command"]
        check = profiled.retrievers[0].options["check_command"]
        self.assertLess(query.index("--working-dir"), query.index("query"))
        self.assertGreater(query.index("--limit"), query.index("query"))
        self.assertEqual(query[query.index("--limit") + 1], "8")
        self.assertEqual(check[-2:], ["--limit", "8"])
        self.assertEqual(profiled, repeated)
        self.assertEqual(profiled.retrievers[1], _config().retrievers[1])
        self.assertEqual(report["retrievers_modified"], ["graph"])
        self.assertEqual(report["commands_modified"], 2)

    def test_required_retriever_must_be_present(self) -> None:
        profile = _profile()
        profile["rules"][0]["retrievers"] = ["missing"]

        with self.assertRaisesRegex(ValueError, "requires missing retrievers"):
            apply_retriever_profile(_config(), profile)

    def test_false_boolean_removes_only_the_profiled_flag(self) -> None:
        config = _config()
        graph = config.retrievers[0]
        options = dict(graph.options)
        options["command"] = [
            "python",
            "query_graphrag_index.py",
            "query",
            "--smoke",
            "--question",
            "{question}",
        ]
        config = ExperimentConfig(
            name=config.name,
            retrievers=[RetrieverConfig(**{**graph.__dict__, "options": options})],
        )
        profile = {
            "schema_version": 1,
            "name": "remove-boolean",
            "rules": [
                {
                    "retrievers": ["graph"],
                    "command_keys": ["command"],
                    "subcommand_options": {"--smoke": False},
                }
            ],
        }

        profiled, _ = apply_retriever_profile(config, profile)

        command = profiled.retrievers[0].options["command"]
        self.assertNotIn("--smoke", command)
        self.assertIn("--question", command)

    def test_cli_loads_profile_and_writes_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            profile_path = root / "profile.json"
            output_path = root / "profiled.json"
            write_experiment_config(_config(), config_path)
            profile_path.write_text(json.dumps(_profile()), encoding="utf-8")

            code = main(
                [
                    "apply-retriever-profile",
                    str(config_path),
                    str(profile_path),
                    "--output",
                    str(output_path),
                ]
            )
            written = load_experiment_config(output_path)

        self.assertEqual(code, 0)
        self.assertIn("--working-dir", written.retrievers[0].options["command"])

    def test_loader_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            path.write_text(
                json.dumps({"schema_version": 2, "name": "bad", "rules": [{}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_retriever_profile(path)


if __name__ == "__main__":
    unittest.main()
