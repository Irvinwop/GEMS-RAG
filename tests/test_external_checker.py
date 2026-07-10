from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_external_adapters.py"
    spec = importlib.util.spec_from_file_location("check_external_adapters", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalChecker(unittest.TestCase):
    def test_checker_covers_command_backed_adapter_families(self) -> None:
        mod = _load_checker()
        self.assertEqual(
            [item["name"] for item in mod.CHECKS],
            [
                "qdrant_hash_vector_command",
                "dpr",
                "gfmrag",
                "megarag",
                "mrag_reference",
                "graphrag",
                "lightrag",
                "raganything",
                "hipporag",
                "visrag",
                "paperqa2",
            ],
        )

    def test_local_openai_options_are_inserted_by_adapter_shape(self) -> None:
        mod = _load_checker()
        args = argparse.Namespace(allow_missing_api_key=True, local_openai_base_url="http://localhost:9000/v1")

        graphrag = mod._with_local_openai_options(
            {"name": "graphrag", "command": ["py", "scripts/query_graphrag_index.py", "check"]},
            args,
        )
        lightrag = mod._with_local_openai_options(
            {"name": "lightrag", "command": ["py", "scripts/query_lightrag_index.py", "check"]},
            args,
        )
        paperqa = mod._with_local_openai_options(
            {"name": "paperqa2", "command": ["py", "scripts/query_paperqa_index.py", "check"]},
            args,
        )
        megarag = mod._with_local_openai_options(
            {"name": "megarag", "command": ["py", "scripts/query_megarag_index.py", "check"]},
            args,
        )

        self.assertEqual(
            graphrag["command"],
            [
                "py",
                "scripts/query_graphrag_index.py",
                "--base-url",
                "http://localhost:9000/v1",
                "--allow-missing-api-key",
                "check",
            ],
        )
        self.assertEqual(
            lightrag["command"],
            ["py", "scripts/query_lightrag_index.py", "check", "--base-url", "http://localhost:9000/v1", "--allow-missing-api-key"],
        )
        self.assertEqual(
            paperqa["command"],
            ["py", "scripts/query_paperqa_index.py", "--base-url", "http://localhost:9000/v1", "--allow-missing-api-key", "check"],
        )
        self.assertEqual(
            megarag["command"],
            [
                "py",
                "scripts/query_megarag_index.py",
                "--base-url",
                "http://localhost:9000/v1",
                "--allow-missing-api-key",
                "check",
            ],
        )

        vector_db = mod._with_local_openai_options(
            {"name": "qdrant_hash_vector_command", "command": ["py", "scripts/query_vector_db.py", "check"]},
            args,
        )
        self.assertEqual(vector_db["command"], ["py", "scripts/query_vector_db.py", "check"])

    def test_api_key_usable_prevents_credential_block_classification(self) -> None:
        mod = _load_checker()
        item = {
            "stdout_json": {
                "repo_found": True,
                "missing_or_failed_imports": {},
                "api_key_present": False,
                "api_key_usable": True,
            }
        }
        self.assertTrue(mod._environment_ready(item))
        self.assertFalse(mod._blocked_by_credentials(item))

    def test_explicit_environment_ready_is_authoritative(self) -> None:
        mod = _load_checker()
        item = {
            "stdout_json": {
                "runnable": False,
                "environment_ready": True,
                "index_ready": False,
                "api_key_usable": True,
            }
        }
        self.assertTrue(mod._environment_ready(item))
        self.assertFalse(mod._blocked_by_credentials(item))

    def test_model_service_block_is_separate_from_credentials(self) -> None:
        mod = _load_checker()
        item = {
            "stdout_json": {
                "environment_ready": True,
                "credential_available": True,
                "model_service_ready": False,
            }
        }

        self.assertFalse(mod._blocked_by_credentials(item))
        self.assertTrue(mod._blocked_by_model_service(item))


if __name__ == "__main__":
    unittest.main()
