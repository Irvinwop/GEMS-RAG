from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gems_rag.config import (
    ExperimentConfig,
    RagBackendConfig,
    RetrieverConfig,
    load_experiment_config,
    write_experiment_config,
)
from gems_rag.rag_backends import backend_command, configure_retriever_backend, rag_backend_from_payload


class TestRagBackends(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = RagBackendConfig(
            provider="local_openai",
            api_key_env="LOCAL_OPENAI_API_KEY",
            base_url="http://localhost:9000/v1",
            allow_missing_api_key=True,
            chat_model="qwen3:8b",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            vision_model="qwen2.5vl:7b",
            reasoning_effort="none",
        )

    def test_local_profile_defaults_and_validation(self) -> None:
        profile = rag_backend_from_payload({"provider": "local_openai"})

        self.assertEqual(profile.api_key_env, "LOCAL_OPENAI_API_KEY")
        self.assertTrue(profile.allow_missing_api_key)
        self.assertEqual(profile.base_url, "http://localhost:8000/v1")
        self.assertEqual(profile.reasoning_effort, "none")
        with self.assertRaisesRegex(ValueError, "absolute"):
            rag_backend_from_payload({"provider": "local_openai", "base_url": "localhost:8000"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            rag_backend_from_payload({"provider": "anthropic"})
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            rag_backend_from_payload(
                {"provider": "local_openai", "reasoning_effort": "maximum"}
            )

    def test_backend_command_respects_each_adapter_parser(self) -> None:
        graph = backend_command(
            ["python", "scripts/query_graphrag_index.py", "query", "--question", "{question}"],
            "graphrag",
            self.backend,
        )
        light = backend_command(
            ["python", "scripts/query_lightrag_index.py", "query", "--question", "{question}"],
            "lightrag",
            self.backend,
        )
        raganything = backend_command(
            ["python", "scripts/query_raganything_index.py", "query", "--question", "{question}"],
            "raganything",
            self.backend,
        )
        hippo = backend_command(
            ["python", "scripts/query_hipporag_index.py", "query", "--question", "{question}"],
            "hipporag",
            self.backend,
        )
        mega = backend_command(
            ["python", "scripts/query_megarag_index.py", "query", "--question", "{question}"],
            "megarag",
            self.backend,
        )
        paper = backend_command(
            ["python", "scripts/query_paperqa_index.py", "query", "--question", "{question}"],
            "paperqa2",
            self.backend,
        )
        paper_index = backend_command(
            ["python", "scripts/query_paperqa_index.py", "index"],
            "paperqa2",
            self.backend,
        )

        self.assertLess(graph.index("--base-url"), graph.index("query"))
        self.assertGreater(light.index("--base-url"), light.index("query"))
        self.assertGreater(raganything.index("--vision-model"), raganything.index("query"))
        self.assertLess(hippo.index("--llm-model"), hippo.index("query"))
        self.assertLess(paper.index("--base-url"), paper.index("query"))
        self.assertGreater(paper.index("--embedding"), paper.index("query"))
        self.assertGreater(paper_index.index("--embedding"), paper_index.index("index"))
        self.assertNotIn("--llm", paper_index)
        self.assertEqual(light[light.index("--embedding-dim") + 1], "768")
        self.assertEqual(raganything[raganything.index("--vision-model") + 1], "qwen2.5vl:7b")
        for command in [graph, light, raganything, hippo, mega]:
            self.assertEqual(command[command.index("--reasoning-effort") + 1], "none")
        self.assertNotIn("--reasoning-effort", paper)

    def test_configuration_is_idempotent_and_leaves_non_model_rags_unchanged(self) -> None:
        base = RetrieverConfig(
            name="light",
            kind="external_command",
            options={
                "command": ["python", "scripts/query_lightrag_index.py", "query"],
                "check_command": ["python", "scripts/query_lightrag_index.py", "check"],
            },
        )
        configured = configure_retriever_backend(base, "lightrag", self.backend)
        configured_twice = configure_retriever_backend(configured, "lightrag", self.backend)
        unaffected = configure_retriever_backend(base, "dpr", self.backend)

        self.assertEqual(configured_twice, configured)
        self.assertEqual(unaffected, base)
        self.assertEqual(configured.options["command"].count("--llm-model"), 1)

    def test_profile_round_trips_with_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            write_experiment_config(ExperimentConfig(name="profile", rag_backend=self.backend), path)
            loaded = load_experiment_config(path)

        self.assertEqual(loaded.rag_backend, self.backend)


if __name__ == "__main__":
    unittest.main()
