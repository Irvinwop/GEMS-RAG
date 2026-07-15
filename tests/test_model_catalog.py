from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gems_rag.matrix import load_model_specs_file
from gems_rag.model_catalog import (
    catalog_pricing_payload,
    load_model_catalog,
    pricing_coverage_for_config,
    render_model_specs,
    select_model_catalog,
)


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "options": {"temperature": 0, "max_tokens": 900},
                    "provider_options": {
                        "local_openai": {
                            "base_url": "http://localhost:8000/v1",
                            "allow_missing_api_key": True,
                        }
                    },
                },
                "models": [
                    {
                        "provider": "openai",
                        "model": "gpt-small",
                        "size": "small",
                        "roles": ["answer"],
                        "tags": ["api", "closed"],
                        "pricing": {"input_per_1m": 1.0, "output_per_1m": 2.0},
                    },
                    {
                        "provider": "anthropic",
                        "model": "claude-medium",
                        "size": "medium",
                        "roles": ["answer"],
                        "tags": ["api", "closed", "litellm"],
                        "enabled": False,
                    },
                    {
                        "provider": "local_openai",
                        "model": "llama-small",
                        "size": "small",
                        "roles": ["answer"],
                        "tags": ["local", "openai-compatible"],
                    },
                    {
                        "provider": "openai",
                        "model": "judge",
                        "size": "judge",
                        "roles": ["grader"],
                        "tags": ["api", "judge"],
                        "enabled": False,
                        "options": {"max_tokens": 1600},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class TestModelCatalog(unittest.TestCase):
    def test_example_catalog_covers_the_full_answer_model_menu(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "model-catalog.example.json"
        entries = load_model_catalog(catalog_path)
        answer_entries = [entry for entry in entries if "answer" in entry.roles]
        model_ids = {(entry.config.provider, entry.config.model) for entry in answer_entries}
        provider_counts = {
            provider: sum(entry.config.provider == provider for entry in answer_entries)
            for provider in {entry.config.provider for entry in answer_entries}
        }

        self.assertEqual(len(model_ids), 87)
        self.assertEqual(
            provider_counts,
            {"openai": 21, "anthropic": 11, "xai": 5, "qwen": 47, "local_openai": 3},
        )
        self.assertTrue(
            {
                ("openai", "gpt-5.6-luna"),
                ("openai", "gpt-5.6-sol"),
                ("anthropic", "claude-haiku-4-5"),
                ("anthropic", "claude-fable-5"),
                ("xai", "grok-4.5"),
                ("qwen", "qwen3.6-flash"),
                ("qwen", "qwen3-vl-235b-a22b-instruct"),
                ("qwen", "qwen3-vl-2b-instruct"),
                ("local_openai", "local-small"),
                ("local_openai", "local-large"),
            }.issubset(model_ids)
        )

    def test_catalog_filters_and_merges_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "catalog.json"
            _write_catalog(catalog_path)
            entries = load_model_catalog(catalog_path)

        selected = select_model_catalog(entries, sizes=["small"], roles=["answer"])

        self.assertEqual([(entry.config.provider, entry.config.model) for entry in selected], [("openai", "gpt-small"), ("local_openai", "llama-small")])
        self.assertEqual(selected[0].config.options["temperature"], 0)
        self.assertEqual(selected[0].pricing, {"input_per_1m": 1.0, "output_per_1m": 2.0})
        self.assertEqual(selected[1].config.options["base_url"], "http://localhost:8000/v1")
        self.assertIs(selected[1].config.options["allow_missing_api_key"], True)

    def test_catalog_skips_disabled_unless_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "catalog.json"
            _write_catalog(catalog_path)
            entries = load_model_catalog(catalog_path)

        enabled = select_model_catalog(entries, providers=["anthropic"], roles=["answer"])
        disabled = select_model_catalog(entries, providers=["anthropic"], roles=["answer"], include_disabled=True)
        graders = select_model_catalog(entries, roles=["grader"], include_disabled=True)

        self.assertEqual(enabled, [])
        self.assertEqual([(entry.config.provider, entry.config.model) for entry in disabled], [("anthropic", "claude-medium")])
        self.assertEqual([(entry.config.provider, entry.config.model) for entry in graders], [("openai", "judge")])
        self.assertEqual(graders[0].config.options["max_tokens"], 1600)

    def test_catalog_pricing_payload_exposes_provider_and_unique_model_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "catalog.json"
            _write_catalog(catalog_path)
            pricing = catalog_pricing_payload(load_model_catalog(catalog_path))

        self.assertEqual(pricing["openai:gpt-small"]["input_per_1m"], 1.0)
        self.assertEqual(pricing["gpt-small"]["output_per_1m"], 2.0)
        self.assertNotIn("local_openai:llama-small", pricing)

    def test_pricing_coverage_requires_every_paid_answer_and_judge_model(self) -> None:
        config = ExperimentConfig(
            name="priced",
            dataset=DatasetConfig(qa_path=Path("qa.jsonl"), mrag_dir=Path("MRAG")),
            retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
            context_modes=["injected"],
            models=[ModelConfig(provider="openai", model="answer")],
            grader=GraderConfig(provider="openai", model="judge"),
        )
        pricing = {
            "openai:answer": {"input_per_1m": 1.0, "output_per_1m": 2.0},
        }

        incomplete = pricing_coverage_for_config(config, pricing)
        dry_run = pricing_coverage_for_config(
            ExperimentConfig(
                name=config.name,
                dataset=config.dataset,
                retrievers=config.retrievers,
                context_modes=config.context_modes,
                models=config.models,
                grader=config.grader,
                dry_run=True,
            ),
            None,
        )

        self.assertFalse(incomplete["ok"])
        self.assertEqual(incomplete["required_models"], 2)
        self.assertEqual(incomplete["missing"][0]["role"], "judge")
        self.assertTrue(dry_run["ok"])
        self.assertEqual(dry_run["required_models"], 0)

    def test_catalog_rejects_negative_pricing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "provider": "openai",
                                "model": "bad-price",
                                "pricing": {"input_per_1m": -1, "output_per_1m": 2},
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                load_model_catalog(path)

    def test_rendered_specs_round_trip_through_models_file_parser(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_path = root / "catalog.json"
            specs_path = root / "models.txt"
            _write_catalog(catalog_path)

            entries = select_model_catalog(load_model_catalog(catalog_path), tags=["local"])
            specs_path.write_text(render_model_specs(entries), encoding="utf-8")
            models = load_model_specs_file(specs_path)

        self.assertEqual([(model.provider, model.model) for model in models], [("local_openai", "llama-small")])
        self.assertEqual(models[0].options["base_url"], "http://localhost:8000/v1")


if __name__ == "__main__":
    unittest.main()
