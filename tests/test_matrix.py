from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gem_rags.config import (
    DatasetConfig,
    ExperimentConfig,
    GraderConfig,
    ModelConfig,
    RetrieverConfig,
    load_experiment_config,
    write_experiment_config,
)
from gem_rags.matrix import filter_ready_config, load_model_specs_file, materialize_config, parse_grader_spec, parse_model_spec


def _base_config(tmp: Path) -> ExperimentConfig:
    mrag_dir = tmp / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text('{"qa_id":"qa_1","question":"What is required?","gold_answer":{},"references":[]}\n', encoding="utf-8")
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"exists")
    return ExperimentConfig(
        name="base",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
        retrievers=[
            RetrieverConfig(name="bm25", kind="bm25"),
            RetrieverConfig(name="missing_external", kind="external_placeholder", options={"path": str(tmp / "missing")}),
        ],
        context_modes=["injected", "tool_explore"],
        models=[
            ModelConfig(provider="dry_run", model="dry-run"),
            ModelConfig(provider="unknown", model="blocked"),
        ],
        grader=GraderConfig(provider="heuristic", model="heuristic"),
    )


class TestMatrix(unittest.TestCase):
    def test_parse_model_spec_coerces_options(self) -> None:
        model = parse_model_spec("local_openai:llama-8b,base_url=http://localhost:8000/v1,max_tokens=256,temperature=0.2,allow_missing_api_key=true")
        self.assertEqual(model.provider, "local_openai")
        self.assertEqual(model.model, "llama-8b")
        self.assertEqual(model.options["base_url"], "http://localhost:8000/v1")
        self.assertEqual(model.options["max_tokens"], 256)
        self.assertEqual(model.options["temperature"], 0.2)
        self.assertIs(model.options["allow_missing_api_key"], True)

    def test_load_model_specs_file_accepts_json_and_plain_specs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "models.json"
            json_path.write_text(
                """
                {
                  "models": [
                    "openai:gpt-4.1-mini,max_tokens=300",
                    {
                      "provider": "local_openai",
                      "model": "llama-3.1-8b",
                      "options": {"base_url": "http://localhost:8000/v1", "max_tokens": 500}
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            plain_path = root / "models.txt"
            plain_path.write_text(
                """
                # quick local/provider smoke set
                xai:grok-mini,api_key_env=XAI_API_KEY
                qwen:qwen3-8b,max_tokens=400
                """,
                encoding="utf-8",
            )

            json_models = load_model_specs_file(json_path)
            plain_models = load_model_specs_file(plain_path)

        self.assertEqual([(model.provider, model.model) for model in json_models], [("openai", "gpt-4.1-mini"), ("local_openai", "llama-3.1-8b")])
        self.assertEqual(json_models[1].options["base_url"], "http://localhost:8000/v1")
        self.assertEqual([(model.provider, model.model) for model in plain_models], [("xai", "grok-mini"), ("qwen", "qwen3-8b")])
        self.assertEqual(plain_models[1].options["max_tokens"], 400)

    def test_materialize_selects_retrievers_and_replaces_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = materialize_config(
                _base_config(Path(td)),
                name="small-sweep",
                limit=5,
                retriever_names=["bm25"],
                context_modes=["tool_explore"],
                models=[parse_model_spec("anthropic:anthropic/claude-small,max_tokens=400")],
                grader=parse_grader_spec("openai:gpt-judge,max_tokens=1000"),
            )
        self.assertEqual(config.name, "small-sweep")
        self.assertEqual(config.dataset.limit, 5)
        self.assertEqual([ret.name for ret in config.retrievers], ["bm25"])
        self.assertEqual(config.context_modes, ["tool_explore"])
        self.assertEqual(config.models[0].provider, "anthropic")
        self.assertEqual(config.models[0].options["max_tokens"], 400)
        self.assertEqual(config.grader.model, "gpt-judge")

    def test_materialize_can_force_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = materialize_config(_base_config(Path(td)), dry_run=True)
        self.assertTrue(config.dry_run)

    def test_write_config_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "generated.json"
            original = materialize_config(_base_config(Path(td)), retriever_names=["bm25"])
            write_experiment_config(original, path)
            loaded = load_experiment_config(path)
        self.assertEqual(loaded.name, original.name)
        self.assertEqual(loaded.dataset.qa_path, original.dataset.qa_path)
        self.assertEqual([ret.name for ret in loaded.retrievers], ["bm25"])
        self.assertEqual(loaded.models[0].provider, "dry_run")

    def test_filter_ready_drops_blocked_retrievers_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            filtered, report = filter_ready_config(_base_config(Path(td)))
        self.assertFalse(report["ok"])
        self.assertEqual([ret.name for ret in filtered.retrievers], ["bm25"])
        self.assertEqual([(model.provider, model.model) for model in filtered.models], [("dry_run", "dry-run")])

    def test_filter_ready_rejects_blocked_grader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = materialize_config(
                _base_config(Path(td)),
                retriever_names=["bm25"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="unknown", model="judge"),
            )
            with self.assertRaisesRegex(ValueError, "grader"):
                filter_ready_config(config)


if __name__ == "__main__":
    unittest.main()
