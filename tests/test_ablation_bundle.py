from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.ablation_bundle import prepare_ablation_bundle
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig, load_experiment_config, write_experiment_config


def _write_base(root: Path) -> Path:
    qa_path = root / "MRAG" / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    rows = [
        {"qa_id": "qa_1", "question": "What standard applies?", "gold_answer": {}, "references": [{"section_id": "2A.01"}]},
        {"qa_id": "qa_2", "question": "Which figure is relevant?", "gold_answer": {}, "references": [], "gold_figures": ["fig_1"]},
    ]
    qa_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = ExperimentConfig(
        name="bundle-base",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=root / "MRAG"),
        retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
        context_modes=["injected"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
        grader=GraderConfig(provider="heuristic", model="heuristic"),
        output_dir=root / "runs",
    )
    config_path = root / "base.json"
    write_experiment_config(config, config_path)
    return config_path


def _write_model_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "defaults": {"options": {"temperature": 0, "max_tokens": 900}},
                "models": [
                    {"provider": "openai", "model": "gpt-small", "size": "small", "roles": ["answer"]},
                    {"provider": "qwen", "model": "qwen-large", "size": "large", "roles": ["answer"]},
                    {
                        "provider": "openai",
                        "model": "judge-final",
                        "size": "judge",
                        "roles": ["grader"],
                        "tags": ["judge", "final"],
                        "options": {"max_tokens": 1600},
                        "enabled": False,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_retriever_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "retrievers": [
                    {"name": "bm25", "kind": "bm25", "family": "local", "modes": ["lexical"], "tags": ["local"]},
                    {
                        "name": "lightrag_hybrid_context",
                        "kind": "external_command",
                        "family": "lightrag",
                        "modes": ["hybrid"],
                        "tags": ["external"],
                        "options": {"command": [".venv/bin/python", "scripts/query_lightrag_index.py", "query", "--question", "{question}"]},
                    },
                    {
                        "name": "self_rag_adaptive_bm25",
                        "kind": "self_rag_policy",
                        "family": "self_rag_policy",
                        "modes": ["adaptive_retrieval"],
                        "tags": ["local", "policy"],
                        "options": {
                            "mode": "adaptive_retrieval",
                            "base_retriever": {"name": "bm25", "kind": "bm25", "top_k": 2},
                        },
                    },
                    {
                        "name": "crag_bm25_corrective",
                        "kind": "crag_policy",
                        "family": "crag_policy",
                        "modes": ["corrective"],
                        "tags": ["local", "policy"],
                        "options": {
                            "primary_retriever": {"name": "bm25", "kind": "bm25", "top_k": 2},
                            "fallback_retriever": {"name": "bm25", "kind": "bm25", "top_k": 2},
                        },
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


class TestAblationBundle(unittest.TestCase):
    def test_prepare_ablation_bundle_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            report = prepare_ablation_bundle(
                base_config_path=config_path,
                name="small-bundle",
                output_dir=bundle_dir,
                qa_size=1,
                qa_seed=7,
                model_catalog_path=model_catalog,
                model_providers=["openai"],
                model_sizes=["small"],
                retriever_catalog_path=retriever_catalog,
                retriever_families=["local"],
                context_modes=["injected", "tool_search"],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                dry_run=True,
            )
            config = load_experiment_config(bundle_dir / "materialized_config.json")
            plan = json.loads((bundle_dir / "plan.json").read_text(encoding="utf-8"))
            artifact_exists = {
                "qa_split": Path(report["artifacts"]["qa_split"]).exists(),
                "models": Path(report["artifacts"]["models"]).exists(),
                "retrievers": Path(report["artifacts"]["retrievers"]).exists(),
            }

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["models"], 1)
        self.assertEqual(report["retrievers"], 1)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["row_estimate"], 2)
        self.assertEqual(report["total_model_calls"], 4)
        self.assertEqual(report["paid_model_calls"], 0)
        self.assertTrue(artifact_exists["qa_split"])
        self.assertTrue(artifact_exists["models"])
        self.assertTrue(artifact_exists["retrievers"])
        self.assertEqual(config.name, "small-bundle")
        self.assertTrue(config.dry_run)
        self.assertIsNone(config.dataset.limit)
        self.assertEqual(len(config.dataset.qa_ids or []), 1)
        self.assertEqual([model.model for model in config.models], ["gpt-small"])
        self.assertEqual([retriever.name for retriever in config.retrievers], ["bm25"])
        self.assertEqual(plan["dimensions"]["conditions"], 2)
        self.assertIn("sweep", report["next_commands"])
        self.assertNotIn("external_indexes", report["next_commands"])

    def test_prepare_ablation_bundle_adds_external_index_commands_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            report = prepare_ablation_bundle(
                base_config_path=config_path,
                name="external-bundle",
                output_dir=bundle_dir,
                qa_size=1,
                model_catalog_path=model_catalog,
                model_providers=["openai"],
                model_sizes=["small"],
                retriever_catalog_path=retriever_catalog,
                retriever_families=["lightrag"],
                context_modes=["injected"],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                dry_run=True,
            )
            materialized = bundle_dir / "materialized_config.json"

        self.assertEqual(
            report["next_commands"]["external_indexes_dry_run"],
            f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --config {materialized} --dry-run",
        )
        self.assertEqual(
            report["next_commands"]["external_indexes"],
            f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --config {materialized}",
        )

    def test_prepare_ablation_bundle_adds_upstream_export_commands_for_policy_retrievers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            report = prepare_ablation_bundle(
                base_config_path=config_path,
                name="policy-bundle",
                output_dir=bundle_dir,
                qa_size=1,
                model_catalog_path=model_catalog,
                model_providers=["openai"],
                model_sizes=["small"],
                retriever_catalog_path=retriever_catalog,
                retriever_families=["self_rag_policy", "crag_policy"],
                context_modes=["injected"],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                dry_run=True,
            )
            materialized = bundle_dir / "materialized_config.json"
            run_dir = root / "runs" / "policy-bundle"

        self.assertEqual(
            report["next_commands"]["upstream_inputs_self_rag_adaptive_bm25"],
            (
                "PYTHONPATH=src .venv/bin/python -m gem_rags.cli upstream-inputs "
                f"--config {materialized} --retriever self_rag_adaptive_bm25 "
                f"--format selfrag --out-dir {run_dir / 'upstream_inputs' / 'self_rag_adaptive_bm25'}"
            ),
        )
        self.assertEqual(
            report["next_commands"]["upstream_inputs_crag_bm25_corrective"],
            (
                "PYTHONPATH=src .venv/bin/python -m gem_rags.cli upstream-inputs "
                f"--config {materialized} --retriever crag_bm25_corrective "
                f"--format crag --out-dir {run_dir / 'upstream_inputs' / 'crag_bm25_corrective'}"
            ),
        )

    def test_prepare_ablation_bundle_can_select_grader_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            report = prepare_ablation_bundle(
                base_config_path=config_path,
                name="catalog-judge-bundle",
                output_dir=bundle_dir,
                qa_size=1,
                model_catalog_path=model_catalog,
                model_providers=["openai"],
                model_sizes=["small"],
                grader_from_catalog=True,
                grader_providers=["openai"],
                grader_sizes=["judge"],
                grader_tags=["final"],
                include_disabled_graders=True,
                retriever_catalog_path=retriever_catalog,
                retriever_families=["local"],
                context_modes=["injected"],
                dry_run=True,
            )
            config = load_experiment_config(bundle_dir / "materialized_config.json")

        self.assertEqual(report["grader"]["source"], "catalog")
        self.assertEqual(report["grader"]["model"], "judge-final")
        self.assertEqual(report["grader"]["options"]["max_tokens"], 1600)
        self.assertEqual(config.grader.model, "judge-final")
        self.assertEqual(config.grader.options["max_tokens"], 1600)

    def test_prepare_ablation_bundle_requires_enabled_catalog_grader_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            with self.assertRaisesRegex(ValueError, "selected no graders"):
                prepare_ablation_bundle(
                    base_config_path=config_path,
                    model_catalog_path=model_catalog,
                    model_providers=["openai"],
                    model_sizes=["small"],
                    grader_from_catalog=True,
                    grader_providers=["openai"],
                    grader_sizes=["judge"],
                    retriever_catalog_path=retriever_catalog,
                    retriever_families=["local"],
                )

    def test_prepare_ablation_bundle_rejects_grader_filters_without_catalog_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            with self.assertRaisesRegex(ValueError, "require --grader-from-catalog"):
                prepare_ablation_bundle(
                    base_config_path=config_path,
                    model_catalog_path=model_catalog,
                    model_providers=["openai"],
                    model_sizes=["small"],
                    grader_providers=["openai"],
                    retriever_catalog_path=retriever_catalog,
                    retriever_families=["local"],
                )

    def test_prepare_ablation_bundle_reports_budget_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_base(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            _write_model_catalog(model_catalog)
            _write_retriever_catalog(retriever_catalog)

            report = prepare_ablation_bundle(
                base_config_path=config_path,
                name="budgeted-bundle",
                output_dir=bundle_dir,
                qa_size=1,
                model_catalog_path=model_catalog,
                model_providers=["openai"],
                model_sizes=["small"],
                retriever_catalog_path=retriever_catalog,
                retriever_families=["local"],
                context_modes=["injected", "tool_search"],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                max_rows=1,
            )
            plan = json.loads((bundle_dir / "plan.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["budget_ok"])
        self.assertEqual(report["budget"]["exceeded"][0]["name"], "rows")
        self.assertFalse(plan["budget"]["ok"])
        self.assertIn("--max-rows 1", report["next_commands"]["sweep"])


if __name__ == "__main__":
    unittest.main()
