from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from gem_rags import cli
from gem_rags.analysis import RUBRIC_KEYS
from gem_rags.cli import main
from gem_rags.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig, load_experiment_config, write_experiment_config
from gem_rags.matrix import load_model_specs_file
from gem_rags.qa_sets import write_qa_split


def _write_fixture_config(root: Path) -> Path:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa_1",
                "question": "What standard applies to signs?",
                "gold_answer": {"direct_answer": "Use the standard sign."},
                "references": [],
                "gold_figures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"placeholder")
    config = ExperimentConfig(
        name="sweep-mini",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
        retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
        context_modes=["injected", "tool_explore"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
        grader=GraderConfig(provider="heuristic", model="heuristic"),
        output_dir=root / "runs",
        max_evidence_chars=500,
    )
    config_path = root / "base.json"
    write_experiment_config(config, config_path)
    return config_path


class TestCli(unittest.TestCase):
    def test_sweep_writes_run_summary_and_context_compare(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["sweep", str(config_path), "--overwrite"])
            payload = json.loads(stdout.getvalue())
            run_dir = root / "runs" / "sweep-mini"

            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["validation_ok"])
            self.assertEqual(payload["rows"], 2)
            self.assertEqual(payload["matched_context_pairs"], 1)
            self.assertTrue((run_dir / "materialized_config.json").exists())
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "runs.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "validation.json").exists())
            self.assertTrue((run_dir / "context-compare.json").exists())
            self.assertTrue((run_dir / "context-pairs.csv").exists())

    def test_analyze_writes_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["sweep", str(config_path), "--overwrite"]), 0)
            run_dir = root / "runs" / "sweep-mini"
            output_dir = run_dir / "analysis"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "analyze",
                        str(run_dir / "runs.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--qa-path",
                        str(root / "MRAG" / "eval" / "gold_qa.jsonl"),
                        "--axis",
                        "context_mode",
                        "--baseline",
                        "injected",
                    ]
                )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(payload["candidate_values"], ["tool_explore"])
            self.assertEqual(payload["comparisons"][0]["matched_pairs"], 1)
            self.assertTrue((output_dir / "analysis.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "strata-summary.csv").exists())
            self.assertTrue((output_dir / "strata-comparisons.csv").exists())

    def test_sweep_writes_tool_search_context_compare(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            base = load_experiment_config(config_path)
            config = ExperimentConfig(
                name="tool-search-compare",
                dataset=base.dataset,
                retrievers=base.retrievers,
                context_modes=["injected", "tool_search"],
                models=base.models,
                grader=base.grader,
                output_dir=base.output_dir,
                max_evidence_chars=base.max_evidence_chars,
            )
            write_experiment_config(config, config_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["sweep", str(config_path), "--overwrite"])
            payload = json.loads(stdout.getvalue())
            run_dir = root / "runs" / "tool-search-compare"

            self.assertEqual(code, 0)
            self.assertEqual(payload["rows"], 2)
            self.assertIn("tool_search", payload["context_comparisons"])
            self.assertEqual(payload["context_comparisons"]["tool_search"]["matched_pairs"], 1)
            self.assertTrue((run_dir / "context-tool-search-compare.json").exists())
            self.assertTrue((run_dir / "context-tool-search-pairs.csv").exists())

    def test_materialize_accepts_qa_ids_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            split_path = root / "split.json"
            write_qa_split(split_path, {"qa_ids": ["qa_1"]})
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["materialize", str(config_path), "--qa-ids-file", str(split_path)])
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(payload["dataset"]["qa_ids"], ["qa_1"])

    def test_materialize_accepts_models_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            models_path = root / "models.txt"
            models_path.write_text(
                """
                openai:gpt-4.1-mini,max_tokens=300
                local_openai:llama-3.1-8b,base_url=http://localhost:8000/v1,max_tokens=700
                """,
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["materialize", str(config_path), "--models-file", str(models_path)])
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(
                [(item["provider"], item["model"]) for item in payload["models"]],
                [("openai", "gpt-4.1-mini"), ("local_openai", "llama-3.1-8b")],
            )
            self.assertEqual(payload["models"][1]["options"]["base_url"], "http://localhost:8000/v1")

    def test_plan_budget_limit_exits_nonzero_with_budget_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["plan", str(config_path), "--max-rows", "1"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 2)
        self.assertFalse(payload["budget"]["ok"])
        self.assertEqual(payload["budget"]["exceeded"][0]["name"], "rows")

    def test_sweep_budget_limit_blocks_before_running_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["sweep", str(config_path), "--overwrite", "--max-rows", "1"])
            payload = json.loads(stdout.getvalue())
            run_dir = root / "runs" / "sweep-mini"

            self.assertEqual(code, 2)
            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(payload["reason"], "budget")
            self.assertFalse(payload["budget"]["ok"])
            self.assertTrue((run_dir / "materialized_config.json").exists())
            self.assertTrue((run_dir / "plan.json").exists())
            self.assertFalse((run_dir / "runs.jsonl").exists())

    def test_validate_strict_fails_when_observed_token_budget_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            base = load_experiment_config(config_path)
            config = ExperimentConfig(
                name=base.name,
                dataset=base.dataset,
                retrievers=base.retrievers,
                context_modes=["injected"],
                models=base.models,
                grader=base.grader,
                output_dir=base.output_dir,
                max_evidence_chars=base.max_evidence_chars,
            )
            write_experiment_config(config, config_path)
            runs_path = root / "runs.jsonl"
            row = {
                "qa_id": "qa_1",
                "config": {
                    "experiment": "sweep-mini",
                    "retriever": "bm25",
                    "context_mode": "injected",
                    "model_provider": "dry_run",
                    "model": "dry-run",
                    "grader": "heuristic",
                },
                "answer": "Use the standard sign.",
                "evidence": [],
                "model_raw": {"usage": {"input_tokens": 60, "output_tokens": 20, "total_tokens": 80}},
                "grader_raw": {"model_raw": {"usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}}},
                "judge_scores": {key: {"score": 3, "note": ""} for key in RUBRIC_KEYS},
            }
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "validate",
                        str(config_path),
                        "--runs",
                        str(runs_path),
                        "--max-total-tokens",
                        "100",
                        "--strict",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["budget_ok"])
        self.assertEqual(payload["token_usage"]["total_tokens"], 120)
        self.assertEqual(payload["budget_checks"][0]["name"], "total_tokens")
        self.assertEqual(payload["budget_checks"][0]["actual"], 120)

    def test_model_matrix_writes_specs_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_path = root / "catalog.json"
            output_path = root / "models.txt"
            catalog_path.write_text(
                json.dumps(
                    {
                        "defaults": {"options": {"max_tokens": 900}},
                        "models": [
                            {"provider": "openai", "model": "gpt-small", "size": "small", "roles": ["answer"]},
                            {"provider": "openai", "model": "gpt-large", "size": "large", "roles": ["answer"]},
                            {"provider": "openai", "model": "judge", "size": "judge", "roles": ["grader"], "enabled": False},
                            {"provider": "qwen", "model": "qwen-small", "size": "small", "roles": ["answer"]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "model-matrix",
                        str(catalog_path),
                        "--providers",
                        "openai,qwen",
                        "--sizes",
                        "small",
                        "--output",
                        str(output_path),
                    ]
                )
            models = load_model_specs_file(output_path)

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), str(output_path))
        self.assertEqual([(model.provider, model.model) for model in models], [("openai", "gpt-small"), ("qwen", "qwen-small")])
        self.assertEqual(models[0].options["max_tokens"], 900)

    def test_retriever_matrix_writes_specs_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_path = root / "retrievers.json"
            output_path = root / "retrievers.generated.json"
            config_path = _write_fixture_config(root)
            catalog_path.write_text(
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
                                "options": {
                                    "command": [".venv/bin/python", "scripts/query_lightrag_index.py", "query", "--mode", "hybrid", "--question", "{question}"],
                                    "check_command": [".venv/bin/python", "scripts/query_lightrag_index.py", "check"],
                                },
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["retriever-matrix", str(catalog_path), "--families", "lightrag", "--output", str(output_path)])
            materialized_stdout = io.StringIO()
            with redirect_stdout(materialized_stdout):
                materialize_code = main(["materialize", str(config_path), "--retrievers-file", str(output_path)])
            payload = json.loads(materialized_stdout.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(materialize_code, 0)
        self.assertEqual(stdout.getvalue().strip(), str(output_path))
        self.assertEqual([(item["name"], item["kind"]) for item in payload["retrievers"]], [("lightrag_hybrid_context", "external_command")])
        self.assertEqual(payload["retrievers"][0]["options"]["command"][2], "query")

    def test_prepare_ablation_writes_catalog_driven_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            model_catalog.write_text(
                json.dumps(
                    {
                        "models": [
                            {"provider": "openai", "model": "gpt-small", "size": "small", "roles": ["answer"]},
                            {"provider": "qwen", "model": "qwen-large", "size": "large", "roles": ["answer"]},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            retriever_catalog.write_text(
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
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "prepare-ablation",
                        str(config_path),
                        "--name",
                        "bundle-cli",
                        "--output-dir",
                        str(bundle_dir),
                        "--qa-size",
                        "1",
                        "--model-catalog",
                        str(model_catalog),
                        "--model-providers",
                        "openai",
                        "--model-sizes",
                        "small",
                        "--retriever-catalog",
                        str(retriever_catalog),
                        "--retriever-families",
                        "local",
                        "--context-modes",
                        "injected,tool_search",
                        "--grader",
                        "heuristic:heuristic",
                        "--dry-run",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            generated_config = load_experiment_config(bundle_dir / "materialized_config.json")
            bundle_files_exist = {
                "qa_split": (bundle_dir / "qa_split.json").exists(),
                "models": (bundle_dir / "models.txt").exists(),
                "retrievers": (bundle_dir / "retrievers.json").exists(),
                "config": (bundle_dir / "materialized_config.json").exists(),
                "plan_json": (bundle_dir / "plan.json").exists(),
                "plan_csv": (bundle_dir / "plan.csv").exists(),
            }

        self.assertEqual(code, 0)
        self.assertEqual(payload["experiment"], "bundle-cli")
        self.assertTrue(payload["dry_run"])
        self.assertTrue(generated_config.dry_run)
        self.assertEqual(payload["row_estimate"], 2)
        self.assertEqual(payload["total_model_calls"], 4)
        self.assertEqual(payload["paid_model_calls"], 0)
        self.assertTrue(bundle_files_exist["qa_split"])
        self.assertTrue(bundle_files_exist["models"])
        self.assertTrue(bundle_files_exist["retrievers"])
        self.assertTrue(bundle_files_exist["config"])
        self.assertTrue(bundle_files_exist["plan_json"])
        self.assertTrue(bundle_files_exist["plan_csv"])
        self.assertIn("sweep", payload["next_commands"])

    def test_prepare_ablation_can_select_catalog_grader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            model_catalog = root / "models.json"
            retriever_catalog = root / "retrievers.json"
            bundle_dir = root / "bundle"
            model_catalog.write_text(
                json.dumps(
                    {
                        "models": [
                            {"provider": "openai", "model": "gpt-small", "size": "small", "roles": ["answer"]},
                            {
                                "provider": "openai",
                                "model": "judge-final",
                                "size": "judge",
                                "roles": ["grader"],
                                "tags": ["judge", "final"],
                                "options": {"max_tokens": 1600},
                                "enabled": False,
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            retriever_catalog.write_text(
                json.dumps(
                    {
                        "retrievers": [
                            {"name": "bm25", "kind": "bm25", "family": "local", "modes": ["lexical"], "tags": ["local"]},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "prepare-ablation",
                        str(config_path),
                        "--name",
                        "catalog-grader-cli",
                        "--output-dir",
                        str(bundle_dir),
                        "--qa-size",
                        "1",
                        "--model-catalog",
                        str(model_catalog),
                        "--model-providers",
                        "openai",
                        "--model-sizes",
                        "small",
                        "--grader-from-catalog",
                        "--grader-providers",
                        "openai",
                        "--grader-sizes",
                        "judge",
                        "--grader-tags",
                        "final",
                        "--include-disabled-graders",
                        "--retriever-catalog",
                        str(retriever_catalog),
                        "--retriever-families",
                        "local",
                        "--context-modes",
                        "injected",
                        "--dry-run",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            generated_config = load_experiment_config(bundle_dir / "materialized_config.json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["grader"]["source"], "catalog")
        self.assertEqual(payload["grader"]["model"], "judge-final")
        self.assertEqual(generated_config.grader.model, "judge-final")
        self.assertEqual(generated_config.grader.options["max_tokens"], 1600)

    def test_external_indexes_cli_delegates_to_setup_builder(self) -> None:
        report = {
            "root": "/tmp/project",
            "dry_run": True,
            "force": False,
            "allow_missing_api_key": True,
            "selected": ["lightrag"],
            "built": [],
            "already_ready": [],
            "check_only": [],
            "would_run": ["lightrag"],
            "skipped": [],
            "failed": [],
            "results": [],
        }
        stdout = io.StringIO()

        with patch("gem_rags.cli.build_external_indexes", return_value=report) as build, redirect_stdout(stdout):
            code = main(["external-indexes", "--only", "lightrag", "--dry-run", "--allow-missing-api-key"])
        payload = json.loads(stdout.getvalue())
        args = build.call_args.args[0]

        self.assertEqual(code, 0)
        self.assertEqual(payload["would_run"], ["lightrag"])
        self.assertEqual(args.only, "lightrag")
        self.assertIsNone(args.config)
        self.assertTrue(args.dry_run)
        self.assertTrue(args.allow_missing_api_key)

    def test_external_indexes_cli_accepts_config(self) -> None:
        report = {
            "root": "/tmp/project",
            "dry_run": True,
            "force": False,
            "allow_missing_api_key": False,
            "selected": ["lightrag"],
            "built": [],
            "already_ready": [],
            "check_only": [],
            "would_run": ["lightrag"],
            "skipped": [],
            "failed": [],
            "results": [],
        }
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "materialized_config.json"
            stdout = io.StringIO()

            with patch("gem_rags.cli.build_external_indexes", return_value=report) as build, redirect_stdout(stdout):
                code = main(["external-indexes", "--config", str(config_path), "--dry-run"])
            payload = json.loads(stdout.getvalue())
            args = build.call_args.args[0]

        self.assertEqual(code, 0)
        self.assertEqual(payload["selected"], ["lightrag"])
        self.assertEqual(args.config, config_path)
        self.assertTrue(args.dry_run)

    def test_upstream_inputs_cli_exports_config_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            cache = root / "MRAG" / "mmrag_cache_v3"
            (cache / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk-1",
                        "section_id": "2A.04",
                        "content_type": "Standard",
                        "ordinal": "01",
                        "section_title": "Standardization of Application",
                        "page_printed": "23",
                        "part": "Part 2",
                        "chapter": "2A",
                        "text": "Standard signs apply to signs and support the requested standard.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            base = load_experiment_config(config_path)
            write_experiment_config(
                ExperimentConfig(
                    name=base.name,
                    dataset=base.dataset,
                    retrievers=[
                        RetrieverConfig(
                            name="self_policy",
                            kind="self_rag_policy",
                            top_k=1,
                            options={
                                "mode": "always_retrieve",
                                "base_retriever": {"name": "bm25", "kind": "bm25", "top_k": 1},
                            },
                        )
                    ],
                    context_modes=base.context_modes,
                    models=base.models,
                    grader=base.grader,
                    output_dir=base.output_dir,
                    max_evidence_chars=base.max_evidence_chars,
                    dry_run=base.dry_run,
                ),
                config_path,
            )
            out_dir = root / "upstream"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "upstream-inputs",
                        "--config",
                        str(config_path),
                        "--retriever",
                        "self_policy",
                        "--format",
                        "selfrag",
                        "--out-dir",
                        str(out_dir),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            selfrag_output_exists = Path(payload["outputs"]["selfrag_jsonl"]).exists()
            manifest_exists = Path(payload["outputs"]["manifest"]).exists()

        self.assertEqual(code, 0)
        self.assertEqual(payload["formats"], ["selfrag"])
        self.assertEqual(payload["retriever"]["name"], "self_policy")
        self.assertTrue(selfrag_output_exists)
        self.assertTrue(manifest_exists)
        self.assertIn("selfrag_run_short_form", payload["upstream_commands"])

    def test_cli_runs_from_project_root_for_repo_relative_configs(self) -> None:
        workspace = cli.ROOT / "data" / "working" / "test-cli-cwd"
        workspace.mkdir(parents=True, exist_ok=True)
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(dir=workspace) as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir(parents=True)
            qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
            qa_path.parent.mkdir(parents=True)
            qa_path.write_text('{"qa_id":"qa_1","question":"Q?","gold_answer":{},"references":[]}\n', encoding="utf-8")
            (cache / "chunks.jsonl").write_text("", encoding="utf-8")
            (cache / "figures.jsonl").write_text("", encoding="utf-8")
            (cache / "graph.gpickle").write_bytes(b"graph")
            rel_qa = qa_path.relative_to(cli.ROOT)
            rel_mrag = mrag_dir.relative_to(cli.ROOT)
            config_path = root / "repo-relative.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "repo-relative",
                        "dataset": {"qa_path": str(rel_qa), "mrag_dir": str(rel_mrag), "limit": 1},
                        "retrievers": [{"name": "bm25", "kind": "bm25"}],
                        "context_modes": ["injected"],
                        "models": [{"provider": "dry_run", "model": "dry-run"}],
                        "grader": {"provider": "heuristic", "model": "heuristic"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            try:
                os.chdir("/tmp")
                with redirect_stdout(stdout):
                    code = main(["preflight", str(config_path), "--no-external-checks"])
            finally:
                os.chdir(previous_cwd)
            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["sections"]["dataset"]["status"], "ready")
        self.assertEqual(payload["sections"]["dataset"]["qa_count"], 1)

    def test_run_retry_errors_replaces_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = _write_fixture_config(root)
            base = load_experiment_config(config_path)
            config = ExperimentConfig(
                name="retry-cli",
                dataset=base.dataset,
                retrievers=[
                    RetrieverConfig(
                        name="external_slot",
                        kind="external_command",
                        options={"command": [sys.executable, "-c", "import sys; sys.exit(9)"]},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            write_experiment_config(config, config_path)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(config_path), "--overwrite"]), 0)

            fixed = ExperimentConfig(
                name=config.name,
                dataset=config.dataset,
                retrievers=[
                    RetrieverConfig(
                        name="external_slot",
                        kind="external_command",
                        options={"command": [sys.executable, "-c", "print('{{\"contexts\": []}}')"]},
                    )
                ],
                context_modes=config.context_modes,
                models=config.models,
                grader=config.grader,
                output_dir=config.output_dir,
            )
            write_experiment_config(fixed, config_path)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(config_path), "--retry-errors"]), 0)

            runs_path = root / "runs" / "retry-cli" / "runs.jsonl"
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["retrieval_error"])


if __name__ == "__main__":
    unittest.main()
