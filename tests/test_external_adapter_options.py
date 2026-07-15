from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalAdapterOptions(unittest.TestCase):
    def test_mrag_flagembedding_compat_translates_dtype_once(self) -> None:
        mod = _load_script("query_mrag_reference.py")
        calls = []

        class AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls.append((args, kwargs))
                return "model"

        transformers = SimpleNamespace(AutoModel=AutoModel)
        with patch.dict("sys.modules", {"transformers": transformers}):
            mod._install_flagembedding_transformers_compat()
            mod._install_flagembedding_transformers_compat()

        result = AutoModel.from_pretrained("model-id", dtype="float32", revision="main")

        self.assertEqual(result, "model")
        self.assertEqual(calls, [(("model-id",), {"torch_dtype": "float32", "revision": "main"})])

    def test_mrag_dependency_check_is_mode_specific(self) -> None:
        mod = _load_script("query_mrag_reference.py")
        available = ({*mod.REQUIRED_MODULES, "FlagEmbedding"} - {"networkx"})

        def find_spec(name):
            return object() if name in available else None

        args = argparse.Namespace(python=Path("missing-python"), mode="dense")
        with patch.object(mod.importlib.util, "find_spec", side_effect=find_spec):
            dense = mod._dependency_report(args)
        self.assertTrue(dense["runnable"])
        self.assertEqual(dense["missing_alternative_groups"], [])

        args.mode = "full"
        with patch.object(mod.importlib.util, "find_spec", side_effect=find_spec):
            full = mod._dependency_report(args)
        self.assertFalse(full["runnable"])
        self.assertEqual(
            {group["group"] for group in full["missing_alternative_groups"]},
            {"graph", "reranking", "visual_embedding"},
        )

    def test_local_openai_adapter_checks_require_reachable_endpoint(self) -> None:
        down = {
            "checked": True,
            "url": "http://localhost:8000/v1/models",
            "reachable": False,
            "authorized": None,
            "usable": False,
            "status_code": None,
            "error": "URLError",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = []

            lightrag = _load_script("query_lightrag_index.py")
            lightrag_repo = root / "lightrag"
            lightrag_repo.mkdir()
            lightrag_index = root / "lightrag-index"
            lightrag_index.mkdir()
            (lightrag_index / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
            cases.append(
                (
                    lightrag,
                    argparse.Namespace(
                        repo=lightrag_repo,
                        working_dir=lightrag_index,
                        corpus=root / "corpus.txt",
                        api_key_env="OPENAI_API_KEY",
                        allow_missing_api_key=True,
                        base_url="http://localhost:8000/v1",
                    ),
                )
            )

            raganything = _load_script("query_raganything_index.py")
            raganything_repo = root / "raganything"
            raganything_repo.mkdir()
            raganything_index = root / "raganything-index"
            raganything_index.mkdir()
            (raganything_index / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
            cases.append(
                (
                    raganything,
                    argparse.Namespace(
                        repo=raganything_repo,
                        lightrag_repo=lightrag_repo,
                        working_dir=raganything_index,
                        content_list=root / "content.json",
                        api_key_env="OPENAI_API_KEY",
                        allow_missing_api_key=True,
                        base_url="http://localhost:8000/v1",
                    ),
                )
            )

            paperqa = _load_script("query_paperqa_index.py")
            paperqa_repo = root / "paperqa"
            paperqa_repo.mkdir()
            paperqa_index = root / "docs.pkl"
            paperqa_index.write_bytes(b"pickle")
            cases.append(
                (
                    paperqa,
                    argparse.Namespace(
                        repo=paperqa_repo,
                        index=paperqa_index,
                        chunks=root / "chunks.jsonl",
                        api_key_env="OPENAI_API_KEY",
                        allow_missing_api_key=True,
                        base_url="http://localhost:8000/v1",
                    ),
                )
            )

            for mod, args in cases:
                with self.subTest(adapter=mod.__name__):
                    with (
                        patch.object(mod, "_import_errors", return_value={}),
                        patch.object(mod, "probe_openai_endpoint", return_value=down),
                        patch.dict(os.environ, {}, clear=True),
                    ):
                        report = mod._dependency_report(args)
                    self.assertTrue(report["environment_ready"])
                    self.assertTrue(report["index_ready"])
                    self.assertFalse(report["endpoint_reachable"])
                    self.assertFalse(report["model_service_ready"])
                    self.assertFalse(report["runnable"])

    def test_lightrag_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_raganything_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_paperqa_sets_dummy_local_key(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._ensure_api_key(args)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local")

    def test_graphrag_applies_dummy_local_key_to_subprocess_env(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            env = mod._env(repo)
        args = argparse.Namespace(api_key_env="GRAPHRAG_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._apply_local_api_key(args, env)
            self.assertEqual(env["GRAPHRAG_API_KEY"], "local")

    def test_graphrag_reuses_openai_key_by_default(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            env = mod._env(repo)
        args = argparse.Namespace(api_key_env="GRAPHRAG_API_KEY", allow_missing_api_key=False)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            mod._apply_local_api_key(args, env)

        self.assertEqual(env["GRAPHRAG_API_KEY"], "openai-key")

    def test_graphrag_configures_local_api_base_for_completion_and_embedding(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            settings = Path(td) / "settings.yaml"
            settings.write_text(
                """
completion_models:
  default_completion_model:
    model_provider: openai
    model: answer
embedding_models:
  default_embedding_model:
    model_provider: openai
    model: embed
""".lstrip(),
                encoding="utf-8",
            )

            with patch("builtins.print"):
                code = mod._configure_api_base(settings, "http://localhost:8000/v1")
            import yaml

            payload = yaml.safe_load(settings.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            payload["completion_models"]["default_completion_model"]["api_base"],
            "http://localhost:8000/v1",
        )
        self.assertEqual(
            payload["embedding_models"]["default_embedding_model"]["api_base"],
            "http://localhost:8000/v1",
        )

    def test_lightrag_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            working_dir = root / "index"
            working_dir.mkdir()
            args = argparse.Namespace(
                repo=repo,
                working_dir=working_dir,
                corpus=root / "corpus.txt",
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                (working_dir / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

    def test_raganything_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_raganything_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "raganything"
            lightrag_repo = root / "lightrag"
            repo.mkdir()
            lightrag_repo.mkdir()
            working_dir = root / "index"
            working_dir.mkdir()
            args = argparse.Namespace(
                repo=repo,
                lightrag_repo=lightrag_repo,
                working_dir=working_dir,
                content_list=root / "content.json",
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                (working_dir / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

    def test_raganything_context_query_disables_vlm_and_emits_contexts(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(
            mode="hybrid",
            question="What does Section 2A.04 require?",
            top_k=5,
            chunk_top_k=7,
            only_need_context=True,
            response_type="Bullet Points",
        )

        self.assertEqual(
            mod._query_kwargs(args),
            {
                "top_k": 5,
                "chunk_top_k": 7,
                "only_need_context": True,
                "response_type": "Bullet Points",
                "vlm_enhanced": False,
            },
        )

        payload = mod._query_payload(args, "retrieved context")
        self.assertEqual(payload["result"], "retrieved context")
        self.assertEqual(payload["contexts"][0]["text"], "retrieved context")
        self.assertEqual(payload["contexts"][0]["metadata"]["top_k"], 5)
        self.assertEqual(payload["contexts"][0]["metadata"]["chunk_top_k"], 7)

    def test_raganything_query_ready_initializes_lightrag(self) -> None:
        mod = _load_script("query_raganything_index.py")

        class FakeRag:
            called = False

            async def _ensure_lightrag_initialized(self):
                self.called = True
                return {"success": True}

        rag = FakeRag()
        asyncio.run(mod._ensure_query_ready(rag))
        self.assertTrue(rag.called)

        class BrokenRag:
            async def _ensure_lightrag_initialized(self):
                return {"success": False, "error": "missing index"}

        with self.assertRaisesRegex(RuntimeError, "missing index"):
            asyncio.run(mod._ensure_query_ready(BrokenRag()))

    def test_graphrag_index_files_ignore_prepared_input(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            (working_dir / "settings.yaml").write_text("settings", encoding="utf-8")
            (working_dir / "input").mkdir()
            (working_dir / "input" / "mutcd_chunks.txt").write_text("prepared", encoding="utf-8")
            self.assertEqual(mod._index_files(working_dir), [])

            output = working_dir / "output" / "artifacts"
            output.mkdir(parents=True)
            (output / "create_final_nodes.parquet").write_text("indexed", encoding="utf-8")
            self.assertEqual(mod._index_files(working_dir), ["output/artifacts/create_final_nodes.parquet"])

    def test_graphrag_json_contexts_are_harness_native_and_capped(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        args = argparse.Namespace(
            question="What does Section 2A.04 require?",
            method="local",
            top_k=2,
            response_type="Multiple Paragraphs",
            community_level=2,
            dynamic_community_selection=False,
        )
        stdout = json.dumps(
            {
                "response": "GraphRAG answer",
                "context_data": {
                    "sources": [
                        {
                            "id": "source-1",
                            "text": "source text",
                            "section_id": "2A.04",
                            "rank": 3,
                        }
                    ],
                    "reports": [
                        {
                            "title": "Warning Signs",
                            "content": "report content",
                            "rank": 2,
                        }
                    ],
                    "relationships": [
                        {
                            "source": "A",
                            "target": "B",
                            "description": "relationship text",
                        }
                    ],
                },
            }
        )

        payload = mod._query_payload_from_stdout(args, stdout)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["result"], "GraphRAG answer")
        self.assertEqual(len(payload["contexts"]), 2)
        self.assertEqual(payload["contexts"][0]["name"], "source-1")
        self.assertEqual(payload["contexts"][0]["kind"], "chunk")
        self.assertEqual(payload["contexts"][0]["text"], "source text")
        self.assertEqual(payload["contexts"][0]["score"], 3.0)
        self.assertEqual(payload["contexts"][0]["metadata"]["graph_group"], "sources")
        self.assertIn("Warning Signs", payload["contexts"][1]["text"])
        self.assertEqual(payload["contexts"][1]["metadata"]["graph_group"], "reports")
        self.assertEqual(mod._contexts_from_graphrag_data(json.loads(stdout)["context_data"], top_k=0, method="local"), [])

    def test_paperqa_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            index = root / "docs.pkl"
            args = argparse.Namespace(
                repo=repo,
                index=index,
                chunks=root / "chunks.jsonl",
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                index.write_bytes(b"pickle")
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

    def test_paperqa_query_budget_and_context_records_are_harness_native(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        settings = SimpleNamespace(answer=SimpleNamespace(evidence_k=10, answer_max_sources=5))
        mod._apply_query_budget(settings, argparse.Namespace(top_k=7))
        self.assertEqual(settings.answer.evidence_k, 7)
        self.assertEqual(settings.answer.answer_max_sources, 7)

        source = SimpleNamespace(
            text="original source chunk",
            name="MUTCD11e_2A04_Standard_13",
            doc=SimpleNamespace(docname="MUTCD 11th Edition", dockey="mutcd11e", citation="MUTCD"),
            model_extra={"section_id": "2A.04", "content_type": "Standard", "ordinal": 13, "page_printed": "31"},
        )
        context = SimpleNamespace(id="pqac-test", context="relevant summary", text=source, score=8)
        record = mod._paperqa_context_to_record(context)

        self.assertEqual(record["name"], "pqac-test")
        self.assertEqual(record["kind"], "chunk")
        self.assertIn("relevant summary", record["text"])
        self.assertIn("original source chunk", record["text"])
        self.assertEqual(record["score"], 8)
        self.assertEqual(record["metadata"]["source_name"], "MUTCD11e_2A04_Standard_13")
        self.assertEqual(record["metadata"]["section_id"], "2A.04")
        self.assertEqual(record["metadata"]["docname"], "MUTCD 11th Edition")

    def test_hipporag_sidecar_enriches_contexts_without_counting_as_index(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_dir = root / "hipporag"
            chunks = root / "chunks.jsonl"
            row = {
                "doc_id": "MUTCD11e_2A04_Standard_13",
                "title": "Section 2A.04 Standard 13 - General",
                "text": "chunk text",
                "metadata": {"section_id": "2A.04", "page_printed": "31", "content_type": "Standard"},
            }
            chunks.write_text(json.dumps(row) + "\n", encoding="utf-8")

            sidecar = mod._write_metadata_sidecar(save_dir, [row])
            self.assertTrue(sidecar.exists())
            self.assertEqual(mod._index_files(save_dir), [])

            (save_dir / "hipporag_index.bin").write_text("indexed", encoding="utf-8")
            self.assertEqual(mod._index_files(save_dir), ["hipporag_index.bin"])

            manifest = mod._load_metadata_by_text(save_dir, chunks)
            context = mod._context_from_hit("chunk text", 0.75, 1, manifest)
            self.assertEqual(context["name"], "MUTCD11e_2A04_Standard_13")
            self.assertEqual(context["kind"], "chunk")
            self.assertEqual(context["score"], 0.75)
            self.assertEqual(context["metadata"]["section_id"], "2A.04")
            self.assertEqual(context["metadata"]["title"], "Section 2A.04 Standard 13 - General")

    def test_vector_db_check_is_runnable_before_lazy_index_exists(self) -> None:
        mod = _load_script("query_vector_db.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir(parents=True)
            chunks = cache / "chunks.jsonl"
            chunks.write_text("", encoding="utf-8")
            args = argparse.Namespace(mrag_dir=mrag_dir, path=root / "qdrant_hash_vector")

            with patch.object(mod.importlib.util, "find_spec", return_value=object()):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertTrue(report["runnable"])
                self.assertFalse(report["index_ready"])

                chunks.unlink()
                report = mod._dependency_report(args)
                self.assertFalse(report["environment_ready"])
                self.assertFalse(report["runnable"])

    def test_heavy_adapters_report_isolated_python_paths(self) -> None:
        for script, env_name in [
            ("query_mrag_reference.py", "mrag-reference"),
            ("query_hipporag_index.py", "hipporag"),
            ("query_visrag_index.py", "visrag"),
        ]:
            mod = _load_script(script)
            args = argparse.Namespace(
                python=ROOT / "data" / "working" / "venvs" / env_name / "bin" / "python",
                repo=ROOT / "external",
                save_dir=ROOT / "data" / "working" / "hipporag_index",
                mrag_dir=ROOT / "data",
                manifest=ROOT / "missing-manifest.jsonl",
                embeddings=ROOT / "missing-embeddings.npy",
                model_name_or_path="local-model",
            )
            report = mod._dependency_report(args)
            self.assertEqual(report["adapter_python"], str(args.python))
            self.assertIn("adapter_python_found", report)
            self.assertIn("current_python", report)

    def test_heavy_adapter_reexec_is_noop_when_python_missing(self) -> None:
        mod = _load_script("query_visrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(mod._maybe_reexec(Path(td) / "missing-python"))


if __name__ == "__main__":
    unittest.main()
