from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import tempfile
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
