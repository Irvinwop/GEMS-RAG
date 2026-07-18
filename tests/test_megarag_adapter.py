from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "query_megarag_index.py"
    spec = importlib.util.spec_from_file_location("query_megarag_index", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMegaRAGAdapter(unittest.TestCase):
    def test_embedding_runtime_defaults_are_memory_bounded(self) -> None:
        mod = _load_script()
        with patch.object(sys, "argv", ["query_megarag_index.py", "check"]):
            args = mod._parse_args()

        self.assertEqual(args.embedding_batch_size, 1)
        self.assertEqual(args.embedding_max_async, 1)

    def test_embedding_runtime_limits_must_be_positive(self) -> None:
        mod = _load_script()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "query_megarag_index.py",
                    "--embedding-batch-size",
                    "0",
                    "check",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            mod._parse_args()

    def test_completion_model_routes_image_calls_to_vision_model(self) -> None:
        mod = _load_script()
        args = SimpleNamespace(llm_model="qwen3:8b", vision_model="qwen2.5vl:3b")

        self.assertEqual(mod._completion_model(args, None), "qwen3:8b")
        self.assertEqual(mod._completion_model(args, []), "qwen3:8b")
        self.assertEqual(mod._completion_model(args, ["page.png"]), "qwen2.5vl:3b")

    def test_prepare_gives_blank_pages_unique_text_for_upstream_chunk_hashes(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            pages = mrag_dir / "page_images"
            cache.mkdir(parents=True)
            pages.mkdir()
            (cache / "chunks.jsonl").write_text("", encoding="utf-8")
            (cache / "figures.jsonl").write_text("", encoding="utf-8")
            (pages / "page_0001.png").write_bytes(b"one")
            (pages / "page_0002.png").write_bytes(b"two")
            out = root / "pages_content.json"

            mod.prepare_pages_content(mrag_dir, out)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(payload["0"]["text"], "[PDF Page: 1]")
        self.assertEqual(payload["1"]["text"], "[PDF Page: 2]")
        self.assertNotEqual(payload["0"]["text"], payload["1"]["text"])

    def test_prepare_can_start_at_a_substantive_pdf_page(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            pages = mrag_dir / "page_images"
            cache.mkdir(parents=True)
            pages.mkdir()
            (pages / "page_0001.png").write_bytes(b"cover")
            (pages / "page_0042.png").write_bytes(b"purpose")
            (cache / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "purpose",
                        "page_pdf": 42,
                        "text": "The purpose of the MUTCD is national uniformity.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (cache / "figures.jsonl").write_text("", encoding="utf-8")
            out = root / "pages_content.json"

            report = mod.prepare_pages_content(
                mrag_dir,
                out,
                start_page=42,
                limit=1,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse((out.parent / ".pages_content.json.tmp").exists())

        self.assertEqual(report["start_page"], 42)
        self.assertEqual(report["limit"], 1)
        self.assertEqual(payload["0"]["page_pdf"], 42)
        self.assertIn("purpose of the MUTCD", payload["0"]["text"])

    def test_empty_graph_patch_skips_asyncio_wait_on_an_empty_task_set(self) -> None:
        patch_text = (ROOT / "patches" / "megarag-empty-graph.patch").read_text(
            encoding="utf-8"
        )

        self.assertIn("+    if not tasks:", patch_text)
        self.assertIn("+        return", patch_text)
        self.assertIn("await asyncio.wait(tasks", patch_text)
        self.assertIn("+    chunk_results_at_stage_one = {}", patch_text)
        self.assertIn("+        stage_one = chunk_results_at_stage_one.get(", patch_text)
        self.assertEqual(patch_text.count("+                                        raise"), 2)
        self.assertIn("+                                raise", patch_text)
        self.assertIn("+            page_image_paths =", patch_text)
        self.assertIn(
            "+                    embeddings = await self.embedding_func(images=images)",
            patch_text,
        )
        self.assertNotIn("+        embeddings_list = await asyncio.gather", patch_text)
        self.assertIn("+    def save(self):", patch_text)
        self.assertIn("+            os.replace(tmp_path, self.storage_file)", patch_text)
        self.assertIn("+                if not await self.index_done_callback():", patch_text)
        self.assertIn("+        embeddings = embed_model.embed(", patch_text)
        self.assertIn("+            texts=[None] * len(images),", patch_text)
        addon_text = (ROOT / "configs" / "megarag-addon-params.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("embedding_checkpoint_interval: 8", addon_text)

    def test_smoke_scope_cannot_satisfy_a_full_index_check(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "megarag"
            lightrag_repo = root / "lightrag"
            (repo / "megarag").mkdir(parents=True)
            (lightrag_repo / "lightrag").mkdir(parents=True)
            (repo / "megarag" / "megarag.py").write_text("", encoding="utf-8")
            (lightrag_repo / "lightrag" / "base.py").write_text("", encoding="utf-8")
            pages_content = root / "pages.json"
            pages_content.write_text("{}\n", encoding="utf-8")
            addon_config = root / "addon.yaml"
            addon_config.write_text("addon_params: {}\n", encoding="utf-8")
            working_dir = root / "index"
            working_dir.mkdir()
            for name in mod.CORE_INDEX_FILES:
                (working_dir / name).write_text("indexed", encoding="utf-8")
            args = SimpleNamespace(
                repo=repo,
                lightrag_repo=lightrag_repo,
                pages_content=pages_content,
                addon_config=addon_config,
                working_dir=working_dir,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
                embedding_model="gme",
                llm_model="chat",
                vision_model="vision",
                reasoning_effort=None,
                llm_max_tokens=2048,
                start_page=42,
                limit=1,
                python=root / "python",
            )
            mod._write_json_atomic(
                working_dir / mod.INDEX_SENTINEL,
                mod._index_identity(args),
            )
            endpoint = {
                "checked": False,
                "reachable": True,
                "usable": True,
            }

            with (
                patch.object(mod, "_import_errors", return_value={}),
                patch.object(mod, "probe_openai_endpoint", return_value=endpoint),
            ):
                self.assertTrue(mod._dependency_report(args)["index_ready"])
                args.limit = None
                self.assertFalse(mod._dependency_report(args)["index_ready"])
                args.limit = 1
                args.start_page = None
                self.assertFalse(mod._dependency_report(args)["index_ready"])

                args.start_page = 42
                prior_document_id = mod._document_id(args)
                pages_content.write_text('{"changed": true}\n', encoding="utf-8")
                self.assertNotEqual(mod._document_id(args), prior_document_id)
                self.assertFalse(mod._dependency_report(args)["index_ready"])

    def test_changed_input_requires_force_and_attempt_identity_is_stable(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            working_dir = root / "index"
            working_dir.mkdir()
            (working_dir / "partial.json").write_text("{}\n", encoding="utf-8")
            pages_content = root / "pages.json"
            pages_content.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                working_dir=working_dir,
                pages_content=pages_content,
                mrag_dir=root,
                force=False,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
                embedding_model="gme",
                llm_model="chat",
                vision_model="vision",
                reasoning_effort=None,
                llm_max_tokens=2048,
                start_page=42,
                limit=1,
            )
            report = {
                "environment_ready": True,
                "api_key_usable": True,
                "input_ready": True,
                "index_ready": False,
            }
            mod._write_json_atomic(
                working_dir / mod.INDEX_ATTEMPT,
                {"different": "input"},
            )

            with (
                patch.object(mod, "_dependency_report", return_value=report),
                patch.object(mod, "_initialize_rag") as initialize,
                patch("builtins.print"),
            ):
                self.assertEqual(asyncio.run(mod._index(args)), 2)
                initialize.assert_not_called()

            mod._write_json_atomic(
                working_dir / mod.INDEX_ATTEMPT,
                mod._index_identity(args),
            )
            self.assertEqual(
                json.loads((working_dir / mod.INDEX_ATTEMPT).read_text(encoding="utf-8")),
                mod._index_identity(args),
            )

    def test_index_checks_the_manifest_specific_document_status(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pages_content = root / "pages.json"
            pages_content.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                working_dir=root / "index",
                pages_content=pages_content,
                mrag_dir=root,
                force=False,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
                embedding_model="gme",
                llm_model="chat",
                vision_model="vision",
                reasoning_effort=None,
                llm_max_tokens=2048,
                start_page=42,
                limit=1,
            )

            class Status:
                async def get_by_id(self, doc_id):
                    self.doc_id = doc_id
                    return {"status": "processed"}

            class Rag:
                doc_status = Status()

                async def ainsert(self, **kwargs):
                    self.insert_args = kwargs

                async def finalize_storages(self):
                    return None

            rag = Rag()
            initial = {
                "environment_ready": True,
                "api_key_usable": True,
                "input_ready": True,
                "index_ready": False,
            }
            final = {"index_ready": True}
            with (
                patch.object(mod, "_dependency_report", side_effect=[initial, final]),
                patch.object(mod, "_initialize_rag", return_value=(rag, "tokens")),
                patch.object(mod, "_api_key", return_value="local"),
                patch("builtins.print"),
            ):
                self.assertEqual(asyncio.run(mod._index(args)), 0)

            document_id = mod._document_id(args)
            self.assertEqual(rag.insert_args["ids"], document_id)
            self.assertEqual(rag.doc_status.doc_id, document_id)
            sentinel = json.loads(
                (args.working_dir / mod.INDEX_SENTINEL).read_text(encoding="utf-8")
            )
            self.assertEqual(sentinel["document_id"], document_id)

    def test_prepare_uses_existing_pages_figures_and_canonical_chunks(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            pages = mrag_dir / "page_images"
            figures = mrag_dir / "figures"
            cache.mkdir(parents=True)
            pages.mkdir()
            figures.mkdir()
            (pages / "page_0001.png").write_bytes(b"page")
            (figures / "figure_1A-1_p0001.png").write_bytes(b"figure")
            chunks = [
                {
                    "chunk_id": "same",
                    "page_pdf": 1,
                    "section_id": "1A.01",
                    "section_title": "Purpose",
                    "content_type": "Support",
                    "ordinal": 1,
                    "text": "The complete canonical provision applies.",
                },
                {"chunk_id": "same", "page_pdf": 1, "text": "4.5"},
            ]
            (cache / "chunks.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in chunks),
                encoding="utf-8",
            )
            (cache / "figures.jsonl").write_text(
                json.dumps(
                    {
                        "figure_id": "Figure 1A-1",
                        "page_pdf": 1,
                        "image_path": "/content/drive/MRAG/figures/figure_1A-1_p0001.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "pages_content.json"

            report = mod.prepare_pages_content(mrag_dir, out)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["pages"], 1)
        self.assertEqual(report["chunks"], 1)
        self.assertEqual(report["figure_images"], 1)
        self.assertTrue(payload["0"]["text"].startswith("[PDF Page: 1]"))
        self.assertIn("[Chunk ID: same]", payload["0"]["text"])
        self.assertIn("complete canonical provision", payload["0"]["text"])
        self.assertNotIn("4.5", payload["0"]["text"])
        self.assertTrue(payload["0"]["page_image"].endswith("page_0001.png"))
        self.assertTrue(payload["0"]["figure_images"][0].endswith("figure_1A-1_p0001.png"))

    def test_dual_retrieval_preserves_context_only_on_both_official_branches(self) -> None:
        mod = _load_script()

        class FakeRag:
            def __init__(self) -> None:
                self.calls = []

            async def aquery(self, question, param):
                self.calls.append((question, param))
                return f"{param.mode} context"

        rag = FakeRag()
        with patch.object(
            mod,
            "_query_param_class",
            return_value=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            kg_context, page_context = asyncio.run(mod.retrieve_dual_context(rag, "Question?", top_k=7))

        self.assertEqual({param.mode for _question, param in rag.calls}, {"hybrid", "naive"})
        self.assertTrue(all(param.only_need_context for _question, param in rag.calls))
        self.assertTrue(all(param.chunk_top_k == 7 for _question, param in rag.calls))
        self.assertEqual(kg_context, "hybrid context")
        self.assertEqual(page_context, "naive context")

    def test_query_payload_recovers_chunk_and_page_evidence(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            pages = mrag_dir / "page_images"
            cache.mkdir(parents=True)
            pages.mkdir()
            (pages / "page_0001.png").write_bytes(b"page")
            chunk = {
                "chunk_id": "chunk-1",
                "page_pdf": 1,
                "section_id": "1A.01",
                "section_title": "Purpose",
                "content_type": "Support",
                "ordinal": 1,
                "text": "Canonical text.",
            }
            (cache / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
            context = """-----Document Chunks(DC)-----
[Chunk ID: chunk-1]
Canonical text.
-----Page Images (PI)-----
{\"image_0\": \"filename:page_0001.png\"}
"""

            payload = mod.query_payload("Question?", context, "page branch", mrag_dir, top_k=3)

        self.assertEqual(payload["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(payload["pages"][0]["page_pdf"], 1)
        self.assertTrue(payload["pages"][0]["image_path"].endswith("page_0001.png"))
        self.assertEqual(payload["contexts"][0]["metadata"]["upstream_second_generation_bypassed"], True)


if __name__ == "__main__":
    unittest.main()
