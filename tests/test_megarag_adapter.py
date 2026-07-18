from __future__ import annotations

import asyncio
import importlib.util
import json
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
