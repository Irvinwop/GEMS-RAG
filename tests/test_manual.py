from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag.manual import ingestion_matrix, manual_status, write_manual_manifest


class TestManualStatus(unittest.TestCase):
    def test_status_verifies_pdf_derivatives_and_native_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag = root / "MRAG"
            cache = mrag / "mmrag_cache_v3"
            pages = mrag / "page_images"
            figures = mrag / "figures"
            shared = root / "data" / "working" / "mrag_corpus"
            for path in [cache, pages, figures, shared, mrag / "eval", root / "configs"]:
                path.mkdir(parents=True, exist_ok=True)
            (mrag / "mutcd11theditionr1hl.pdf").write_bytes(b"%PDF-fixture")
            (cache / "chunks.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (cache / "figures.jsonl").write_text("{}\n", encoding="utf-8")
            (cache / "graph.gpickle").write_bytes(b"graph")
            (mrag / "eval" / "gold_qa.jsonl").write_text("{}\n", encoding="utf-8")
            (pages / "page_0001.png").write_bytes(b"png")
            (figures / "figure.png").write_bytes(b"png")
            (shared / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
            (shared / "manifest.json").write_text(json.dumps({"chunk_canonicalization": {"raw_rows": 2}}), encoding="utf-8")
            (root / "configs" / "manuscript-rags.json").write_text(
                json.dumps({"entries": [{"method_id": "paperqa", "label": "PaperQA2", "retrievers": ["paper"]}]}),
                encoding="utf-8",
            )
            (root / "configs" / "retriever-catalog.example.json").write_text(
                json.dumps({"retrievers": [{"name": "paper", "family": "paperqa2"}]}), encoding="utf-8"
            )

            with patch("gems_rag.manual._pdf_info", return_value={"Pages": "1", "Title": "Fixture"}):
                report = manual_status(root=root, mrag_dir=mrag)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["artifacts"]["raw_chunks"], 2)
        self.assertEqual(report["ingestion"]["native_method_count"], 1)
        self.assertEqual(report["ingestion"]["methods"][0]["retrievers"][0]["native"]["mode"], "native_pdf")

    def test_missing_artifacts_are_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "configs").mkdir()
            (root / "configs" / "manuscript-rags.json").write_text('{"entries": []}', encoding="utf-8")
            (root / "configs" / "retriever-catalog.example.json").write_text('{"retrievers": []}', encoding="utf-8")
            report = manual_status(root=root, mrag_dir=root / "missing")

        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["checks"][0]["ok"])

    def test_manifest_writer_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "manifest.json"
            write_manual_manifest(path, {"status": "ready"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "ready")

    def test_repository_matrix_covers_all_manuscript_methods(self) -> None:
        matrix = ingestion_matrix()
        self.assertEqual(matrix["method_count"], 19)
        self.assertGreaterEqual(matrix["native_method_count"], 3)
        self.assertIn("raganything", matrix["native_families"])
        self.assertTrue(all(row["manual_lineage"] == "verified" for row in matrix["methods"]))
