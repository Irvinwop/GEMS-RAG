from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from gem_rags.manuscript_rags import (
    REQUIRED_MANUSCRIPT_METHOD_IDS,
    load_manuscript_rag_catalog,
    validate_manuscript_rag_coverage,
)
from gem_rags.retriever_catalog import load_retriever_catalog


ROOT = Path(__file__).resolve().parents[1]


class TestManuscriptRagCatalog(unittest.TestCase):
    def test_catalog_records_every_manuscript_system_and_baseline(self) -> None:
        catalog = load_manuscript_rag_catalog(ROOT / "configs" / "manuscript-rags.json")

        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(
            {entry.method_id for entry in catalog.entries if entry.coverage_required},
            REQUIRED_MANUSCRIPT_METHOD_IDS,
        )

    def test_every_required_method_has_an_enabled_catalog_retriever(self) -> None:
        catalog = load_manuscript_rag_catalog(ROOT / "configs" / "manuscript-rags.json")
        retrievers = load_retriever_catalog(ROOT / "configs" / "retriever-catalog.example.json")

        report = validate_manuscript_rag_coverage(catalog, retrievers)

        self.assertTrue(report["ok"])
        self.assertEqual(report["required_method_count"], 19)
        self.assertEqual(report["integrated_method_count"], 19)
        self.assertGreater(report["referenced_retriever_count"], 19)
        self.assertEqual(report["problems"], [])

    def test_coverage_report_fails_for_pending_or_missing_retrievers(self) -> None:
        catalog = load_manuscript_rag_catalog(ROOT / "configs" / "manuscript-rags.json")
        retrievers = load_retriever_catalog(ROOT / "configs" / "retriever-catalog.example.json")
        entries = tuple(
            replace(entry, integration_status="acquired_adapter_pending")
            if entry.method_id == "megarag"
            else entry
            for entry in catalog.entries
        )
        broken_catalog = replace(catalog, entries=entries)
        broken_retrievers = [
            entry for entry in retrievers if entry.config.name != "megarag_hybrid_context"
        ]

        report = validate_manuscript_rag_coverage(broken_catalog, broken_retrievers)

        self.assertFalse(report["ok"])
        self.assertEqual(report["pending_method_ids"], ["megarag"])
        self.assertEqual(report["missing_retriever_names"], ["megarag_hybrid_context"])


if __name__ == "__main__":
    unittest.main()
