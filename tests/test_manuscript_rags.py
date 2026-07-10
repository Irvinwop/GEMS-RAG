from __future__ import annotations

import unittest
from pathlib import Path

from gem_rags.manuscript_rags import load_manuscript_rag_catalog


ROOT = Path(__file__).resolve().parents[1]


class TestManuscriptRagCatalog(unittest.TestCase):
    def test_catalog_records_every_manuscript_system_and_baseline(self) -> None:
        catalog = load_manuscript_rag_catalog(ROOT / "configs" / "manuscript-rags.json")

        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(
            {entry.method_id for entry in catalog.entries if entry.coverage_required},
            {
                "bm25",
                "canonical_rag",
                "crag",
                "dense_rag",
                "dpr",
                "gems_rag",
                "gfm_rag",
                "graphrag",
                "hybrid_rag",
                "kg2rag",
                "lpkg",
                "m3kg_rag",
                "megarag",
                "mm_rag",
                "okh_rag",
                "paperqa",
                "sam_rag",
                "self_rag",
                "visrag",
            },
        )


if __name__ == "__main__":
    unittest.main()
