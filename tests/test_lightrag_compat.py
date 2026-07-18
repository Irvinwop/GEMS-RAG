from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from gems_rag.lightrag_compat import (
    cap_completion_tokens,
    lightrag_document_status_report,
)


class TestLightRagCompat(unittest.TestCase):
    def test_completion_cap_preserves_smaller_caller_budget(self) -> None:
        empty: dict[str, int] = {}
        smaller = {"max_tokens": 512}
        larger = {"max_completion_tokens": 4096}

        cap_completion_tokens(empty, 2048)
        cap_completion_tokens(smaller, 2048)
        cap_completion_tokens(larger, 2048)

        self.assertEqual(empty, {"max_tokens": 2048})
        self.assertEqual(smaller, {"max_tokens": 512})
        self.assertEqual(larger, {"max_completion_tokens": 2048})

    def test_status_report_rejects_failed_or_missing_documents(self) -> None:
        class CountStorage:
            async def get_status_counts(self):
                return {"processed": 1, "failed": 1, "pending": 0}

        class DocumentStorage:
            async def get_by_id(self, doc_id):
                return {"status": "processed"} if doc_id == "ok" else None

        failed = asyncio.run(
            lightrag_document_status_report(SimpleNamespace(doc_status=CountStorage()))
        )
        missing = asyncio.run(
            lightrag_document_status_report(
                SimpleNamespace(doc_status=DocumentStorage()),
                doc_ids=["ok", "missing"],
            )
        )

        self.assertFalse(failed["complete"])
        self.assertEqual(failed["status_counts"], {"failed": 1, "processed": 1})
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["status_counts"], {"missing": 1, "processed": 1})


if __name__ == "__main__":
    unittest.main()
