from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.config import DatasetConfig, ExperimentConfig, RetrieverConfig
from gems_rag.rag_audit import audit_retrievers, write_rag_audit


def _dataset(root: Path, *, with_chunk: bool = True) -> tuple[Path, Path]:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    chunk = {
        "chunk_id": "chunk-1",
        "section_id": "2A.01",
        "content_type": "standard",
        "ordinal": 1,
        "section_title": "Standard Signs",
        "page_printed": 12,
        "part": "Part 2",
        "text": "A standard sign shall be used for this traffic control application.",
    }
    (cache / "chunks.jsonl").write_text(
        json.dumps(chunk) + "\n" if with_chunk else "",
        encoding="utf-8",
    )
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa-audit",
                "question": "Which standard sign shall be used?",
                "gold_answer": {"direct_answer": "Use the standard sign."},
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return mrag_dir, qa_path


class TestRagAudit(unittest.TestCase):
    def test_query_driven_rag_runs_all_four_compatible_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _dataset(root)
            config = ExperimentConfig(
                name="rag-audit",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
            )

            report = audit_retrievers(config, check_external=False)

        row = report["retrievers"][0]
        self.assertTrue(report["ok"])
        self.assertEqual(row["status"], "ready")
        self.assertEqual(
            [check["context_mode"] for check in row["context_checks"]],
            ["injected", "tool_explore", "tool_search", "tool_native"],
        )
        self.assertTrue(all(check["evidence_count"] > 0 for check in row["context_checks"]))
        self.assertEqual(report["summary"]["compatible_modes_ready"], 4)

    def test_no_retrieval_control_passes_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _dataset(root)
            config = ExperimentConfig(
                name="no-retrieval-audit",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[
                    RetrieverConfig(
                        name="no-retrieval",
                        kind="self_rag_policy",
                        options={"mode": "no_retrieval", "base_kind": "bm25"},
                        context_modes=("injected",),
                        interaction="no_retrieval",
                    )
                ],
            )

            report = audit_retrievers(config, check_external=False)

        check = report["retrievers"][0]["context_checks"][0]
        self.assertTrue(report["ok"])
        self.assertEqual(check["status"], "ready")
        self.assertEqual(check["evidence_count"], 0)

    def test_empty_query_driven_result_fails_and_report_is_writable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _dataset(root, with_chunk=False)
            config = ExperimentConfig(
                name="empty-audit",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir),
                retrievers=[
                    RetrieverConfig(
                        name="fixed",
                        kind="bm25",
                        context_modes=("injected", "tool_explore"),
                        interaction="fixed_question",
                    )
                ],
            )

            report = audit_retrievers(config, check_external=False)
            output = root / "reports" / "audit.json"
            write_rag_audit(report, output)
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(report["ok"])
        self.assertEqual(report["retrievers"][0]["status"], "failed")
        self.assertEqual(len(report["retrievers"][0]["context_checks"]), 2)
        self.assertEqual(persisted["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
