from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from gems_rag.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RetrieverConfig,
    load_experiment_config,
    write_experiment_config,
)
from gems_rag.retrieval import Retriever
from gems_rag.retrieval_snapshots import (
    build_retrieval_snapshot,
    load_retrieval_snapshot,
    retrieval_snapshot_status,
)
from gems_rag.runner import run_experiment
from gems_rag.types import Evidence, QAItem, RetrievalResult


class _FixtureRetriever(Retriever):
    def __init__(self, name: str, *, error: str | None = None) -> None:
        self.name = name
        self.error = error
        self.calls = 0

    def retrieve(self, item: QAItem) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=(
                []
                if self.error
                else [
                    Evidence(
                        evidence_id=f"{item.qa_id}-hit",
                        kind="chunk",
                        text=f"Frozen evidence for {item.qa_id}",
                        metadata={"section_id": "1A.01"},
                        score=1.0,
                    )
                ]
            ),
            debug={"fixture": True, "call": self.calls},
            error=self.error,
        )


def _fixture(root: Path) -> ExperimentConfig:
    qa_path = root / "questions.jsonl"
    qa_path.write_text(
        "".join(
            json.dumps({"question_id": f"Q{index}", "question": f"Question {index}?"}) + "\n"
            for index in range(1, 3)
        ),
        encoding="utf-8",
    )
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    chunk = {
        "chunk_id": "chunk-1",
        "section_id": "1A.01",
        "section_title": "Purpose",
        "content_type": "Support",
        "ordinal": 1,
        "page_printed": "1",
        "part": "Part 1",
        "text": "Fixture MUTCD evidence.",
    }
    (cache / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    return ExperimentConfig(
        name="snapshot-fixture",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=2),
        retrievers=[
            RetrieverConfig(
                name="bm25",
                kind="bm25",
                top_k=1,
                context_modes=("injected",),
            )
        ],
        context_modes=["injected"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
        retrieval_snapshot=root / "retrieval.jsonl",
        output_dir=root / "runs",
        dry_run=True,
    )


class TestRetrievalSnapshots(unittest.TestCase):
    def test_build_resume_and_load_reuse_exact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            retriever = _FixtureRetriever("bm25")
            with patch("gems_rag.retrieval_snapshots.build_retriever", return_value=retriever):
                first = build_retrieval_snapshot(config)
            before = hashlib.sha256(config.retrieval_snapshot.read_bytes()).hexdigest()
            with patch("gems_rag.retrieval_snapshots.build_retriever") as live_build:
                second = build_retrieval_snapshot(config)
            after = hashlib.sha256(config.retrieval_snapshot.read_bytes()).hexdigest()
            snapshot = load_retrieval_snapshot(config)
            result = snapshot.retriever(config.retrievers[0]).retrieve(
                QAItem("Q1", "Question 1?", None, False, {}, [])
            )

        self.assertTrue(first["ok"])
        self.assertEqual(first["rows_written"], 2)
        self.assertEqual(second["rows_written"], 0)
        self.assertEqual(second["rows_skipped"], 2)
        self.assertEqual(before, after)
        live_build.assert_not_called()
        self.assertEqual(result.evidence[0].text, "Frozen evidence for Q1")
        self.assertTrue(result.debug["snapshot_reused"])

    def test_runner_uses_snapshot_without_live_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                build_retrieval_snapshot(config)
            with patch("gems_rag.runner.build_retriever") as live_build:
                runs_path = run_experiment(config, overwrite=True)
            rows = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
            ]
            snapshot_sha256 = hashlib.sha256(config.retrieval_snapshot.read_bytes()).hexdigest()

        live_build.assert_not_called()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["retrieval_debug"]["snapshot_reused"] for row in rows))
        self.assertEqual(
            {row["retrieval_debug"]["snapshot_sha256"] for row in rows},
            {snapshot_sha256},
        )

    def test_retry_errors_archives_and_replaces_only_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25", error="temporary failure"),
            ):
                failed = build_retrieval_snapshot(config)
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                repaired = build_retrieval_snapshot(config, retry_errors=True)
            archive = Path(repaired["retry_archive"])
            archive_exists = archive.is_file()
            archive_rows = len(archive.read_text(encoding="utf-8").splitlines())

        self.assertFalse(failed["ok"])
        self.assertEqual(failed["error_rows"], 2)
        self.assertTrue(repaired["ok"])
        self.assertEqual(repaired["rows_written"], 2)
        self.assertTrue(archive_exists)
        self.assertEqual(archive_rows, 2)

    def test_resume_repairs_a_truncated_final_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                build_retrieval_snapshot(config)
            with config.retrieval_snapshot.open("ab") as handle:
                handle.write(b'{"qa_id":"partial"')
            resumed = build_retrieval_snapshot(config)

        self.assertTrue(resumed["ok"])
        self.assertTrue(resumed["truncated_tail_repaired"])
        self.assertEqual(resumed["rows_written"], 0)

    def test_conflicting_retriever_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                build_retrieval_snapshot(config)
            conflict = ExperimentConfig(
                name=config.name,
                dataset=config.dataset,
                retrievers=[
                    RetrieverConfig(
                        name="bm25",
                        kind="bm25",
                        top_k=2,
                        context_modes=("injected",),
                    )
                ],
                context_modes=config.context_modes,
                models=config.models,
                grader=config.grader,
                rag_backend=config.rag_backend,
                retrieval_snapshot=config.retrieval_snapshot,
                output_dir=config.output_dir,
                dry_run=True,
            )

            with self.assertRaisesRegex(ValueError, "different configuration"):
                build_retrieval_snapshot(conflict)

    def test_snapshot_can_be_extended_with_another_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                build_retrieval_snapshot(config)
            paperqa = RetrieverConfig(
                name="paperqa2_chunks",
                kind="external_command",
                top_k=1,
                options={"command": ["paperqa", "query"]},
                context_modes=("injected",),
            )
            extension = replace(config, retrievers=[paperqa])
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("paperqa2_chunks"),
            ):
                report = build_retrieval_snapshot(extension)
            original_status = retrieval_snapshot_status(config)

        self.assertTrue(report["ok"])
        self.assertEqual(report["rows_written"], 2)
        self.assertEqual(report["rows_on_disk"], 4)
        self.assertTrue(original_status["ok"])

    def test_completed_snapshot_rejects_valid_json_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _fixture(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever("bm25"),
            ):
                build_retrieval_snapshot(config)
            with config.retrieval_snapshot.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"qa_id": "tampered", "retriever": "bm25"}) + "\n")
            status = retrieval_snapshot_status(config)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_retrieval_snapshot(config)

        self.assertFalse(status["ok"])
        self.assertIn("SHA-256", " ".join(status["problems"]))

    def test_missing_snapshot_is_ready_to_build_and_config_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = _fixture(root)
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            loaded = load_experiment_config(config_path)
            status = retrieval_snapshot_status(loaded)

        self.assertEqual(loaded.retrieval_snapshot, config.retrieval_snapshot)
        self.assertFalse(status["ok"])
        self.assertEqual(status["status"], "ready_to_build")


if __name__ == "__main__":
    unittest.main()
