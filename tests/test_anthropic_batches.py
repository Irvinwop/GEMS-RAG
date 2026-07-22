from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag.anthropic_batches import retry_anthropic_batch_failure, run_anthropic_batch
from gems_rag.config import DatasetConfig, ExperimentConfig, ModelConfig, RetrieverConfig
from gems_rag.retrieval import Retriever
from gems_rag.retrieval_snapshots import build_retrieval_snapshot
from gems_rag.types import Evidence, QAItem, RetrievalResult


class _FixtureRetriever(Retriever):
    def retrieve(self, item: QAItem) -> RetrievalResult:
        return RetrievalResult(
            adapter="bm25",
            query=item.question,
            evidence=[
                Evidence(
                    evidence_id=f"{item.qa_id}-hit",
                    kind="chunk",
                    text=f"Frozen evidence for {item.qa_id}",
                    metadata={"section_id": "1A.01"},
                    score=1.0,
                )
            ],
            debug={"fixture": True},
        )


class _FakeBatchTransport:
    def __init__(self) -> None:
        self.requests = []
        self.create_calls = 0

    def create(self, requests):
        self.create_calls += 1
        self.requests = requests
        return {
            "id": "msgbatch_fixture",
            "processing_status": "in_progress",
            "request_counts": {"processing": len(requests), "succeeded": 0},
        }

    def retrieve(self, batch_id):
        return {
            "id": batch_id,
            "processing_status": "ended",
            "request_counts": {"processing": 0, "succeeded": len(self.requests)},
            "ended_at": "2026-07-21T00:00:00Z",
        }

    def results(self, batch_id):
        return [
            {
                "custom_id": request["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": f"msg_{index}",
                        "model": request["params"]["model"],
                        "content": [{"type": "text", "text": f"Answer {index}"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    },
                },
            }
            for index, request in enumerate(self.requests)
        ]


class _RetryBatchTransport(_FakeBatchTransport):
    def __init__(self) -> None:
        super().__init__()
        self.message_calls = 0

    def results(self, batch_id):
        rows = super().results(batch_id)
        rows[0]["result"] = {
            "type": "errored",
            "error": {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Please retry.",
                },
            },
        }
        return rows

    def create_message(self, params):
        self.message_calls += 1
        return {
            "id": "msg_retry",
            "type": "message",
            "model": params["model"],
            "content": [{"type": "text", "text": "Recovered answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 21, "output_tokens": 6},
        }


def _config(root: Path) -> ExperimentConfig:
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
    (cache / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "section_id": "1A.01",
                "section_title": "Purpose",
                "content_type": "Support",
                "ordinal": 1,
                "page_printed": "1",
                "part": "Part 1",
                "text": "Fixture MUTCD evidence.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ExperimentConfig(
        name="anthropic-batch-fixture",
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
        models=[
            ModelConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                options={
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "max_tokens": 128000,
                    "temperature": None,
                    "thinking": "disabled",
                },
            )
        ],
        retrieval_snapshot=root / "retrieval.jsonl",
        output_dir=root / "runs",
    )


class TestAnthropicBatches(unittest.TestCase):
    def test_batch_is_persisted_replayed_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _config(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever(),
            ):
                build_retrieval_snapshot(config)
            transport = _FakeBatchTransport()
            submitted = run_anthropic_batch(
                config,
                poll_interval_s=0,
                wait=False,
                transport=transport,
                sleep=lambda _: None,
            )
            runs_path = config.output_dir / config.name / "runs.jsonl"
            self.assertFalse(runs_path.exists())
            resumed = run_anthropic_batch(config, transport=transport)
            completed = run_anthropic_batch(config, transport=transport)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            state = json.loads(
                (config.output_dir / config.name / "anthropic_batch_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(submitted["status"], "in_progress")
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(transport.create_calls, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["run_status"] == "successful" for row in rows))
        self.assertTrue(all(row["model_raw"]["api"] == "anthropic_batch" for row in rows))
        self.assertTrue(all(row["model_raw"]["finish_reason"] == "stop" for row in rows))
        self.assertTrue(all(row["retrieval_debug"]["snapshot_reused"] for row in rows))
        self.assertEqual(state["status"], "complete")
        self.assertNotIn("temperature", transport.requests[0]["params"])
        self.assertEqual(transport.requests[0]["params"]["thinking"], {"type": "disabled"})
        self.assertEqual(transport.requests[0]["params"]["max_tokens"], 128000)

    def test_failed_request_can_be_retried_once_without_rewriting_batch_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = _config(Path(td))
            with patch(
                "gems_rag.retrieval_snapshots.build_retriever",
                return_value=_FixtureRetriever(),
            ):
                build_retrieval_snapshot(config)
            transport = _RetryBatchTransport()
            run_anthropic_batch(
                config,
                poll_interval_s=0,
                wait=False,
                transport=transport,
                sleep=lambda _: None,
            )
            custom_id = transport.requests[0]["custom_id"]
            with self.assertRaisesRegex(RuntimeError, "failures"):
                run_anthropic_batch(config, transport=transport)

            retry = retry_anthropic_batch_failure(config, custom_id, transport=transport)
            completed = run_anthropic_batch(config, transport=transport)
            repeated = retry_anthropic_batch_failure(config, custom_id, transport=transport)
            output_dir = config.output_dir / config.name
            rows = [
                json.loads(line)
                for line in (output_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            raw_results = [
                json.loads(line)
                for line in (output_dir / "anthropic_batch_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        recovered = next(row for row in rows if row["model_raw"]["api"] == "anthropic_sync_retry")
        self.assertEqual(retry["status"], "retried")
        self.assertEqual(repeated["status"], "already_retried")
        self.assertEqual(transport.message_calls, 1)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["retry_count"], 1)
        self.assertEqual(completed["usage"]["input_tokens"], 41)
        self.assertEqual(completed["usage"]["output_tokens"], 11)
        self.assertEqual(recovered["answer"], "Recovered answer")
        self.assertEqual(recovered["model_raw"]["recovery"]["original_result"]["type"], "errored")
        self.assertEqual(raw_results[0]["result"]["type"], "errored")
