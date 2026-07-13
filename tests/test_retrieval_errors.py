from __future__ import annotations

import json
import base64
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag import retrieval
from gems_rag.analysis import metric_value, summarize_rows
from gems_rag.config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from gems_rag.grading import RUBRIC_KEYS
from gems_rag.runner import run_experiment
from gems_rag.types import RetrievalResult


def _fixture_mrag(root: Path) -> tuple[Path, Path]:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    cache.mkdir(parents=True)
    (cache / "chunks.jsonl").write_text("", encoding="utf-8")
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    (cache / "graph.gpickle").write_bytes(b"placeholder")
    qa_path = mrag_dir / "eval" / "gold_qa.jsonl"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa_fail",
                "question": "What does the adapter return?",
                "gold_answer": {},
                "references": [],
                "gold_figures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return mrag_dir, qa_path


class TestRetrievalErrors(unittest.TestCase):
    def test_retriever_build_failure_is_recorded_as_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="build-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bad_retriever", kind="unknown_external")],
                context_modes=["injected", "tool_explore"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["retrieval_error"].startswith("retriever_build_failed: ValueError") for row in rows))
        self.assertTrue(all(row["config"]["retriever"] == "bad_retriever" for row in rows))
        self.assertTrue(all(row["retrieval_debug"]["retriever_build_error"] == row["retrieval_error"] for row in rows))

    def test_retriever_runtime_exception_is_recorded_and_run_continues(self) -> None:
        class RaisingRetriever:
            def retrieve(self, _item):
                raise RuntimeError("index missing")

        class EmptyRetriever:
            def retrieve(self, item):
                return RetrievalResult(adapter="ok_retriever", query=item.question, evidence=[], debug={"ok": True})

        def fake_build(config, _mrag_dir):
            return RaisingRetriever() if config.name == "bad_retriever" else EmptyRetriever()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="runtime-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(name="bad_retriever", kind="external_command"),
                    RetrieverConfig(name="ok_retriever", kind="external_command"),
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            with patch("gems_rag.runner.build_retriever", side_effect=fake_build):
                runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 2)
        by_retriever = {row["config"]["retriever"]: row for row in rows}
        self.assertEqual(by_retriever["bad_retriever"]["retrieval_error"], "retriever_failed: RuntimeError: index missing")
        self.assertIsNone(by_retriever["ok_retriever"]["retrieval_error"])

    def test_retry_errors_prunes_failed_rows_and_reruns_only_those_keys(self) -> None:
        class RaisingRetriever:
            def retrieve(self, _item):
                raise RuntimeError("index missing")

        class EmptyRetriever:
            def retrieve(self, item):
                return RetrievalResult(adapter="flaky_retriever", query=item.question, evidence=[], debug={"fixed": True})

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="retry-errors",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="flaky_retriever", kind="external_command")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            with patch("gems_rag.runner.build_retriever", return_value=RaisingRetriever()):
                runs_path = run_experiment(config, overwrite=True)
            failed_row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(failed_row["retrieval_error"], "retriever_failed: RuntimeError: index missing")

            with patch("gems_rag.runner.build_retriever", return_value=EmptyRetriever()):
                rerun_path = run_experiment(config, retry_errors=True)
            rows = [json.loads(line) for line in rerun_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest = json.loads((root / "runs" / "retry-errors" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["retrieval_error"])
        self.assertTrue(rows[0]["retrieval_debug"]["fixed"])
        self.assertEqual(manifest["summary"]["mode"], "retry_errors")
        self.assertEqual(manifest["summary"]["rows_pruned_for_retry"], 1)
        self.assertEqual(manifest["summary"]["rows_kept_for_retry"], 0)
        self.assertEqual(manifest["summary"]["rows_written"], 1)

    def test_retry_errors_prunes_incomplete_judge_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="retry-incomplete-judge",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            run_dir = root / "runs" / "retry-incomplete-judge"
            run_dir.mkdir(parents=True)
            partial_row = {
                "qa_id": "qa_fail",
                "question": "What does the adapter return?",
                "config": {
                    "experiment": config.name,
                    "retriever": "bm25",
                    "context_mode": "injected",
                    "model_provider": "dry_run",
                    "model": "dry-run",
                    "grader": "heuristic",
                },
                "answer": "old partial answer",
                "retrieval_error": None,
                "model_error": None,
                "evidence": [],
                "judge_scores": {"factual_accuracy": {"score": 1, "note": "partial"}},
                "judge_error": None,
            }
            (run_dir / "runs.jsonl").write_text(json.dumps(partial_row) + "\n", encoding="utf-8")

            runs_path = run_experiment(config, retry_errors=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["judge_scores"]), set(RUBRIC_KEYS))
        self.assertEqual(manifest["summary"]["rows_pruned_for_retry"], 1)
        self.assertEqual(manifest["summary"]["rows_written"], 1)

    def test_retry_errors_prunes_stale_grader_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            old_config = ExperimentConfig(
                name="retry-stale-grader",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="old-judge"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(old_config, overwrite=True)
            old_row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(old_row["config"]["grader"], "old-judge")

            new_config = ExperimentConfig(
                name=old_config.name,
                dataset=old_config.dataset,
                retrievers=old_config.retrievers,
                context_modes=old_config.context_modes,
                models=old_config.models,
                grader=GraderConfig(provider="heuristic", model="new-judge"),
                output_dir=old_config.output_dir,
            )
            rerun_path = run_experiment(new_config, retry_errors=True)
            rows = [json.loads(line) for line in rerun_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest = json.loads((root / "runs" / "retry-stale-grader" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["config"]["grader"], "new-judge")
        self.assertEqual(manifest["summary"]["rows_pruned_for_retry"], 1)
        self.assertEqual(manifest["summary"]["rows_written"], 1)

    def test_model_and_grader_build_failures_are_recorded_as_row_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="model-grader-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[RetrieverConfig(name="bm25", kind="bm25")],
                context_modes=["injected"],
                models=[ModelConfig(provider="unknown_provider", model="bad-model")],
                grader=GraderConfig(provider="unknown_grader", model="bad-judge"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["model_error"], "model_build_failed: ValueError: unknown model provider: unknown_provider")
        self.assertEqual(row["model_raw"]["model_build_error"], row["model_error"])
        self.assertEqual(row["judge_error"], "grade_failed: ValueError: unknown grader provider: unknown_grader")

    def test_external_command_failure_is_recorded_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="retrieval-error",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="failing_external",
                        kind="external_command",
                        options={
                            "command": [
                                sys.executable,
                                "-c",
                                "import sys; sys.stderr.write('adapter exploded'); sys.exit(7)",
                            ]
                        },
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["retrieval_error"], "external command exited with return code 7")
        self.assertEqual(rows[0]["retrieval_debug"]["returncode"], 7)
        self.assertIn("adapter exploded", rows[0]["retrieval_debug"]["stderr"])
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["retrieval_errors"], 1)
        self.assertEqual(metric_value(rows[0], "retrieval_failed"), 1.0)

    def test_external_command_runs_from_project_root(self) -> None:
        payload = {"evidence": [{"evidence_id": "hit", "kind": "tool_trace", "text": "ok"}]}
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
        with tempfile.TemporaryDirectory() as td, patch("gems_rag.retrieval.subprocess.run", return_value=completed) as run:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="external-cwd",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="external",
                        kind="external_command",
                        options={"command": ".venv/bin/python scripts/query_vector_db.py search --question '{question}'"},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        command = run.call_args.args[0]
        self.assertEqual(run.call_args.kwargs["cwd"], retrieval.ROOT)
        self.assertEqual(command[-1], "What does the adapter return?")
        self.assertIsNone(row["retrieval_error"])
        self.assertEqual(row["retrieval_debug"]["cwd"], str(retrieval.ROOT))
        self.assertEqual(row["evidence"][0]["evidence_id"], "hit")

    def test_external_command_accepts_explicit_cwd(self) -> None:
        payload = {"evidence": [{"evidence_id": "hit", "kind": "tool_trace", "text": "ok"}]}
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
        with tempfile.TemporaryDirectory() as td, patch("gems_rag.retrieval.subprocess.run", return_value=completed) as run:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="external-cwd-override",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="external",
                        kind="external_command",
                        options={
                            "command": [sys.executable, "-c", "print('ok')"],
                            "cwd": "external/rag-implementations",
                        },
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        expected_cwd = retrieval.ROOT / "external/rag-implementations"
        self.assertEqual(run.call_args.kwargs["cwd"], expected_cwd)
        self.assertEqual(row["retrieval_debug"]["cwd"], str(expected_cwd))

    def test_external_command_template_preserves_literal_braces(self) -> None:
        payload = {"evidence": [{"evidence_id": "hit", "kind": "tool_trace", "text": "ok"}]}
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
        with tempfile.TemporaryDirectory() as td, patch("gems_rag.retrieval.subprocess.run", return_value=completed) as run:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="external-braces",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="external",
                        kind="external_command",
                        top_k=9,
                        options={
                            "command": [
                                sys.executable,
                                "-c",
                                "print('{}')",
                                "--question",
                                "{question}",
                                "--qa",
                                "{qa_id}",
                                "--mrag",
                                "{mrag_dir}",
                                "--top-k",
                                "{top_k}",
                                "--json",
                                '{"open_hit_ids":[]}',
                                "--unknown",
                                "{unknown_placeholder}",
                                "--escaped",
                                "{{literal}}",
                            ]
                        },
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        command = run.call_args.args[0]
        self.assertIn("print('{}')", command)
        self.assertEqual(command[command.index("--question") + 1], "What does the adapter return?")
        self.assertEqual(command[command.index("--qa") + 1], "qa_fail")
        self.assertEqual(command[command.index("--mrag") + 1], str(mrag_dir))
        self.assertEqual(command[command.index("--top-k") + 1], "9")
        self.assertEqual(command[command.index("--json") + 1], '{"open_hit_ids":[]}')
        self.assertEqual(command[command.index("--unknown") + 1], "{unknown_placeholder}")
        self.assertEqual(command[command.index("--escaped") + 1], "{literal}")
        self.assertIsNone(row["retrieval_error"])

    def test_external_command_context_preserves_visual_metadata(self) -> None:
        payload = {
            "contexts": [
                {
                    "name": "visrag:page:0001",
                    "kind": "page",
                    "text": "MUTCD document page image 1. Sections: 2A.01.",
                    "score": 0.87,
                    "image_path": "/content/drive/MyDrive/MRAG/page_images/page_0001.png",
                    "metadata": {"page_pdf": 1, "section_ids": ["2A.01"]},
                }
            ]
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        command = [sys.executable, "-c", f"import base64; print(base64.b64decode('{encoded}').decode())"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            page_images = mrag_dir / "page_images"
            page_images.mkdir()
            local_image = page_images / "page_0001.png"
            local_image.write_bytes(b"fixture")
            config = ExperimentConfig(
                name="visual-context",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="visual_external",
                        kind="external_command",
                        options={"command": command},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        evidence = row["evidence"][0]
        self.assertEqual(evidence["kind"], "page")
        self.assertEqual(evidence["metadata"]["image_path"], str(local_image.resolve()))
        self.assertEqual(evidence["metadata"]["page_pdf"], 1)
        self.assertEqual(evidence["metadata"]["section_ids"], ["2A.01"])

    def test_external_command_preserves_harness_evidence_rows(self) -> None:
        payload = {
            "evidence": [
                {
                    "evidence_id": "chunk-2A-01",
                    "kind": "chunk",
                    "score": 0.77,
                    "metadata": {
                        "chunk_id": "chunk-2A-01",
                        "section_id": "2A.01",
                        "content_type": "Standard",
                        "page_printed": "2",
                    },
                    "text": "Traffic control devices shall fulfill a need.",
                },
                {
                    "evidence_id": "1",
                    "kind": "page",
                    "score": 0.5,
                    "metadata": {"page_pdf": 1, "image_path": "/tmp/page_0001.png"},
                    "text": "MUTCD page image with Section 2A.01.",
                },
            ]
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        command = [sys.executable, "-c", f"import base64; print(base64.b64decode('{encoded}').decode())"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="external-evidence-rows",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="vector_db_command_like",
                        kind="external_command",
                        options={"command": command},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        by_id = {item["evidence_id"]: item for item in row["evidence"]}
        self.assertEqual(set(by_id), {"chunk-2A-01", "page:1"})
        self.assertEqual(by_id["chunk-2A-01"]["kind"], "chunk")
        self.assertEqual(by_id["chunk-2A-01"]["metadata"]["section_id"], "2A.01")
        self.assertEqual(by_id["page:1"]["kind"], "page")
        self.assertEqual(by_id["page:1"]["metadata"]["image_path"], "/tmp/page_0001.png")

    def test_external_command_preserves_mrag_figures_and_pages(self) -> None:
        payload = {
            "chunks": [
                {
                    "chunk_id": "chunk-2A-01",
                    "section_id": "2A.01",
                    "section_title": "Function and Purpose",
                    "content_type": "Standard",
                    "ordinal": 1,
                    "page_printed": "2",
                    "part": "Part 2",
                    "text": "Traffic control devices shall fulfill a need.",
                    "score": 4.2,
                }
            ],
            "figures": [
                {
                    "figure_id": "Figure 2A-1",
                    "caption": "Example regulatory sign.",
                    "image_path": "/tmp/figure_2A-1.png",
                    "score": 0.91,
                }
            ],
            "pages": [
                {
                    "page_pdf": 1,
                    "page_printed": "2",
                    "text": "MUTCD page image with Section 2A.01.",
                    "image_path": "/tmp/page_0001.png",
                    "score": 0.82,
                }
            ],
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        command = [sys.executable, "-c", f"import base64; print(base64.b64decode('{encoded}').decode())"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir, qa_path = _fixture_mrag(root)
            config = ExperimentConfig(
                name="mrag-visual-external",
                dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                retrievers=[
                    RetrieverConfig(
                        name="mrag_reference_like",
                        kind="external_command",
                        options={"command": command},
                    )
                ],
                context_modes=["injected"],
                models=[ModelConfig(provider="dry_run", model="dry-run")],
                grader=GraderConfig(provider="heuristic", model="heuristic"),
                output_dir=root / "runs",
            )
            runs_path = run_experiment(config, overwrite=True)
            row = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])

        by_kind = {item["kind"]: item for item in row["evidence"]}
        self.assertEqual(set(by_kind), {"chunk", "figure", "page"})
        self.assertEqual(by_kind["chunk"]["metadata"]["section_id"], "2A.01")
        self.assertEqual(by_kind["figure"]["evidence_id"], "Figure 2A-1")
        self.assertEqual(by_kind["figure"]["metadata"]["image_path"], "/tmp/figure_2A-1.png")
        self.assertEqual(by_kind["page"]["evidence_id"], "page:1")
        self.assertEqual(by_kind["page"]["metadata"]["page_pdf"], 1)
        self.assertEqual(by_kind["page"]["metadata"]["image_path"], "/tmp/page_0001.png")


if __name__ == "__main__":
    unittest.main()
