from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag.comparison_study import (
    BENCHMARK_DISTRIBUTION,
    COMPARISON_MAX_EVIDENCE_CHARS,
    COMPARISON_RETRIEVERS,
    bundle_comparison,
    comparison_contract,
    run_comparison,
)
from gems_rag.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RetrieverConfig,
    load_experiment_config,
    write_experiment_config,
)
from gems_rag.run_bundles import run_row_id


def _fixture(root: Path) -> tuple[ExperimentConfig, str]:
    qa_path = root / "questions.jsonl"
    rows = []
    for prefix, count in BENCHMARK_DISTRIBUTION.items():
        rows.extend(
            {"question_id": f"{prefix}{index:03d}", "question": f"Question {prefix} {index}?"}
            for index in range(1, count + 1)
        )
    qa_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(qa_path.read_bytes()).hexdigest()
    mrag_dir = root / "MRAG"
    (mrag_dir / "mmrag_cache_v3").mkdir(parents=True)
    (mrag_dir / "mutcd11theditionr1hl.pdf").write_bytes(b"%PDF-1.4 fixture")
    (mrag_dir / "mmrag_cache_v3" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    snapshot_path = root / "retrieval_snapshot.jsonl"
    snapshot_path.write_text("", encoding="utf-8")
    snapshot_path.with_suffix(".manifest.json").write_text("{}\n", encoding="utf-8")
    retrievers = [
        RetrieverConfig(
            name="bm25",
            kind="bm25",
            top_k=6,
            context_modes=("injected",),
        ),
        RetrieverConfig(
            name="graphrag_local",
            kind="external_command",
            top_k=6,
            options={"command": ["graph", "query"], "check_command": ["graph", "check"]},
            context_modes=("injected",),
        ),
        RetrieverConfig(
            name="paperqa2_chunks",
            kind="external_command",
            top_k=6,
            options={"command": ["paperqa", "query"], "check_command": ["paperqa", "check"]},
            context_modes=("injected",),
        ),
    ]
    config = ExperimentConfig(
        name="comparison-fixture",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=150),
        retrievers=retrievers,
        context_modes=["injected"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
        retrieval_snapshot=snapshot_path,
        output_dir=root / "runs",
        max_evidence_chars=COMPARISON_MAX_EVIDENCE_CHARS,
        dry_run=True,
    )
    return config, digest


class TestComparisonStudy(unittest.TestCase):
    def test_contract_accepts_only_the_locked_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, digest = _fixture(root)
            report = comparison_contract(config, root=root, expected_sha256=digest)

        self.assertTrue(report["ok"])
        self.assertEqual(report["retrievers"], list(COMPARISON_RETRIEVERS))
        self.assertEqual(report["expected_rows"], 450)
        self.assertEqual(report["benchmark"]["distribution"], BENCHMARK_DISTRIBUTION)

    def test_contract_rejects_partial_or_tool_context_studies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, digest = _fixture(root)
            invalid = ExperimentConfig(
                name=config.name,
                dataset=DatasetConfig(
                    qa_path=config.dataset.qa_path,
                    mrag_dir=config.dataset.mrag_dir,
                    limit=12,
                ),
                retrievers=config.retrievers[:2],
                context_modes=["tool_native"],
                models=config.models,
                grader=config.grader,
                output_dir=config.output_dir,
                max_evidence_chars=1600,
            )
            report = comparison_contract(invalid, root=root, expected_sha256=digest)

        self.assertFalse(report["ok"])
        self.assertTrue(any("retrievers must be exactly" in problem for problem in report["problems"]))
        self.assertTrue(any("context_modes" in problem for problem in report["problems"]))
        self.assertTrue(any("dataset limit" in problem for problem in report["problems"]))

    def test_runner_defaults_to_resume_and_returns_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, digest = _fixture(root)
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            runs_path = root / "runs" / config.name / "runs.jsonl"
            runs_path.parent.mkdir(parents=True)
            runs_path.write_text("{}\n", encoding="utf-8")
            validation = {"ok": True, "status": "ready"}
            with (
                patch(
                    "gems_rag.comparison_study.require_comparison_contract",
                    return_value={"ok": True, "benchmark": {"sha256": digest}},
                ),
                patch("gems_rag.comparison_study.run_experiment", return_value=runs_path) as runner,
                patch("gems_rag.comparison_study.validate_comparison_run", return_value=validation),
            ):
                report = run_comparison(config_path, root=root, create_bundle=False)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["run_mode"], "resume")
        runner.assert_called_once_with(config, overwrite=False, resume=True, retry_errors=False)

    def test_tracked_template_uses_exact_study_conditions(self) -> None:
        template = load_experiment_config(Path("configs/mutcd150-comparison.json"))

        self.assertEqual([retriever.name for retriever in template.retrievers], list(COMPARISON_RETRIEVERS))
        self.assertTrue(
            all(
                tuple(retriever.context_modes) == ("injected",)
                for retriever in template.retrievers
            )
        )
        self.assertEqual(template.context_modes, ["injected"])
        self.assertEqual(template.dataset.limit, 150)
        self.assertEqual(template.max_evidence_chars, COMPARISON_MAX_EVIDENCE_CHARS)
        self.assertEqual(
            template.retrieval_snapshot,
            Path("data/working/mutcd150-comparison/retrieval_snapshot.jsonl"),
        )

    def test_final_bundle_is_blocked_before_export_when_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, _digest = _fixture(root)
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            runs_path = root / "runs.jsonl"
            runs_path.write_text("{}\n", encoding="utf-8")
            validation = {"ok": False, "problems": ["missing expected rows: 449"]}
            with (
                patch("gems_rag.comparison_study.validate_comparison_run", return_value=validation),
                patch("gems_rag.comparison_study.export_run_bundle") as export,
                self.assertRaisesRegex(ValueError, "missing expected rows"),
            ):
                bundle_comparison(config_path, runs_path=runs_path, root=root)

        export.assert_not_called()

    def test_bundle_writes_canonical_answer_retrieval_and_study_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, _digest = _fixture(root)
            config_path = root / "config.json"
            write_experiment_config(config, config_path)
            run_dir = root / "runs" / config.name
            run_dir.mkdir(parents=True)
            runs_path = run_dir / "runs.jsonl"
            answer = "Direct Answer: Use the standard."
            row = {
                "qa_id": "T001",
                "question": "What is required?",
                "run_status": "successful",
                "answer": answer,
                "serialized_return": {"answer": answer},
                "config": {
                    "retriever": "bm25",
                    "context_mode": "injected",
                    "model_provider": "dry_run",
                    "model": "dry-run",
                },
                "run": {"run_id": "primary"},
                "model_raw": {"finish_reason": "stop"},
                "retrieval_error": None,
                "model_error": None,
                "judge_error": None,
                "evidence": [],
                "retrieval_debug": {
                    "snapshot_reused": True,
                    "retrieved_evidence_count": 0,
                    "provided_evidence_count": 0,
                },
            }
            runs_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            grader_spec = root / "grader.md"
            grader_spec.write_text("# Grader\n", encoding="utf-8")
            validation = {"ok": True, "problems": [], "expected_rows": 450}
            with (
                patch("gems_rag.comparison_study.validate_comparison_run", return_value=validation),
                patch(
                    "gems_rag.comparison_study.export_run_bundle",
                    return_value={"status": "complete", "output": str(root / "bundle.zip")},
                ) as export,
            ):
                report = bundle_comparison(
                    config_path,
                    runs_path=runs_path,
                    grader_spec_path=grader_spec,
                    root=root,
                )
            manifest = json.loads((run_dir / "study_manifest.json").read_text(encoding="utf-8"))
            canonical_answer = json.loads(
                (run_dir / "canonical_answers.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(canonical_answer["answer"], answer)
        self.assertEqual(canonical_answer["row_id"], run_row_id(row))
        self.assertEqual(manifest["canonical_rows"], 1)
        self.assertEqual(
            manifest["retrieval_snapshot"]["bundle_paths"],
            ["retrieval_snapshot.jsonl", "retrieval_snapshot_manifest.json"],
        )
        self.assertFalse(manifest["source_authority"]["evaluator_annotations_in_bundle"])
        export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
