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
    retrievers = [
        RetrieverConfig(name="bm25", kind="bm25", top_k=6),
        RetrieverConfig(
            name="graphrag_local",
            kind="external_command",
            top_k=6,
            options={"command": ["graph", "query"], "check_command": ["graph", "check"]},
        ),
        RetrieverConfig(
            name="paperqa2_chunks",
            kind="external_command",
            top_k=6,
            options={"command": ["paperqa", "query"], "check_command": ["paperqa", "check"]},
        ),
    ]
    config = ExperimentConfig(
        name="comparison-fixture",
        dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=150),
        retrievers=retrievers,
        context_modes=["injected"],
        models=[ModelConfig(provider="dry_run", model="dry-run")],
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
                patch("gems_rag.comparison_study.validate_run", return_value=validation),
            ):
                report = run_comparison(config_path, root=root)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["run_mode"], "resume")
        runner.assert_called_once_with(config, overwrite=False, resume=True, retry_errors=False)

    def test_tracked_template_uses_exact_study_conditions(self) -> None:
        template = load_experiment_config(Path("configs/mutcd150-comparison.json"))

        self.assertEqual([retriever.name for retriever in template.retrievers], list(COMPARISON_RETRIEVERS))
        self.assertEqual(template.context_modes, ["injected"])
        self.assertEqual(template.dataset.limit, 150)
        self.assertEqual(template.max_evidence_chars, COMPARISON_MAX_EVIDENCE_CHARS)


if __name__ == "__main__":
    unittest.main()
