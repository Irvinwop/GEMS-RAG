from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from gems_rag.grading import RUBRIC_KEYS
from gems_rag.run_bundles import export_run_bundle, import_pro_grades, redact_secrets, run_row_id


def _row(image: Path | None = None) -> dict:
    metadata = {"section_id": "2A.04", "api_key": "secret"}
    if image is not None:
        metadata["image_path"] = str(image)
    return {
        "qa_id": "qa_1",
        "question": "What is required?",
        "config": {
            "experiment": "bundle-test",
            "retriever": "bm25",
            "context_mode": "injected",
            "model_provider": "dry_run",
            "model": "dry-run",
            "api_key": "do-not-export",
        },
        "run": {"run_id": "run-1"},
        "answer": "Use the standard sign.",
        "evidence": [{"evidence_id": "chunk-1", "kind": "chunk", "score": 1.0, "text": "Standard text", "metadata": metadata}],
    }


class TestRunBundles(unittest.TestCase):
    def test_redaction_keeps_backend_mode_metadata_but_removes_secrets(self) -> None:
        payload = redact_secrets(
            {
                "api_key": "secret",
                "allow_missing_api_key": True,
                "api_key_env": "LOCAL_OPENAI_API_KEY",
            }
        )

        self.assertEqual(payload["api_key"], "[REDACTED]")
        self.assertIs(payload["allow_missing_api_key"], True)
        self.assertEqual(payload["api_key_env"], "LOCAL_OPENAI_API_KEY")

    def test_gpt_pro_bundle_contains_tasks_images_template_and_redacted_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "figure.png"
            image.write_bytes(b"image")
            runs = root / "runs.jsonl"
            runs.write_text(json.dumps(_row(image)) + "\n", encoding="utf-8")
            qa = root / "qa.jsonl"
            qa.write_text(
                json.dumps({"qa_id": "qa_1", "question": "What is required?", "gold_answer": {"direct_answer": "Use it."}, "references": [], "gold_figures": []}) + "\n",
                encoding="utf-8",
            )
            output = root / "bundle.zip"
            grader_spec = root / "grader.md"
            grader_spec_text = "# Canonical grader\n\nUse the locked rubric.\n"
            grader_spec.write_text(grader_spec_text, encoding="utf-8")
            report = export_run_bundle(
                runs,
                output_path=output,
                qa_path=qa,
                grader_spec_path=grader_spec,
            )
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                task = json.loads(archive.read("grading_tasks.jsonl").decode().splitlines()[0])
                qa_pair = json.loads(archive.read("qa_pairs.jsonl").decode().splitlines()[0])
                manifest = json.loads(archive.read("manifest.json"))
                archived_row = json.loads(archive.read("run/runs.jsonl").decode().splitlines()[0])
                template = json.loads(archive.read("grades.template.jsonl").decode().splitlines()[0])
                bundled_grader_spec = archive.read(
                    "grader/MUTCD_RAG_EVALUATION_SPECIFICATION.md"
                ).decode()
                instructions = archive.read("GRADING.md").decode()

        self.assertEqual(report["grading_tasks"], 1)
        self.assertEqual(report["qa_pairs"], 1)
        self.assertEqual(report["evidence_images"], 1)
        self.assertIn("GRADING.md", names)
        self.assertTrue(any(name.startswith("evidence_images/") for name in names))
        self.assertTrue(task["retrieved_evidence"][0]["metadata"]["image_path"].startswith("evidence_images/"))
        self.assertEqual(task["rag_config"]["api_key"], "[REDACTED]")
        self.assertTrue(task["has_gold_answer"])
        self.assertEqual(qa_pair["question"], "What is required?")
        self.assertEqual(qa_pair["gold_answer"]["direct_answer"], "Use it.")
        self.assertEqual(manifest["qa_pairs"], 1)
        self.assertEqual(len(manifest["qa_sha256"]), 64)
        self.assertEqual(archived_row["config"]["api_key"], "[REDACTED]")
        self.assertEqual(set(template["judge_scores"]), set(RUBRIC_KEYS))
        self.assertEqual(bundled_grader_spec, grader_spec_text)
        self.assertTrue(manifest["grader_specification"]["included"])
        self.assertEqual(len(manifest["grader_specification"]["sha256"]), 64)
        self.assertTrue(report["grader_spec_included"])
        self.assertIn("canonical evaluation protocol", instructions)

    def test_question_only_bundle_preserves_ids_and_includes_manual_as_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            mrag_dir = root / "MRAG"
            run_dir.mkdir()
            mrag_dir.mkdir()
            manual = mrag_dir / "mutcd11theditionr1hl.pdf"
            manual_bytes = b"%PDF-1.4 fixture"
            manual.write_bytes(manual_bytes)
            qa = root / "questions.jsonl"
            qa.write_text(
                json.dumps({"question_id": "T001", "question": "What is the purpose?"}) + "\n",
                encoding="utf-8",
            )
            runs = run_dir / "runs.jsonl"
            runs.write_text(
                json.dumps({**_row(), "qa_id": "T001", "question": "What is the purpose?"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "materialized_config.json").write_text(
                json.dumps({"dataset": {"qa_path": str(qa), "mrag_dir": str(mrag_dir)}}) + "\n",
                encoding="utf-8",
            )

            report = export_run_bundle(runs, output_path=run_dir / "bundle.zip")
            with zipfile.ZipFile(run_dir / "bundle.zip") as archive:
                task = json.loads(archive.read("grading_tasks.jsonl").decode().splitlines()[0])
                pair = json.loads(archive.read("qa_pairs.jsonl").decode().splitlines()[0])
                manifest = json.loads(archive.read("manifest.json"))
                instructions = archive.read("GRADING.md").decode()
                bundled_manual = archive.read("source/mutcd-manual.pdf")

        self.assertEqual(task["qa_id"], "T001")
        self.assertFalse(task["has_gold_answer"])
        self.assertEqual(task["gold_answer"], {})
        self.assertFalse(pair["has_gold_answer"])
        self.assertEqual(manifest["question_only_pairs"], 1)
        self.assertEqual(manifest["gold_answer_pairs"], 0)
        self.assertTrue(manifest["manual"]["included"])
        self.assertEqual(report["question_only_pairs"], 1)
        self.assertTrue(report["manual_included"])
        self.assertEqual(bundled_manual, manual_bytes)
        self.assertIn("Upstream model-generated answers are intentionally not used as gold", instructions)

    def test_import_merges_external_grades_and_preserves_answer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = _row()
            runs = root / "runs.jsonl"
            runs.write_text(json.dumps(row) + "\n", encoding="utf-8")
            grades = root / "grades.jsonl"
            grades.write_text(
                json.dumps(
                    {
                        "row_id": run_row_id(row),
                        "judge_scores": {"factual_accuracy": {"score": 5, "note": "correct"}},
                        "judge_confidence": 0.9,
                        "judge_explanation": "Grounded.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "graded.jsonl"
            report = import_pro_grades(runs, grades, output_path=output)
            updated = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(report["ok"])
        self.assertEqual(updated["answer"], row["answer"])
        self.assertEqual(updated["config"]["grader_provider"], "gpt_pro")
        self.assertEqual(updated["judge_scores"]["factual_accuracy"]["score"], 5)
        self.assertEqual(set(updated["judge_scores"]), set(RUBRIC_KEYS))

    def test_partial_import_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [_row(), {**_row(), "qa_id": "qa_2"}]
            runs = root / "runs.jsonl"
            runs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            grades = root / "grades.jsonl"
            grades.write_text(json.dumps({"row_id": run_row_id(rows[0]), "judge_scores": {}}) + "\n", encoding="utf-8")
            report = import_pro_grades(runs, grades, output_path=root / "graded.jsonl")

        self.assertFalse(report["ok"])
        self.assertEqual(report["grades_missing"], 1)
