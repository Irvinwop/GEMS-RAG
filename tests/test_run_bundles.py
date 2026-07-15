from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from gems_rag.grading import RUBRIC_KEYS
from gems_rag.run_bundles import export_run_bundle, import_pro_grades, run_row_id


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
            report = export_run_bundle(runs, output_path=output, qa_path=qa)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                task = json.loads(archive.read("grading_tasks.jsonl").decode().splitlines()[0])
                qa_pair = json.loads(archive.read("qa_pairs.jsonl").decode().splitlines()[0])
                manifest = json.loads(archive.read("manifest.json"))
                archived_row = json.loads(archive.read("run/runs.jsonl").decode().splitlines()[0])
                template = json.loads(archive.read("grades.template.jsonl").decode().splitlines()[0])

        self.assertEqual(report["grading_tasks"], 1)
        self.assertEqual(report["qa_pairs"], 1)
        self.assertEqual(report["evidence_images"], 1)
        self.assertIn("GRADING.md", names)
        self.assertTrue(any(name.startswith("evidence_images/") for name in names))
        self.assertTrue(task["retrieved_evidence"][0]["metadata"]["image_path"].startswith("evidence_images/"))
        self.assertEqual(task["rag_config"]["api_key"], "[REDACTED]")
        self.assertEqual(qa_pair["question"], "What is required?")
        self.assertEqual(qa_pair["gold_answer"]["direct_answer"], "Use it.")
        self.assertEqual(manifest["qa_pairs"], 1)
        self.assertEqual(len(manifest["qa_sha256"]), 64)
        self.assertEqual(archived_row["config"]["api_key"], "[REDACTED]")
        self.assertEqual(set(template["judge_scores"]), set(RUBRIC_KEYS))

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
