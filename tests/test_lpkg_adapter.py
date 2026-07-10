from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "prepare_lpkg_plans.py"
    spec = importlib.util.spec_from_file_location("prepare_lpkg_plans", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLPKGAdapter(unittest.TestCase):
    def test_normalize_aligns_official_predictions_with_qa_ids(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            predictions = root / "generated_predictions.jsonl"
            out = root / "plans.jsonl"
            qa_path.write_text(
                json.dumps({"qa_id": "qa-1", "question": "What applies?", "gold_answer": {}}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps(
                    {
                        "label": "answer",
                        "predict": 'Sub_Question_1: str = "Which provision applies?"',
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = mod.normalize_predictions(predictions, qa_path, out)
            row = json.loads(out.read_text(encoding="utf-8"))
            check = mod.check_plans(out, qa_path, ROOT / "external" / "rag-implementations" / "lpkg")

        self.assertEqual(report["plan_count"], 1)
        self.assertEqual(row["qa_id"], "qa-1")
        self.assertEqual(row["question"], "What applies?")
        self.assertTrue(check["runnable"])
        self.assertEqual(check["missing_qa_ids"], [])

    def test_normalize_rejects_row_count_mismatch(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path = root / "qa.jsonl"
            predictions = root / "generated_predictions.jsonl"
            qa_path.write_text(
                json.dumps({"qa_id": "qa-1", "question": "Question?", "gold_answer": {}}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prediction/QA row mismatch"):
                mod.normalize_predictions(predictions, qa_path, root / "plans.jsonl")


if __name__ == "__main__":
    unittest.main()
