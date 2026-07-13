from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.data import load_qa_items
from gems_rag.qa_sets import (
    evaluate_qa_coverage,
    load_qa_ids_file,
    make_qa_split,
    qa_coverage_for_selection,
    qa_coverage_report,
    summarize_qa_items,
    write_qa_split,
)


def _qa_path(root: Path) -> Path:
    rows = [
        {
            "qa_id": "qa_1",
            "question": "Normal with refs",
            "expected_refusal": False,
            "references": [{"section_id": "2A.04", "content_type": "Standard"}],
            "gold_figures": [],
        },
        {
            "qa_id": "qa_2",
            "question": "Refusal",
            "expected_refusal": True,
            "references": [],
            "gold_figures": [],
        },
        {
            "qa_id": "qa_3",
            "question": "Figure case",
            "expected_refusal": False,
            "references": [{"section_id": "4J.03", "content_type": "Guidance"}],
            "gold_figures": [{"figure_id": "Figure 4J-3"}],
        },
        {
            "qa_id": "qa_4",
            "question": "Normal no refs",
            "expected_refusal": False,
            "references": [],
            "gold_figures": [],
        },
    ]
    path = root / "qa.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


class TestQaSets(unittest.TestCase):
    def test_summarize_qa_items_counts_refusals_figures_and_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            items = load_qa_items(_qa_path(Path(td)))
        summary = summarize_qa_items(items)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["expected_refusal"], {"true": 1, "false": 3})
        self.assertEqual(summary["has_gold_figures"], {"true": 1, "false": 3})
        self.assertEqual(summary["has_references"], {"true": 2, "false": 2})
        self.assertEqual(summary["reference_content_types"], {"Guidance": 1, "Standard": 1})
        self.assertEqual(summary["reference_parts"], {"Part 2": 1, "Part 4": 1})

    def test_balanced_split_is_deterministic_and_covers_strata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            items = load_qa_items(_qa_path(Path(td)))
        first = make_qa_split(items, size=3, seed=7, strategy="balanced")
        second = make_qa_split(items, size=3, seed=7, strategy="balanced")
        self.assertEqual(first["qa_ids"], second["qa_ids"])
        self.assertEqual(first["size"], 3)
        self.assertIn("qa_2", first["qa_ids"])
        self.assertIn("qa_3", first["qa_ids"])

    def test_qa_coverage_report_compares_selected_to_available_strata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            items = load_qa_items(_qa_path(Path(td)))
        selected = [item for item in items if item.qa_id in {"qa_2", "qa_3"}]

        report = qa_coverage_report(items, selected)

        self.assertEqual(report["available"]["total"], 4)
        self.assertEqual(report["selected"]["total"], 2)
        self.assertEqual(report["coverage"]["selected_fraction"], 0.5)
        self.assertEqual(report["coverage"]["strata_total"], 4)
        self.assertEqual(report["coverage"]["strata_covered"], 2)
        missing = [row for row in report["strata"] if not row["covered"]]
        self.assertEqual(len(missing), 2)
        figure_row = next(row for row in report["strata"] if row["has_gold_figures"])
        self.assertEqual(figure_row["selected_count"], 1)

    def test_qa_coverage_gate_reports_each_missing_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = _qa_path(Path(td))
            report = qa_coverage_for_selection(path, qa_ids=["qa_2", "qa_3"])

        gate = evaluate_qa_coverage(report, min_selected_per_stratum=1)

        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertFalse(gate["ok"])
        self.assertEqual(len(gate["checks"]), 4)
        self.assertEqual(len(gate["failed"]), 2)
        self.assertTrue(all(check["shortfall"] == 1 for check in gate["failed"]))

    def test_qa_coverage_gate_requires_a_positive_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            evaluate_qa_coverage({"strata": []}, min_selected_per_stratum=0)

    def test_load_qa_ids_file_accepts_split_json_and_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_path = root / "split.json"
            write_qa_split(split_path, {"qa_ids": ["qa_1", "qa_2"]})
            text_path = root / "ids.txt"
            text_path.write_text("qa_3\nqa_4\n", encoding="utf-8")
            self.assertEqual(load_qa_ids_file(split_path), ["qa_1", "qa_2"])
            self.assertEqual(load_qa_ids_file(text_path), ["qa_3", "qa_4"])


if __name__ == "__main__":
    unittest.main()
