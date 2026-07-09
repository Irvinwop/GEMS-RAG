from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from gem_rags.config import RetrieverConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "export_upstream_eval_inputs.py"
    spec = importlib.util.spec_from_file_location("export_upstream_eval_inputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> tuple[Path, Path]:
    mrag_dir = root / "MRAG"
    cache = mrag_dir / "mmrag_cache_v3"
    eval_dir = mrag_dir / "eval"
    cache.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    qa_path = eval_dir / "gold_qa.jsonl"
    qa_path.write_text(
        json.dumps(
            {
                "qa_id": "qa_1",
                "question": "What does Section 2A.04 require for standard signs?",
                "gold_answer": {"direct_answer": "Standard signs must follow the stated design."},
                "references": [{"section_id": "2A.04"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cache / "chunks.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "chunk-1",
                        "section_id": "2A.04",
                        "content_type": "Standard",
                        "ordinal": "01",
                        "section_title": "Standardization of Application",
                        "page_printed": "23",
                        "part": "Part 2",
                        "chapter": "2A",
                        "text": "Standard signs shall follow the MUTCD design and application requirements.",
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "chunk-2",
                        "section_id": "2A.05",
                        "content_type": "Guidance",
                        "ordinal": "01",
                        "section_title": "Classification of Signs",
                        "page_printed": "24",
                        "part": "Part 2",
                        "chapter": "2A",
                        "text": "Warning signs should be used where road users need warning.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cache / "figures.jsonl").write_text("", encoding="utf-8")
    return qa_path, mrag_dir


class TestUpstreamExports(unittest.TestCase):
    def test_exports_selfrag_and_crag_inputs_from_retriever_evidence(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_fixture(root)
            out_dir = root / "exports"

            report = mod.export_upstream_inputs(
                qa_path=qa_path,
                mrag_dir=mrag_dir,
                out_dir=out_dir,
                retriever_config=RetrieverConfig(name="bm25", kind="bm25", top_k=2),
                formats={"selfrag", "crag"},
                crag_ndocs=3,
            )

            self.assertEqual(report["qa_count"], 1)
            self.assertEqual(report["formats"], ["crag", "selfrag"])
            self.assertTrue((out_dir / "manifest.json").exists())

            selfrag_row = json.loads((out_dir / "selfrag_input.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(selfrag_row["qa_id"], "qa_1")
            self.assertEqual(selfrag_row["answers"], ["Standard signs must follow the stated design."])
            self.assertEqual(selfrag_row["ctxs"][0]["id"], "chunk-1")
            self.assertIn("Standard signs shall follow", selfrag_row["ctxs"][0]["text"])

            crag_lines = (out_dir / "crag_test_mutcd.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(crag_lines), 3)
            self.assertTrue(crag_lines[0].startswith("What does Section 2A.04 require"))
            self.assertIn(" [SEP] ", crag_lines[0])
            self.assertTrue(crag_lines[-1].endswith(" [SEP] "))

            retrieved = (out_dir / "crag_retrieved_psgs").read_text(encoding="utf-8")
            self.assertIn("Standard signs shall follow", retrieved)
            answers = [json.loads(line) for line in (out_dir / "crag_answers.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(answers[0]["qa_id"], "qa_1")

    def test_single_format_export_does_not_write_other_format(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_fixture(root)
            out_dir = root / "exports"

            report = mod.export_upstream_inputs(
                qa_path=qa_path,
                mrag_dir=mrag_dir,
                out_dir=out_dir,
                retriever_config=RetrieverConfig(name="bm25", kind="bm25", top_k=1),
                formats={"selfrag"},
            )

            self.assertEqual(report["formats"], ["selfrag"])
            self.assertTrue((out_dir / "selfrag_input.jsonl").exists())
            self.assertFalse((out_dir / "crag_test_mutcd.txt").exists())


if __name__ == "__main__":
    unittest.main()
