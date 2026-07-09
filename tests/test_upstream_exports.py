from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from gem_rags.config import DatasetConfig, ExperimentConfig, RetrieverConfig, write_experiment_config


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
                selfrag_repo=root / "self-rag",
                crag_repo=root / "crag",
                selfrag_model="selfrag/test-model",
                selfrag_output=out_dir / "selfrag-preds.json",
                crag_output=out_dir / "crag-preds.txt",
                crag_device="cpu",
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

            self.assertEqual(report["upstream_repos"]["selfrag"]["repo"], str(root / "self-rag"))
            self.assertFalse(report["upstream_repos"]["selfrag"]["entrypoint_found"])
            self.assertEqual(report["upstream_repos"]["crag"]["repo"], str(root / "crag"))
            self.assertFalse(report["upstream_repos"]["crag"]["inference_entrypoint_found"])

            selfrag_command = report["upstream_commands"]["selfrag_run_short_form"]
            self.assertEqual(selfrag_command[0], "python")
            self.assertIn(str(root / "self-rag" / "retrieval_lm" / "run_short_form.py"), selfrag_command)
            self.assertIn("selfrag/test-model", selfrag_command)
            self.assertIn(str(out_dir / "selfrag_input.jsonl"), selfrag_command)
            self.assertIn(str(out_dir / "selfrag-preds.json"), selfrag_command)
            self.assertIn("--use_groundness", selfrag_command)
            self.assertIn("--use_utility", selfrag_command)
            self.assertIn("--use_seqscore", selfrag_command)

            crag_command = report["upstream_commands"]["crag_inference_template"]
            self.assertIn(str(root / "crag" / "scripts" / "CRAG_Inference.py"), crag_command)
            self.assertIn(str(out_dir / "crag_test_mutcd.txt"), crag_command)
            self.assertIn(str(out_dir / "crag-preds.txt"), crag_command)
            self.assertIn("cpu", crag_command)
            crag_eval = report["upstream_commands"]["crag_eval_match"]
            self.assertIn(str(root / "crag" / "scripts" / "eval.py"), crag_eval)
            self.assertIn(str(out_dir / "crag_answers.jsonl"), crag_eval)

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
            self.assertEqual(set(report["upstream_repos"]), {"selfrag"})
            self.assertEqual(set(report["upstream_commands"]), {"selfrag_run_short_form"})

    def test_config_retriever_selection_preserves_nested_policy_options(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qa_path, mrag_dir = _write_fixture(root)
            config_path = root / "config.json"
            write_experiment_config(
                ExperimentConfig(
                    name="upstream-config",
                    dataset=DatasetConfig(qa_path=qa_path, mrag_dir=mrag_dir, limit=1),
                    retrievers=[
                        RetrieverConfig(
                            name="nested_self",
                            kind="self_rag_policy",
                            top_k=1,
                            options={
                                "mode": "always_retrieve",
                                "base_retriever": {
                                    "name": "nested_bm25",
                                    "kind": "bm25",
                                    "top_k": 1,
                                    "options": {"graph_boost": False},
                                },
                            },
                        ),
                        RetrieverConfig(name="bm25", kind="bm25", top_k=1),
                    ],
                ),
                config_path,
            )
            args = argparse.Namespace(
                config=config_path,
                retriever="nested_self",
                qa_path=None,
                mrag_dir=None,
                limit=None,
                qa_ids=None,
                qa_ids_file=None,
                retriever_name="ignored",
                retriever_kind="ignored",
                top_k=99,
                retriever_option=[],
            )

            request = mod._export_request_from_args(args)
            out_dir = root / "exports"
            report = mod.export_upstream_inputs(
                qa_path=request["qa_path"],
                mrag_dir=request["mrag_dir"],
                out_dir=out_dir,
                retriever_config=request["retriever_config"],
                limit=request["limit"],
                qa_ids=request["qa_ids"],
                formats={"selfrag"},
            )
            exported_text = (out_dir / "selfrag_input.jsonl").read_text(encoding="utf-8")

        self.assertEqual(request["qa_path"], qa_path)
        self.assertEqual(request["mrag_dir"], mrag_dir)
        self.assertEqual(request["limit"], 1)
        self.assertEqual(request["retriever_config"].name, "nested_self")
        self.assertEqual(request["retriever_config"].options["base_retriever"]["name"], "nested_bm25")
        self.assertEqual(report["retriever"]["kind"], "self_rag_policy")
        self.assertEqual(report["evidence_counts"]["min"], 1)
        self.assertIn("Standard signs shall follow", exported_text)


if __name__ == "__main__":
    unittest.main()
