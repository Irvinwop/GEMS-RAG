from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gems_rag.cli import main
from gems_rag.config import ExperimentConfig, RetrieverConfig, load_experiment_config, write_experiment_config
from gems_rag.context_segments import write_context_segments


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="all/rags",
        retrievers=[
            RetrieverConfig(name="query", kind="bm25"),
            RetrieverConfig(
                name="gold",
                kind="oracle",
                context_modes=("injected", "tool_explore"),
                interaction="gold_reference",
            ),
            RetrieverConfig(
                name="none",
                kind="self_rag_policy",
                context_modes=("injected",),
                interaction="no_retrieval",
            ),
        ],
    )


class TestContextSegments(unittest.TestCase):
    def test_writes_only_compatible_retrievers_per_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "segments"
            report = write_context_segments(_config(), output_dir)
            segments = {segment["context_mode"]: segment for segment in report["segments"]}
            injected = load_experiment_config(Path(segments["injected"]["config"]))
            native = load_experiment_config(Path(segments["tool_native"]["config"]))

        self.assertEqual(report["segment_count"], 4)
        self.assertEqual(report["total_rows_per_question_model"], 7)
        self.assertEqual(injected.name, "all-rags-injected")
        self.assertEqual([retriever.name for retriever in injected.retrievers], ["query", "gold", "none"])
        self.assertEqual([retriever.name for retriever in native.retrievers], ["query"])
        self.assertEqual(segments["tool_search"]["excluded_retrievers"], ["gold", "none"])
        self.assertTrue(segments["tool_native"]["config"].endswith("all-rags-tool_native.json"))

    def test_cli_writes_selected_segments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            output_dir = root / "segments"
            write_experiment_config(_config(), config_path)

            code = main(
                [
                    "segment-contexts",
                    str(config_path),
                    "--context-modes",
                    "injected,tool_explore",
                    "--output-dir",
                    str(output_dir),
                ]
            )

        self.assertEqual(code, 0)

    def test_rejects_unknown_or_duplicate_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            with self.assertRaisesRegex(ValueError, "invalid context modes"):
                write_context_segments(_config(), output_dir, context_modes=["invalid"])
            with self.assertRaisesRegex(ValueError, "must be unique"):
                write_context_segments(
                    _config(),
                    output_dir,
                    context_modes=["injected", "injected"],
                )


if __name__ == "__main__":
    unittest.main()
