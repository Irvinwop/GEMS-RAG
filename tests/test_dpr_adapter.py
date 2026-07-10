from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "query_dpr_index.py"
    spec = importlib.util.spec_from_file_location("query_dpr_index", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDprAdapter(unittest.TestCase):
    def test_check_separates_environment_and_index_readiness(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "dpr"
            repo.mkdir()
            chunks = root / "chunks.jsonl"
            chunks.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                repo=repo,
                chunks=chunks,
                embeddings=root / "embeddings.npy",
                metadata=root / "metadata.jsonl",
                python=Path("missing-python"),
            )
            with patch.object(mod.importlib.util, "find_spec", return_value=object()):
                missing = mod._dependency_report(args)
            self.assertTrue(missing["environment_ready"])
            self.assertFalse(missing["index_ready"])
            self.assertFalse(missing["runnable"])

            args.embeddings.write_bytes(b"numpy")
            args.metadata.write_text("{}\n", encoding="utf-8")
            with patch.object(mod.importlib.util, "find_spec", return_value=object()):
                ready = mod._dependency_report(args)
            self.assertTrue(ready["index_ready"])
            self.assertTrue(ready["runnable"])

    def test_top_indices_are_ranked_by_dot_product_score(self) -> None:
        mod = _load_script()

        self.assertEqual(mod._top_indices([0.2, 0.9, -0.1, 0.4], 2), [1, 3])


if __name__ == "__main__":
    unittest.main()
