from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "query_visrag_index.py"
    spec = importlib.util.spec_from_file_location("query_visrag_index", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVisragAdapter(unittest.TestCase):
    def test_prepare_manifest_repairs_local_page_and_figure_paths(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            (mrag_dir / "page_images").mkdir(parents=True)
            (mrag_dir / "figures").mkdir(parents=True)
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir()
            (mrag_dir / "page_images" / "page_0001.png").write_bytes(b"fake")
            (mrag_dir / "figures" / "figure_2A-1_p0001.png").write_bytes(b"fake")
            (cache / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "MUTCD11e_2A01_Standard_01",
                        "section_id": "2A.01",
                        "page_pdf": 1,
                        "page_printed": "2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (cache / "figures.jsonl").write_text(
                json.dumps(
                    {
                        "figure_id": "Figure 2A-1",
                        "kind": "Figure",
                        "canonical_id": "2A-1",
                        "page_pdf": 1,
                        "image_path": "/content/drive/MyDrive/MRAG/figures/figure_2A-1_p0001.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = root / "visual_manifest.jsonl"
            report = mod.prepare_manifest(mrag_dir, manifest, scope="both")
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["records"], 2)
        self.assertEqual(report["pages"], 1)
        self.assertEqual(report["figures"], 1)
        self.assertEqual(rows[0]["kind"], "page")
        self.assertEqual(rows[0]["metadata"]["section_ids"], ["2A.01"])
        self.assertTrue(rows[0]["image_path"].endswith("page_0001.png"))
        self.assertEqual(rows[1]["kind"], "figure")
        self.assertTrue(rows[1]["image_path"].endswith("figure_2A-1_p0001.png"))


if __name__ == "__main__":
    unittest.main()
