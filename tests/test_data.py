from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gems_rag.data import canonicalize_chunks, load_chunks, load_figures


class TestChunkLoading(unittest.TestCase):
    def test_chunk_id_collisions_keep_the_highest_quality_record(self) -> None:
        rows = [
            {"chunk_id": "same", "text": "The standard requires a complete stop.", "section_refs": ["2A.01"]},
            {"chunk_id": "same", "text": "4.5", "section_refs": []},
            {"chunk_id": "other", "text": "Other evidence."},
        ]

        canonical, report = canonicalize_chunks(rows)

        self.assertEqual([row["chunk_id"] for row in canonical], ["same", "other"])
        self.assertEqual(canonical[0]["text"], "The standard requires a complete stop.")
        self.assertEqual(
            report,
            {"raw_rows": 3, "unique_chunks": 2, "collision_rows": 1, "colliding_ids": 1},
        )

    def test_load_chunks_applies_collision_canonicalization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mrag_dir = Path(td)
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir()
            rows = [
                {"chunk_id": "same", "text": "short"},
                {"chunk_id": "same", "text": "a substantially richer passage"},
            ]
            (cache / "chunks.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            loaded = load_chunks(mrag_dir)

        self.assertEqual(loaded, [rows[1]])

    def test_load_figures_localizes_colab_paths_to_extracted_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mrag_dir = Path(td)
            cache = mrag_dir / "mmrag_cache_v3"
            figures = mrag_dir / "figures"
            cache.mkdir()
            figures.mkdir()
            image_path = figures / "figure_4J-1_p0768.png"
            image_path.write_bytes(b"fixture")
            row = {
                "figure_id": "Figure 4J-1",
                "kind": "Figure",
                "image_path": "/content/drive/MyDrive/MRAG/figures/figure_4J-1_p0768.png",
            }
            (cache / "figures.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            loaded = load_figures(mrag_dir)

        self.assertEqual(loaded[0]["image_path"], str(image_path.resolve()))


if __name__ == "__main__":
    unittest.main()
