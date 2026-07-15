from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gems_rag.index_completion import (
    completion_marker_matches,
    publish_completion_marker,
    read_completion_marker,
)


class TestIndexCompletion(unittest.TestCase):
    def test_marker_is_atomic_and_requires_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "ready.json"
            identity = {"source": {"sha256": "abc", "bytes": 3}}
            publish_completion_marker(marker, identity, index_files=["index.bin"])

            self.assertTrue(completion_marker_matches(marker, identity))
            self.assertFalse(completion_marker_matches(marker, {"source": None}))
            self.assertEqual(read_completion_marker(marker)["index_files"], ["index.bin"])

    def test_failed_marker_serialization_preserves_previous_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "ready.json"
            marker.write_text('{"complete":true}\n', encoding="utf-8")
            previous = marker.read_bytes()

            with self.assertRaises(TypeError):
                publish_completion_marker(marker, {"bad": object()})

            self.assertEqual(marker.read_bytes(), previous)
            self.assertEqual([path for path in marker.parent.iterdir() if path != marker], [])


if __name__ == "__main__":
    unittest.main()
