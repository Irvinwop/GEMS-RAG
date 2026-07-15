from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from gems_rag.mrag_bundle import import_mrag_bundle


class TestMragBundle(unittest.TestCase):
    def test_import_restores_detached_blob_links_and_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            output = root / "output"
            fallback = root / "fallback"
            raw.mkdir()
            blob = b"model-weights"
            digest = hashlib.sha256(blob).hexdigest()
            detached = raw / f"{digest}-002"
            detached.write_bytes(blob)
            renamed_blob = b"image-blob"
            renamed_digest = hashlib.sha256(renamed_blob).hexdigest()
            fallback_blob = b"tokenizer-config"
            fallback_digest = _git_blob_sha1(fallback_blob)
            fallback_target = fallback / f"models--fixture/blobs/{fallback_digest}"
            fallback_target.parent.mkdir(parents=True)
            fallback_target.write_bytes(fallback_blob)
            qdrant_tar = _qdrant_tar()
            archive = raw / "MRAG-fixture-001.zip"
            pointer = f"../../blobs/{digest}"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("MRAG/mmrag_cache_v3/chunks.jsonl", "{}\n")
                handle.writestr("MRAG/mmrag_cache_v3/figures.jsonl", "{}\n{}\n")
                handle.writestr("MRAG/mmrag_cache_v3/graph.gpickle", b"graph")
                handle.writestr("MRAG/eval/gold_qa.jsonl", "{}\n")
                handle.writestr("MRAG/page_images/page_0001.png", b"png")
                handle.writestr("MRAG/figures/figure.png", b"png")
                handle.writestr("MRAG/qdrant_db.tar", qdrant_tar)
                handle.writestr(
                    "MRAG/hf_cache/models--fixture/snapshots/revision/model.safetensors",
                    pointer,
                )
                handle.writestr(
                    f"MRAG/hf_cache/models--fixture/blobs/{digest}",
                    pointer,
                )
                handle.writestr(
                    f"MRAG/hf_cache/models--fixture/blobs/{renamed_digest}.jpg",
                    renamed_blob,
                )
                handle.writestr(
                    "MRAG/hf_cache/models--fixture/snapshots/revision/image.jpg",
                    f"../../../blobs/{renamed_digest}",
                )
                handle.writestr(
                    "MRAG/hf_cache/models--fixture/snapshots/revision/tokenizer.json",
                    f"../../blobs/{fallback_digest}",
                )
                handle.writestr(
                    f"MRAG/hf_cache/models--fixture/blobs/{fallback_digest}",
                    f"../../blobs/{fallback_digest}",
                )

            report = import_mrag_bundle(raw, output, fallback_hf_caches=[fallback])
            model = output / "MRAG/hf_cache/models--fixture/snapshots/revision/model.safetensors"
            target = output / f"MRAG/hf_cache/models--fixture/blobs/{digest}"
            image = output / "MRAG/hf_cache/models--fixture/snapshots/revision/image.jpg"
            tokenizer = output / "MRAG/hf_cache/models--fixture/snapshots/revision/tokenizer.json"

            self.assertTrue(model.is_symlink())
            self.assertEqual(model.read_bytes(), blob)
            self.assertTrue(target.samefile(detached))
            self.assertTrue(image.is_symlink())
            self.assertEqual(image.read_bytes(), renamed_blob)
            self.assertTrue(tokenizer.is_symlink())
            self.assertEqual(tokenizer.read_bytes(), fallback_blob)
            self.assertEqual((output / "MRAG/qdrant_db/state.txt").read_text(), "ready")
            self.assertEqual(report["artifacts"]["figures"], 2)
            self.assertEqual(report["artifacts"]["gold_qa"], 1)
            self.assertEqual(report["hf_cache"]["restored_links"], 3)
            self.assertEqual(len(report["hf_cache"]["renamed_cache_blobs"]), 1)
            self.assertEqual(len(report["hf_cache"]["fallback_cache_blobs"]), 1)
            self.assertEqual(json.loads((output / "import_manifest.json").read_text())["status"], "complete")

            resumed = import_mrag_bundle(raw, output, fallback_hf_caches=[fallback])
            self.assertGreater(resumed["archives"][0]["skipped_files"], 0)
            self.assertGreater(resumed["qdrant"]["skipped_files"], 0)
            self.assertEqual(len(resumed["hf_cache"]["detached_blobs"]), 1)

    def test_import_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            with zipfile.ZipFile(raw / "MRAG-bad-001.zip", "w") as handle:
                handle.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(ValueError, "escapes output directory"):
                import_mrag_bundle(raw, root / "output", restore_qdrant=False)

    def test_import_rejects_parent_symlink_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            output = root / "output"
            outside = root / "outside"
            raw.mkdir()
            output.mkdir()
            outside.mkdir()
            (output / "MRAG").mkdir()
            (output / "MRAG/escape").symlink_to(outside, target_is_directory=True)
            with zipfile.ZipFile(raw / "MRAG-bad-001.zip", "w") as handle:
                handle.writestr("MRAG/escape/outside.txt", "bad")
            with self.assertRaisesRegex(ValueError, "parent symlink"):
                import_mrag_bundle(raw, output, restore_qdrant=False)
            self.assertFalse((outside / "outside.txt").exists())


def _qdrant_tar() -> bytes:
    output = io.BytesIO()
    payload = b"ready"
    with tarfile.open(fileobj=output, mode="w") as handle:
        info = tarfile.TarInfo("qdrant_db/state.txt")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
