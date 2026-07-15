from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_index_checkpoints_completed_batches_and_resumes(self) -> None:
        import numpy as np

        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "visual_manifest.jsonl"
            records = [{"id": f"page:{index:04d}", "value": index} for index in range(5)]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            embeddings = root / "embeddings.npy"
            args = _index_args(mod, manifest, embeddings)
            signature = mod._index_signature(args, len(records))
            artifacts = mod._index_artifact_paths(embeddings)
            calls = 0

            def interrupted_encoder(batch):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return np.asarray([[row["value"], row["value"] + 10] for row in batch], dtype=np.float32)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                mod._write_resumable_index(
                    records=records,
                    signature=signature,
                    embeddings=embeddings,
                    artifacts=artifacts,
                    batch_size=2,
                    encode_batch=interrupted_encoder,
                    np=np,
                )

            progress = json.loads(artifacts["progress"].read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_rows"], 2)
            self.assertTrue(artifacts["partial"].exists())
            self.assertFalse(embeddings.exists())
            self.assertFalse(mod._index_readiness(args, len(records), None)["ready"])

            resumed_values = []

            def resumed_encoder(batch):
                resumed_values.extend(row["value"] for row in batch)
                return np.asarray([[row["value"], row["value"] + 10] for row in batch], dtype=np.float32)

            report = mod._write_resumable_index(
                records=records,
                signature=signature,
                embeddings=embeddings,
                artifacts=artifacts,
                batch_size=2,
                encode_batch=resumed_encoder,
                np=np,
            )
            matrix = np.load(embeddings).copy()
            readiness = mod._index_readiness(args, len(records), mod._embedding_info(embeddings))
            self.assertFalse(artifacts["partial"].exists())
            self.assertFalse(artifacts["progress"].exists())

        self.assertEqual(report["resumed_from"], 2)
        self.assertEqual(resumed_values, [2, 3, 4])
        np.testing.assert_array_equal(matrix[:, 0], np.arange(5, dtype=np.float32))
        self.assertTrue(readiness["ready"])

    def test_ready_marker_rejects_changed_manifest(self) -> None:
        import numpy as np

        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "visual_manifest.jsonl"
            records = [{"id": "page:0001", "value": 1}]
            manifest.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
            embeddings = root / "embeddings.npy"
            args = _index_args(mod, manifest, embeddings)
            signature = mod._index_signature(args, len(records))
            artifacts = mod._index_artifact_paths(embeddings)
            mod._write_resumable_index(
                records=records,
                signature=signature,
                embeddings=embeddings,
                artifacts=artifacts,
                batch_size=1,
                encode_batch=lambda batch: np.asarray([[1.0, 0.0] for _ in batch], dtype=np.float32),
                np=np,
            )
            self.assertTrue(mod._index_readiness(args, 1, mod._embedding_info(embeddings))["ready"])

            manifest.write_text(json.dumps({"id": "page:0001", "value": 2}) + "\n", encoding="utf-8")
            readiness = mod._index_readiness(args, 1, mod._embedding_info(embeddings))

        self.assertFalse(readiness["ready"])
        self.assertIn("ready_manifest_sha256_mismatch", readiness["reasons"])

    def test_default_model_revision_is_pinned(self) -> None:
        mod = _load_script()
        self.assertEqual(mod.DEFAULT_MODEL_REVISION, "95ef596df871b606167cb7e4b7215caf1bfdf761")


def _index_args(mod, manifest: Path, embeddings: Path) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=manifest,
        embeddings=embeddings,
        model_name_or_path=mod.DEFAULT_MODEL,
        model_revision=mod.DEFAULT_MODEL_REVISION,
        trust_remote_code=True,
        device="mps",
        dtype="float16",
    )


if __name__ == "__main__":
    unittest.main()
