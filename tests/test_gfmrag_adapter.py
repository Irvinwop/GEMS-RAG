from __future__ import annotations

import ast
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "query_gfmrag_index.py"
    spec = importlib.util.spec_from_file_location("query_gfmrag_index", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGfmRagAdapter(unittest.TestCase):
    def test_prepare_exports_mutcd_graph_in_official_stage1_format(self) -> None:
        mod = _load_script()
        graph = nx.MultiDiGraph()
        graph.add_node("chunk:chunk-1", kind="Chunk")
        graph.add_node("section:2A.01", kind="Section")
        graph.add_edge("chunk:chunk-1", "section:2A.01", label="belongs_to")
        chunks = [
            {
                "chunk_id": "chunk-1",
                "section_id": "2A.01",
                "section_title": "General",
                "text": "Shared corpus evidence.",
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            stage1 = Path(td)
            counts = mod._export_stage1(graph, chunks, stage1)
            with (stage1 / "nodes.csv").open(newline="", encoding="utf-8") as handle:
                nodes = list(csv.DictReader(handle))
            with (stage1 / "relations.csv").open(newline="", encoding="utf-8") as handle:
                relations = list(csv.DictReader(handle))
            with (stage1 / "edges.csv").open(newline="", encoding="utf-8") as handle:
                edges = list(csv.DictReader(handle))

        by_name = {row["name"]: row for row in nodes}
        self.assertEqual(by_name["chunk:chunk-1"]["type"], "document")
        self.assertEqual(ast.literal_eval(by_name["chunk:chunk-1"]["attributes"])["text"], "Shared corpus evidence.")
        self.assertEqual(by_name["section:2A.01"]["type"], "entity")
        self.assertEqual({row["name"] for row in relations}, {"belongs_to", "gfm_document_link"})
        self.assertEqual(edges[0]["source"], "chunk:chunk-1")
        self.assertEqual(edges[0]["target"], "section:2A.01")
        self.assertEqual(edges[1]["source"], "section:2A.01")
        self.assertEqual(edges[1]["target"], "chunk:chunk-1")
        self.assertEqual(
            counts,
            {"nodes": 2, "relations": 2, "edges": 2, "documents": 1, "document_links_added": 1},
        )

    def test_lexical_boundary_links_semantic_question_to_section_alias(self) -> None:
        mod = _load_script()
        linker = mod._LexicalEL()
        linker.index(["section:2A.04", "section:8D.09", "chunk:MUTCD11e_2A04_Option_15"])
        linker.set_aliases(
            {
                "section:2A.04": ["Design of Signs alternative legends special word legend signs"],
                "section:8D.09": ["Preemption of highway traffic signals at grade crossings"],
            }
        )

        result = linker(["May an agency use an alternative special word legend sign?"], topk=1)

        self.assertEqual(result["May an agency use an alternative special word legend sign?"][0]["entity"], "section:2A.04")
        self.assertNotIn("chunk:MUTCD11e_2A04_Option_15", linker.entities)

    def test_ner_preserves_explicit_mutcd_identifiers_without_stopword_entities(self) -> None:
        mod = _load_script()

        phrases = mod._LexicalNER()("What does Section 2A.04 say about Figure 2C-11 and sign W14-3?")

        self.assertIn("section:2A.04", phrases)
        self.assertIn("figure:Figure 2C-11", phrases)
        self.assertIn("signcode:W14-3", phrases)
        self.assertNotIn("what", phrases)

    def test_ready_marker_is_bound_to_stage1_content_and_stage2_graph(self) -> None:
        mod = _load_script()
        with tempfile.TemporaryDirectory() as td:
            dataset_root = Path(td)
            stage1 = dataset_root / "processed" / "stage1"
            stage1.mkdir(parents=True)
            for filename in mod.STAGE1_FILENAMES:
                (stage1 / filename).write_text(filename, encoding="utf-8")
            graph = dataset_root / "processed" / "stage2" / "fingerprint" / "graph.pt"
            graph.parent.mkdir(parents=True)
            graph.write_bytes(b"graph")
            fingerprint = mod._stage1_fingerprint(stage1)
            payload = {
                "status": "indexed",
                "model": mod.DEFAULT_MODEL,
                "model_revision": mod.DEFAULT_MODEL_REVISION,
                "stage1_fingerprint": fingerprint,
                "stage2_graphs": [str(graph.relative_to(dataset_root))],
            }

            self.assertTrue(
                mod._marker_matches(
                    payload,
                    model=mod.DEFAULT_MODEL,
                    model_revision=mod.DEFAULT_MODEL_REVISION,
                    stage1_fingerprint=fingerprint,
                    dataset_root=dataset_root,
                )
            )
            (stage1 / "nodes.csv").write_text("changed", encoding="utf-8")
            self.assertFalse(
                mod._marker_matches(
                    payload,
                    model=mod.DEFAULT_MODEL,
                    model_revision=mod.DEFAULT_MODEL_REVISION,
                    stage1_fingerprint=mod._stage1_fingerprint(stage1),
                    dataset_root=dataset_root,
                )
            )


if __name__ == "__main__":
    unittest.main()
