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
        self.assertEqual(relations[0]["name"], "belongs_to")
        self.assertEqual(edges[0]["source"], "chunk:chunk-1")
        self.assertEqual(edges[0]["target"], "section:2A.01")
        self.assertEqual(counts, {"nodes": 2, "relations": 1, "edges": 1, "documents": 1})


if __name__ == "__main__":
    unittest.main()
