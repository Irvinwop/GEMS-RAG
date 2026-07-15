from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from scripts.repair_mrag_metadata import PART_TITLES, repair_graph


class TestRepairMragMetadata(unittest.TestCase):
    def test_repair_graph_preserves_schema_v2_and_rebuilds_part_edges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            graph_path = Path(td) / "graph.gpickle"
            graph = nx.MultiDiGraph(schema_version=2, build_stats={"n_figures": 2})
            part_9 = f"part:{PART_TITLES['9']}"
            chapter_1 = "chapter:Chapter 1A. General"
            chapter_9 = "chapter:Chapter 9A. General"
            section_1 = "section:1A.01"
            graph.add_node(part_9, kind="Part", title=PART_TITLES["9"])
            graph.add_node(chapter_1, kind="Chapter", title="Chapter 1A. General", part=PART_TITLES["9"])
            graph.add_node(chapter_9, kind="Chapter", title="Chapter 9A. General", part=PART_TITLES["9"])
            graph.add_node(section_1, kind="Section", id="1A.01", chapter="Chapter 1A. General", part=PART_TITLES["9"])
            graph.add_edge(part_9, chapter_1, label="contains")
            graph.add_edge(part_9, chapter_9, label="contains")
            graph.add_edge(chapter_1, section_1, label="contains")
            with graph_path.open("wb") as handle:
                pickle.dump(graph, handle)

            report = repair_graph(graph_path, dry_run=False)

            with graph_path.open("rb") as handle:
                repaired = pickle.load(handle)
            self.assertEqual(repaired.graph, {"schema_version": 2, "build_stats": {"n_figures": 2}})
            self.assertEqual(repaired.nodes[chapter_1]["part"], PART_TITLES["1"])
            self.assertEqual(repaired.nodes[section_1]["part"], PART_TITLES["1"])
            self.assertTrue(repaired.has_edge(f"part:{PART_TITLES['1']}", chapter_1))
            self.assertFalse(repaired.has_edge(part_9, chapter_1))
            self.assertTrue(repaired.has_edge(part_9, chapter_9))
            self.assertEqual(sum(data.get("kind") == "Part" for _, data in repaired.nodes(data=True)), 9)
            self.assertEqual(report["removed_part_chapter_edges"], 1)
            self.assertEqual(report["added_part_chapter_edges"], 1)
            self.assertTrue(graph_path.with_name("graph.gpickle.partfix.bak").is_file())

            idempotent = repair_graph(graph_path, dry_run=False)
            self.assertEqual(idempotent["changed_attrs"], 0)
            self.assertEqual(idempotent["removed_part_chapter_edges"], 0)
            self.assertEqual(idempotent["added_part_chapter_edges"], 0)


if __name__ == "__main__":
    unittest.main()
