from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gem_rags.retriever_catalog import load_retriever_catalog, load_retriever_specs_file, select_retriever_catalog, catalog_entries_to_retrievers_payload


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "defaults": {"top_k": 6},
                "retrievers": [
                    {
                        "name": "bm25",
                        "kind": "bm25",
                        "family": "local",
                        "modes": ["lexical"],
                        "tags": ["local"],
                    },
                    {
                        "name": "lightrag_naive_context",
                        "kind": "external_command",
                        "family": "lightrag",
                        "modes": ["naive"],
                        "tags": ["external", "text"],
                        "options": {
                            "command": [".venv/bin/python", "scripts/query_lightrag_index.py", "query", "--mode", "naive", "--question", "{question}"],
                            "check_command": [".venv/bin/python", "scripts/query_lightrag_index.py", "check"],
                        },
                    },
                    {
                        "name": "lightrag_hybrid_context",
                        "kind": "external_command",
                        "family": "lightrag",
                        "modes": ["hybrid"],
                        "tags": ["external", "text"],
                        "options": {
                            "command": [".venv/bin/python", "scripts/query_lightrag_index.py", "query", "--mode", "hybrid", "--question", "{question}"],
                            "check_command": [".venv/bin/python", "scripts/query_lightrag_index.py", "check"],
                        },
                    },
                    {
                        "name": "visrag_figures",
                        "kind": "external_command",
                        "family": "visrag",
                        "modes": ["figures"],
                        "tags": ["external", "visual"],
                        "enabled": False,
                        "options": {
                            "command": [".venv/bin/python", "scripts/query_visrag_index.py", "query", "--question", "{question}"],
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class TestRetrieverCatalog(unittest.TestCase):
    def test_manuscript_paper_algorithms_are_selectable(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "retriever-catalog.example.json"
        entries = load_retriever_catalog(catalog_path)
        kinds = {entry.config.name: entry.config.kind for entry in entries}

        self.assertEqual(
            {name: kinds.get(name) for name in [
                "sam_rag_adaptive_multimodal",
                "lpkg_planned_retrieval",
                "megarag_hybrid_context",
                "kg2rag_graph_guided",
                "m3kg_rag_paper_spec",
                "okh_rag_paper_spec",
            ]},
            {
                "sam_rag_adaptive_multimodal": "sam_rag",
                "lpkg_planned_retrieval": "lpkg",
                "megarag_hybrid_context": "external_command",
                "kg2rag_graph_guided": "kg2rag",
                "m3kg_rag_paper_spec": "m3kg_rag",
                "okh_rag_paper_spec": "okh_rag",
            },
        )

    def test_manuscript_gems_modes_are_explicit(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "retriever-catalog.example.json"
        entries = load_retriever_catalog(catalog_path)
        by_name = {entry.config.name: entry for entry in entries}
        expected = {
            "gems_dense_text": "dense",
            "gems_hybrid_text": "hybrid",
            "gems_multimodal_no_graph": "multimodal",
            "gems_full": "full",
            "gems_no_graph": "no_graph",
            "gems_no_visual": "no_visual",
            "gems_no_rule": "no_rule",
            "gems_no_hierarchy": "no_hierarchy",
        }

        self.assertTrue(set(expected).issubset(by_name))
        for name, mode in expected.items():
            with self.subTest(name=name):
                command = by_name[name].config.options["command"]
                check_command = by_name[name].config.options["check_command"]
                self.assertEqual(command[command.index("--mode") + 1], mode)
                self.assertEqual(check_command[check_command.index("--mode") + 1], mode)

    def test_catalog_filters_by_family_mode_and_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "retrievers.json"
            _write_catalog(catalog_path)
            entries = load_retriever_catalog(catalog_path)

        lightrag = select_retriever_catalog(entries, families=["lightrag"], modes=["hybrid"])
        disabled = select_retriever_catalog(entries, families=["visrag"])
        disabled_included = select_retriever_catalog(entries, families=["visrag"], include_disabled=True)

        self.assertEqual([entry.config.name for entry in lightrag], ["lightrag_hybrid_context"])
        self.assertEqual(disabled, [])
        self.assertEqual([entry.config.name for entry in disabled_included], ["visrag_figures"])
        self.assertEqual(lightrag[0].config.top_k, 6)

    def test_catalog_payload_round_trips_to_retriever_configs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_path = root / "retrievers.json"
            retrievers_path = root / "generated.json"
            _write_catalog(catalog_path)

            entries = select_retriever_catalog(load_retriever_catalog(catalog_path), tags=["external"])
            payload = catalog_entries_to_retrievers_payload(entries)
            retrievers_path.write_text(json.dumps(payload), encoding="utf-8")
            retrievers = load_retriever_specs_file(retrievers_path)

        self.assertEqual([retriever.name for retriever in retrievers], ["lightrag_naive_context", "lightrag_hybrid_context"])
        self.assertEqual(retrievers[0].kind, "external_command")
        self.assertIn("check_command", retrievers[0].options)


if __name__ == "__main__":
    unittest.main()
