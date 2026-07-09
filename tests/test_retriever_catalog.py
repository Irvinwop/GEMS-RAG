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
