from __future__ import annotations

import json
import unittest
from pathlib import Path

from gem_rags.models import is_placeholder_model_name


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_CONFIGS = [
    ROOT / "configs" / "ablation.template.json",
    ROOT / "configs" / "external-rag.template.json",
    ROOT / "configs" / "external-rag.smoke.json",
    ROOT / "configs" / "external-rag.local-openai.smoke.json",
]


class TestConfigTemplates(unittest.TestCase):
    def test_command_backed_retrievers_have_check_commands(self) -> None:
        for path in TEMPLATE_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                for retriever in config["retrievers"]:
                    if retriever["kind"] != "external_command":
                        continue
                    options = retriever["options"]
                    self.assertIn("command", options, retriever["name"])
                    self.assertIn("check_command", options, retriever["name"])
                    self.assertIn("check", options["check_command"], retriever["name"])

    def test_raganything_templates_emit_json(self) -> None:
        for path in TEMPLATE_CONFIGS:
            config = json.loads(path.read_text(encoding="utf-8"))
            for retriever in config["retrievers"]:
                if not retriever["name"].startswith("raganything_"):
                    continue
                with self.subTest(path=path.name, retriever=retriever["name"]):
                    self.assertIn("--json", retriever["options"]["command"])

    def test_lightrag_templates_pass_retrieval_budget(self) -> None:
        for path in TEMPLATE_CONFIGS + [ROOT / "configs" / "retriever-catalog.example.json"]:
            config = json.loads(path.read_text(encoding="utf-8"))
            retrievers = config["retrievers"]
            for retriever in retrievers:
                if not retriever["name"].startswith("lightrag_"):
                    continue
                command = retriever["options"]["command"]
                with self.subTest(path=path.name, retriever=retriever["name"]):
                    self.assertEqual(command[command.index("--top-k") + 1], "{top_k}")
                    self.assertEqual(command[command.index("--chunk-top-k") + 1], "{top_k}")

    def test_ablation_template_includes_command_backed_vector_db(self) -> None:
        config = json.loads((ROOT / "configs" / "ablation.template.json").read_text(encoding="utf-8"))
        by_name = {retriever["name"]: retriever for retriever in config["retrievers"]}
        vector_command = by_name["qdrant_hash_vector_command"]

        self.assertIn("qdrant_hash_vector", by_name)
        self.assertIn("qdrant_hash_vector_command", by_name)
        self.assertEqual(vector_command["kind"], "external_command")
        self.assertIn("scripts/query_vector_db.py", vector_command["options"]["command"])

    def test_tracked_templates_do_not_ship_model_placeholders(self) -> None:
        for path in TEMPLATE_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                for model in config.get("models", []):
                    self.assertFalse(is_placeholder_model_name(model["model"]), model)
                grader = config.get("grader", {})
                if grader:
                    self.assertFalse(is_placeholder_model_name(grader["model"]), grader)


if __name__ == "__main__":
    unittest.main()
