from __future__ import annotations

import json
import unittest
from pathlib import Path

from gems_rag.model_catalog import load_model_catalog
from gems_rag.models import is_placeholder_model_name


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

    def test_raganything_templates_pass_retrieval_budget(self) -> None:
        for path in TEMPLATE_CONFIGS + [ROOT / "configs" / "retriever-catalog.example.json"]:
            config = json.loads(path.read_text(encoding="utf-8"))
            retrievers = config["retrievers"]
            for retriever in retrievers:
                if not retriever["name"].startswith("raganything_"):
                    continue
                command = retriever["options"]["command"]
                with self.subTest(path=path.name, retriever=retriever["name"]):
                    self.assertEqual(command[command.index("--top-k") + 1], "{top_k}")
                    self.assertEqual(command[command.index("--chunk-top-k") + 1], "{top_k}")
                    self.assertIn("--only-need-context", command)

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

    def test_graphrag_templates_pass_retrieval_budget(self) -> None:
        for path in TEMPLATE_CONFIGS + [ROOT / "configs" / "retriever-catalog.example.json"]:
            config = json.loads(path.read_text(encoding="utf-8"))
            retrievers = config["retrievers"]
            for retriever in retrievers:
                if not retriever["name"].startswith("graphrag_"):
                    continue
                command = retriever["options"]["command"]
                with self.subTest(path=path.name, retriever=retriever["name"]):
                    self.assertEqual(command[command.index("--top-k") + 1], "{top_k}")

    def test_paperqa2_templates_pass_retrieval_budget(self) -> None:
        for path in TEMPLATE_CONFIGS + [ROOT / "configs" / "retriever-catalog.example.json"]:
            config = json.loads(path.read_text(encoding="utf-8"))
            retrievers = config["retrievers"]
            for retriever in retrievers:
                if not retriever["name"].startswith("paperqa2_"):
                    continue
                command = retriever["options"]["command"]
                with self.subTest(path=path.name, retriever=retriever["name"]):
                    self.assertEqual(command[command.index("--top-k") + 1], "{top_k}")

    def test_local_smoke_does_not_include_external_placeholders(self) -> None:
        config = json.loads((ROOT / "configs" / "smoke.local.json").read_text(encoding="utf-8"))
        self.assertNotIn("external_placeholder", {retriever["kind"] for retriever in config["retrievers"]})

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

    def test_openai_defaults_use_the_gpt_56_family_by_role(self) -> None:
        catalog = json.loads((ROOT / "configs" / "model-catalog.example.json").read_text(encoding="utf-8"))
        openai_answers = {
            model["model"]: model["size"]
            for model in catalog["models"]
            if model["provider"] == "openai"
            and "answer" in model.get("roles", [])
            and model.get("enabled", True)
        }
        graders = [
            model
            for model in catalog["models"]
            if model["provider"] == "openai" and "grader" in model.get("roles", []) and model.get("enabled", True)
        ]
        ablation = json.loads((ROOT / "configs" / "ablation.template.json").read_text(encoding="utf-8"))

        self.assertEqual(
            openai_answers,
            {"gpt-5.6-luna": "tiny", "gpt-5.6-terra": "small", "gpt-5.6-sol": "medium"},
        )
        self.assertEqual(len(graders), 1)
        self.assertEqual(graders[0]["model"], "gpt-5.6-sol")
        self.assertEqual(graders[0]["options"]["reasoning_effort"], "xhigh")
        self.assertEqual(ablation["models"][0]["model"], "gpt-5.6-terra")
        self.assertEqual(ablation["grader"]["model"], "gpt-5.6-sol")

    def test_model_catalog_marks_each_model_vision_capability(self) -> None:
        entries = load_model_catalog(ROOT / "configs" / "model-catalog.example.json")

        for entry in entries:
            with self.subTest(provider=entry.config.provider, model=entry.config.model):
                self.assertIs(entry.config.options["vision"], "vision" in entry.tags)


if __name__ == "__main__":
    unittest.main()
