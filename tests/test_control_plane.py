from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag.control_plane import ControlPlane, JobManager, _retriever_for_ingestion
from gems_rag.config import RetrieverConfig


class TestControlPlane(unittest.TestCase):
    def test_materialize_builds_valid_config_and_plan(self) -> None:
        control = ControlPlane()
        result = control.materialize(
            {
                "name": "GUI Test",
                "retrievers": ["bm25"],
                "models": ["local_openai:local-small"],
                "context_modes": ["injected", "tool_native"],
                "grader_mode": "gpt_pro",
                "limit": 2,
                "top_k": 4,
                "dry_run": True,
            }
        )
        config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["grader_mode"], "gpt_pro")
        self.assertEqual(result["plan"]["estimates"]["rows"], 4)
        self.assertEqual(config["grader"]["provider"], "heuristic")
        self.assertEqual(config["retrievers"][0]["top_k"], 4)

    def test_native_ingestion_is_added_only_to_supported_commands(self) -> None:
        base = RetrieverConfig(
            name="paper",
            kind="external_command",
            options={"command": ["python", "adapter.py", "query"], "check_command": ["python", "adapter.py", "check"]},
        )
        native = _retriever_for_ingestion(base, "paperqa2", "native_pdf")
        shared = _retriever_for_ingestion(base, "dpr", "native_pdf")

        self.assertEqual(native.options["command"][-2:], ["--ingestion-mode", "native_pdf"])
        self.assertEqual(shared.options, base.options)

    def test_job_manager_rejects_arbitrary_actions_and_outside_configs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manager = JobManager(root)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                manager.start({"action": "shell", "config_path": "/tmp/config.json"})
            outside = Path(td).parent / "outside-control-plane.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "project JSON"):
                    manager.start({"action": "preflight", "config_path": str(outside)})
            finally:
                outside.unlink(missing_ok=True)

    def test_credential_api_returns_status_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {}, clear=True):
            control = ControlPlane()
            control.env_path = Path(td) / ".env"
            status = control.set_credential({"name": "XAI_API_KEY", "value": "xai-secret"})

        self.assertTrue(status["configured"])
        self.assertNotIn("xai-secret", repr(status))

    def test_grade_upload_is_written_to_ignored_import_area(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs = root / "runs" / "sample" / "runs.jsonl"
            runs.parent.mkdir(parents=True)
            runs.write_text("{}\n", encoding="utf-8")
            control = object.__new__(ControlPlane)
            control.root = root.resolve()
            payload = base64.b64encode(b'{"row_id":"row"}\n').decode()
            with patch("gems_rag.control_plane.import_pro_grades", return_value={"ok": True}) as importer:
                result = control.import_grades(
                    {
                        "runs": str(runs),
                        "grades_filename": "grades.jsonl",
                        "grades_base64": payload,
                    }
                )
            uploaded = importer.call_args.args[1]

        self.assertTrue(result["ok"])
        self.assertEqual(uploaded.suffix, ".jsonl")
        self.assertIn("data/working/gui/imports", str(uploaded))
