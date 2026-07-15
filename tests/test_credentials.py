from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gems_rag.credentials import clear_credential, credential_status, load_local_env, set_credential


class TestCredentials(unittest.TestCase):
    def test_status_includes_model_provider_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            names = {row["name"] for row in credential_status(Path(td) / ".env") if row["kind"] == "secret"}

        self.assertEqual(
            names,
            {
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "XAI_API_KEY",
                "DASHSCOPE_API_KEY",
                "LOCAL_OPENAI_API_KEY",
            },
        )

    def test_set_status_load_and_clear_never_return_value(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            path = Path(td) / ".env"
            status = set_credential("OPENAI_API_KEY", "sk-test-value", path)
            text = path.read_text(encoding="utf-8")
            process_mode = stat.S_IMODE(path.stat().st_mode)
            os.environ.pop("OPENAI_API_KEY")
            loaded = load_local_env(path)
            rows = credential_status(path)
            cleared = clear_credential("OPENAI_API_KEY", path)

        self.assertTrue(status["configured"])
        self.assertNotIn("value", status)
        self.assertIn("sk-test-value", text)
        self.assertEqual(process_mode, 0o600)
        self.assertEqual(loaded["OPENAI_API_KEY"], "sk-test-value")
        self.assertNotIn("sk-test-value", repr(rows))
        self.assertEqual(next(row for row in rows if row["name"] == "OPENAI_API_KEY")["source"], "local_file")
        self.assertFalse(cleared["configured"])

    def test_preserves_unknown_env_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            path = Path(td) / ".env"
            path.write_text("# keep\nUNRELATED=value\n", encoding="utf-8")
            set_credential("ANTHROPIC_API_KEY", "key", path)
            clear_credential("ANTHROPIC_API_KEY", path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("# keep", text)
        self.assertIn("UNRELATED=value", text)

    def test_rejects_unknown_names_and_bad_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            with self.assertRaisesRegex(ValueError, "unsupported credential"):
                set_credential("EVIL", "value", path)
            with self.assertRaisesRegex(ValueError, "http"):
                set_credential("OPENAI_BASE_URL", "localhost:8000", path)
