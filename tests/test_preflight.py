from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from gem_rags.preflight import _external_command_check


class TestPreflightExternalCommand(unittest.TestCase):
    def test_external_check_command_override_is_used(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python", "adapter.py", "check", "--allow-missing-api-key"],
            returncode=0,
            stdout='{"runnable": true, "api_key_present": false, "api_key_usable": true}',
            stderr="",
        )
        with patch("gem_rags.preflight.subprocess.run", return_value=completed) as run:
            result = _external_command_check(
                ["python", "adapter.py", "query", "--question", "{question}"],
                check_external=True,
                timeout_s=5,
                check_command=["python", "adapter.py", "check", "--allow-missing-api-key"],
            )

        run.assert_called_once()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["check_command"], ["python", "adapter.py", "check", "--allow-missing-api-key"])
        self.assertEqual(result["problems"], [])

    def test_missing_key_without_usable_flag_blocks_credentials(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python", "adapter.py", "check"],
            returncode=2,
            stdout='{"runnable": false, "api_key_env": "OPENAI_API_KEY", "api_key_present": false}',
            stderr="",
        )
        with patch("gem_rags.preflight.subprocess.run", return_value=completed):
            result = _external_command_check(
                ["python", "adapter.py", "query", "--question", "{question}"],
                check_external=True,
                timeout_s=5,
                check_command=["python", "adapter.py", "check"],
            )

        self.assertEqual(result["status"], "blocked_by_credentials")
        self.assertEqual(result["problems"], ["missing API key env var: OPENAI_API_KEY"])


if __name__ == "__main__":
    unittest.main()
