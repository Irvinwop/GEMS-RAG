from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalAdapterOptions(unittest.TestCase):
    def test_lightrag_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_raganything_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_paperqa_sets_dummy_local_key(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._ensure_api_key(args)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local")

    def test_graphrag_applies_dummy_local_key_to_subprocess_env(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            env = mod._env(repo)
        args = argparse.Namespace(api_key_env="GRAPHRAG_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._apply_local_api_key(args, env)
            self.assertEqual(env["GRAPHRAG_API_KEY"], "local")


if __name__ == "__main__":
    unittest.main()
