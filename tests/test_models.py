from __future__ import annotations

import unittest
from unittest.mock import patch

from gem_rags.config import ModelConfig
from gem_rags.models import OpenAICompatibleModel, build_model, model_api_key_envs, model_backend, model_required_package


class TestModels(unittest.TestCase):
    def test_explicit_provider_backends(self) -> None:
        cases = {
            "openai": ("openai_compatible", "openai", ["OPENAI_API_KEY"]),
            "xai": ("openai_compatible", "openai", ["XAI_API_KEY"]),
            "grok": ("openai_compatible", "openai", ["XAI_API_KEY"]),
            "qwen": ("openai_compatible", "openai", ["DASHSCOPE_API_KEY"]),
            "anthropic": ("litellm", "litellm", ["ANTHROPIC_API_KEY"]),
        }
        for provider, (backend, package, envs) in cases.items():
            config = ModelConfig(provider=provider, model="model")
            self.assertEqual(model_backend(config), backend)
            self.assertEqual(model_required_package(config), package)
            self.assertEqual(model_api_key_envs(config), envs)

    def test_local_openai_does_not_require_api_key_env(self) -> None:
        config = ModelConfig(provider="local_openai", model="local")
        self.assertEqual(model_backend(config), "openai_compatible")
        self.assertEqual(model_api_key_envs(config), [])
        self.assertIsInstance(build_model(config), OpenAICompatibleModel)

    def test_explicit_api_key_env_overrides_provider_default(self) -> None:
        config = ModelConfig(provider="xai", model="grok", options={"api_key_env": "CUSTOM_XAI_KEY"})
        self.assertEqual(model_api_key_envs(config), ["CUSTOM_XAI_KEY"])

    def test_openai_compatible_reports_missing_provider_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = build_model(ModelConfig(provider="xai", model="grok-test")).generate("prompt")
        self.assertEqual(result.error, "missing API key env var: XAI_API_KEY")


if __name__ == "__main__":
    unittest.main()
