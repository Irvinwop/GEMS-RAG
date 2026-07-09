from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gem_rags.config import ModelConfig
from gem_rags.models import OpenAICompatibleModel, build_model, model_api, model_api_key_envs, model_backend, model_required_package


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

    def test_openai_compatible_can_use_responses_api_with_reasoning_effort(self) -> None:
        calls = {}

        class FakeResponses:
            def create(self, **kwargs):
                calls["responses"] = kwargs
                return SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    output_text="answer text",
                    usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
                )

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                calls["client"] = {"api_key": api_key, "base_url": base_url}
                self.responses = FakeResponses()

        config = ModelConfig(
            provider="openai",
            model="gpt-5.5",
            options={"api": "responses", "max_output_tokens": 1200, "reasoning_effort": "xhigh", "temperature": None},
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
            result = build_model(config).generate("prompt")

        self.assertEqual(model_api(config), "responses")
        self.assertEqual(result.output, "answer text")
        self.assertEqual(result.raw["api"], "responses")
        self.assertEqual(result.raw["usage"], {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})
        self.assertEqual(calls["client"], {"api_key": "sk-test", "base_url": None})
        self.assertEqual(
            calls["responses"],
            {
                "model": "gpt-5.5",
                "input": "prompt",
                "max_output_tokens": 1200,
                "reasoning": {"effort": "xhigh"},
            },
        )

    def test_qwen_base_url_can_come_from_environment(self) -> None:
        calls = {}

        class FakeChatCompletions:
            def create(self, **kwargs):
                calls["chat"] = kwargs
                message = SimpleNamespace(content="qwen answer")
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(id="chat_1", choices=[choice], usage={"prompt_tokens": 13, "completion_tokens": 5})

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                calls["client"] = {"api_key": api_key, "base_url": base_url}
                self.chat = SimpleNamespace(completions=FakeChatCompletions())

        config = ModelConfig(provider="qwen", model="qwen3.7-plus", options={"max_tokens": 400, "temperature": 0})
        env = {
            "DASHSCOPE_API_KEY": "dashscope-key",
            "DASHSCOPE_BASE_URL": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict("os.environ", env, clear=True), patch("openai.OpenAI", FakeClient):
            result = build_model(config).generate("prompt")

        self.assertEqual(result.output, "qwen answer")
        self.assertEqual(result.raw["api"], "chat_completions")
        self.assertEqual(result.raw["usage"], {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18})
        self.assertEqual(calls["client"]["base_url"], "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(calls["chat"]["model"], "qwen3.7-plus")

    def test_local_openai_does_not_require_api_key_env(self) -> None:
        config = ModelConfig(provider="local_openai", model="local")
        self.assertEqual(model_backend(config), "openai_compatible")
        self.assertEqual(model_api(config), "chat_completions")
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
