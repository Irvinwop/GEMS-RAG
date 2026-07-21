from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gems_rag.config import ModelConfig
from gems_rag.models import (
    OpenAICompatibleModel,
    ToolSpec,
    build_model,
    model_api,
    model_api_key_envs,
    model_backend,
    model_required_package,
)


class TestModels(unittest.TestCase):
    def test_dry_run_image_input_is_reported_as_dry_run(self) -> None:
        model = build_model(ModelConfig(provider="dry_run", model="target-label", options={"vision": True}))

        result = model.generate_with_images("Inspect", ["/tmp/page.png"])

        self.assertEqual(result.raw["image_input"]["fallback_reason"], "dry_run")
        self.assertEqual(result.raw["image_input"]["attached_images"], 0)

    def test_openai_responses_attaches_local_images_as_input_image_blocks(self) -> None:
        calls = {}

        class FakeResponses:
            def create(self, **kwargs):
                calls["responses"] = kwargs
                return SimpleNamespace(
                    id="resp-vision",
                    status="completed",
                    output_text="visual answer",
                    usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
                )

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.responses = FakeResponses()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            config = ModelConfig(
                provider="openai",
                model="gpt-vision",
                options={"api": "responses", "vision": True, "max_output_tokens": 300},
            )
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
                result = build_model(config).generate_with_images("Inspect the page", [image_path])

        content = calls["responses"]["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "Inspect the page"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(result.raw["image_input"]["attached_images"], 1)
        self.assertEqual(result.raw["image_input"]["mode"], "attached")

    def test_openai_chat_attaches_local_images_as_image_url_blocks(self) -> None:
        calls = {}

        class FakeChatCompletions:
            def create(self, **kwargs):
                calls["chat"] = kwargs
                return SimpleNamespace(
                    id="chat-vision",
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="visual answer"),
                            finish_reason="length",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                )

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.chat = SimpleNamespace(completions=FakeChatCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "figure.jpg"
            image_path.write_bytes(b"\xff\xd8\xfffixture")
            config = ModelConfig(provider="qwen", model="qwen-vl", options={"vision": True})
            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
                result = build_model(config).generate_with_images("Inspect the figure", [str(image_path)])

        content = calls["chat"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Inspect the figure"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(result.raw["image_input"]["attached_images"], 1)
        self.assertEqual(result.raw["finish_reason"], "length")

    def test_litellm_attaches_local_images_with_openai_multimodal_format(self) -> None:
        calls = {}

        def completion(**kwargs):
            calls["completion"] = kwargs
            return SimpleNamespace(
                id="lite-vision",
                choices=[SimpleNamespace(message=SimpleNamespace(content="visual answer"))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
            )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            fake_litellm = SimpleNamespace(completion=completion)
            config = ModelConfig(provider="anthropic", model="anthropic/claude-vision", options={"vision": True})
            with patch.dict(sys.modules, {"litellm": fake_litellm}):
                result = build_model(config).generate_with_images("Inspect the page", [image_path])

        content = calls["completion"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Inspect the page"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(result.raw["image_input"]["attached_images"], 1)

    def test_litellm_can_omit_temperature_and_disable_adaptive_thinking(self) -> None:
        calls = {}

        def completion(**kwargs):
            calls["completion"] = kwargs
            return SimpleNamespace(
                id="lite-sonnet",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="grounded answer"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            )

        fake_litellm = SimpleNamespace(completion=completion)
        config = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-5",
            options={"temperature": None, "max_tokens": 128000, "thinking": "disabled"},
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = build_model(config).generate("Answer from the supplied context")

        self.assertNotIn("temperature", calls["completion"])
        self.assertEqual(calls["completion"]["max_tokens"], 128000)
        self.assertEqual(calls["completion"]["thinking"], {"type": "disabled"})
        self.assertEqual(result.output, "grounded answer")

    def test_image_input_is_explicitly_reported_when_vision_is_disabled(self) -> None:
        calls = {}

        class FakeChatCompletions:
            def create(self, **kwargs):
                calls["chat"] = kwargs
                return SimpleNamespace(
                    id="chat-text",
                    choices=[SimpleNamespace(message=SimpleNamespace(content="text answer"))],
                    usage=None,
                )

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.chat = SimpleNamespace(completions=FakeChatCompletions())

        config = ModelConfig(provider="local_openai", model="text-only")
        with patch("openai.OpenAI", FakeClient):
            result = build_model(config).generate_with_images("Use this evidence", ["/tmp/page.png"])

        self.assertEqual(calls["chat"]["messages"][0]["content"], "Use this evidence")
        self.assertEqual(result.raw["image_input"]["mode"], "text_only")
        self.assertEqual(result.raw["image_input"]["fallback_reason"], "vision_disabled")
        self.assertEqual(result.raw["image_input"]["attached_images"], 0)

    def test_dry_run_native_tool_loop_executes_search_and_open(self) -> None:
        calls = []

        def search(arguments):
            calls.append(("search", arguments))
            return {"hits": [{"id": "hit-1", "preview": "Section 2A.04"}]}

        def open_hits(arguments):
            calls.append(("open", arguments))
            return {"contexts": [{"id": "hit-1", "text": "The standard text."}]}

        tools = [
            ToolSpec(
                name="search",
                description="Search the corpus.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                execute=search,
            ),
            ToolSpec(
                name="open",
                description="Open search hits.",
                parameters={"type": "object", "properties": {"hit_ids": {"type": "array", "items": {"type": "string"}}}},
                execute=open_hits,
            ),
        ]

        result = build_model(ModelConfig(provider="dry_run", model="dry-run")).run_with_tools(
            "What does Section 2A.04 require?",
            tools,
            max_rounds=3,
        )

        self.assertEqual([name for name, _ in calls], ["search", "open"])
        self.assertEqual(calls[0][1]["query"], "What does Section 2A.04 require?")
        self.assertEqual(calls[1][1]["hit_ids"], ["hit-1"])
        self.assertEqual([item["name"] for item in result.raw["tool_calls"]], ["search", "open"])
        self.assertTrue(result.raw["native_tool_calls"])
        self.assertIn("DRY RUN", result.output)

    def test_openai_chat_native_tool_loop_executes_calls_and_aggregates_usage(self) -> None:
        requests = []
        executed = []

        def tool_call(call_id: str, name: str, arguments: str):
            return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))

        responses = [
            SimpleNamespace(
                id="chat-search",
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call("call-search", "search", '{"query":"2A.04","top_k":3}')]))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            ),
            SimpleNamespace(
                id="chat-open",
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call("call-open", "open", '{"hit_ids":["hit-1"]}')]))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            ),
            SimpleNamespace(
                id="chat-answer",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Final grounded answer", tool_calls=[]))],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=4, total_tokens=34),
            ),
        ]

        class FakeChatCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.chat = SimpleNamespace(completions=FakeChatCompletions())

        tools = [
            ToolSpec(
                "search",
                "Search.",
                {"type": "object"},
                lambda args: executed.append(("search", args))
                or {"hits": [{"id": "hit-1", "metadata": {"image_path": "/tmp/search-catalog-only.png"}}]},
            ),
            ToolSpec("open", "Open.", {"type": "object"}, lambda args: executed.append(("open", args)) or {"contexts": [{"id": "hit-1", "text": "standard"}]}),
        ]
        config = ModelConfig(provider="openai", model="gpt-tool", options={"max_tokens": 300, "temperature": 0})

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
            result = build_model(config).run_with_tools("Answer with tools", tools, max_rounds=4)

        self.assertEqual(result.output, "Final grounded answer")
        self.assertEqual([name for name, _ in executed], ["search", "open"])
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "search")
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "call-search")
        self.assertEqual(requests[2]["messages"][-1]["tool_call_id"], "call-open")
        self.assertEqual(result.raw["usage"], {"input_tokens": 60, "output_tokens": 9, "total_tokens": 69})
        self.assertEqual(result.raw["usage_coverage"], {"expected_calls": 3, "observed_calls": 3, "complete": True})
        self.assertEqual([call["name"] for call in result.raw["tool_calls"]], ["search", "open"])
        self.assertTrue(result.raw["native_tool_calls"])
        self.assertNotIn("image_input", result.raw)

    def test_openai_chat_native_tool_loop_attaches_images_returned_by_open(self) -> None:
        requests = []

        def tool_call(call_id: str, name: str, arguments: str):
            return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))

        responses = [
            SimpleNamespace(
                id="chat-open",
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call("call-open", "open", '{"hit_ids":["page-1"]}')]))],
                usage=None,
            ),
            SimpleNamespace(
                id="chat-answer",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Visual answer", tool_calls=[]))],
                usage=None,
            ),
        ]

        class FakeChatCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.chat = SimpleNamespace(completions=FakeChatCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            tools = [
                ToolSpec(
                    "open",
                    "Open.",
                    {"type": "object"},
                    lambda _args: {"evidence": [{"id": "page-1", "metadata": {"image_path": str(image_path)}}]},
                )
            ]
            config = ModelConfig(provider="openai", model="gpt-vision", options={"vision": True})
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
                result = build_model(config).run_with_tools("Answer with tools", tools, max_rounds=2)

        visual_message = requests[1]["messages"][-1]
        tool_message = requests[1]["messages"][-2]
        self.assertEqual(visual_message["role"], "user")
        self.assertNotIn(str(image_path), tool_message["content"])
        self.assertIn("visual evidence handled separately", tool_message["content"])
        self.assertEqual(visual_message["content"][1]["type"], "image_url")
        self.assertTrue(visual_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(result.raw["image_input"]["attached_images"], 1)

    def test_openai_responses_native_tool_loop_attaches_images_returned_by_open(self) -> None:
        requests = []
        responses = [
            SimpleNamespace(
                id="resp-open",
                status="completed",
                output=[SimpleNamespace(type="function_call", call_id="call-open", name="open", arguments='{"hit_ids":["page-1"]}')],
                output_text="",
                usage=None,
            ),
            SimpleNamespace(
                id="resp-answer",
                status="completed",
                output=[],
                output_text="Visual answer",
                usage=None,
            ),
        ]

        class FakeResponses:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.responses = FakeResponses()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            tools = [
                ToolSpec(
                    "open",
                    "Open.",
                    {"type": "object"},
                    lambda _args: {"evidence": [{"id": "page-1", "metadata": {"image_path": str(image_path)}}]},
                )
            ]
            config = ModelConfig(provider="openai", model="gpt-vision", options={"api": "responses", "vision": True})
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
                result = build_model(config).run_with_tools("Answer with tools", tools, max_rounds=2)

        continuation = requests[1]["input"]
        self.assertEqual(continuation[0]["type"], "function_call_output")
        self.assertNotIn(str(image_path), continuation[0]["output"])
        self.assertEqual(continuation[1]["role"], "user")
        self.assertEqual(continuation[1]["content"][1]["type"], "input_image")
        self.assertTrue(continuation[1]["content"][1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(result.raw["image_input"]["attached_images"], 1)

    def test_openai_responses_native_tool_loop_uses_provider_continuations(self) -> None:
        requests = []
        executed = []
        responses = [
            SimpleNamespace(
                id="resp-search",
                status="completed",
                output=[SimpleNamespace(type="function_call", call_id="call-search", name="search", arguments='{"query":"warning signs","top_k":2}')],
                output_text="",
                usage=SimpleNamespace(input_tokens=11, output_tokens=2, total_tokens=13),
            ),
            SimpleNamespace(
                id="resp-open",
                status="completed",
                output=[SimpleNamespace(type="function_call", call_id="call-open", name="open", arguments='{"hit_ids":["hit-2"]}')],
                output_text="",
                usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
            ),
            SimpleNamespace(
                id="resp-answer",
                status="completed",
                output=[],
                output_text="Responses grounded answer",
                usage=SimpleNamespace(input_tokens=13, output_tokens=4, total_tokens=17),
            ),
        ]

        class FakeResponses:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        class FakeClient:
            def __init__(self, *, api_key, base_url=None):
                self.responses = FakeResponses()

        tools = [
            ToolSpec("search", "Search.", {"type": "object"}, lambda args: executed.append(("search", args)) or {"hits": [{"id": "hit-2"}]}),
            ToolSpec("open", "Open.", {"type": "object"}, lambda args: executed.append(("open", args)) or {"contexts": [{"id": "hit-2", "text": "guidance"}]}),
        ]
        config = ModelConfig(
            provider="openai",
            model="gpt-responses-tool",
            options={"api": "responses", "max_output_tokens": 400, "reasoning_effort": "high"},
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True), patch("openai.OpenAI", FakeClient):
            result = build_model(config).run_with_tools("Answer through Responses tools", tools, max_rounds=4)

        self.assertEqual(result.output, "Responses grounded answer")
        self.assertEqual([name for name, _ in executed], ["search", "open"])
        self.assertEqual(requests[0]["tools"][0]["name"], "search")
        self.assertEqual(requests[1]["previous_response_id"], "resp-search")
        self.assertEqual(requests[1]["input"][0]["type"], "function_call_output")
        self.assertEqual(requests[1]["input"][0]["call_id"], "call-search")
        self.assertEqual(requests[2]["previous_response_id"], "resp-open")
        self.assertEqual(requests[2]["input"][0]["call_id"], "call-open")
        self.assertEqual(result.raw["usage"], {"input_tokens": 36, "output_tokens": 9, "total_tokens": 45})
        self.assertEqual([call["name"] for call in result.raw["tool_calls"]], ["search", "open"])
        self.assertEqual(result.raw["api"], "responses")

    def test_litellm_native_tool_loop_executes_provider_tool_calls(self) -> None:
        requests = []
        executed = []
        tool_call = SimpleNamespace(
            id="litellm-search",
            type="function",
            function=SimpleNamespace(name="search", arguments='{"query":"signals","top_k":4}'),
        )
        responses = [
            SimpleNamespace(
                id="lite-search",
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))],
                usage=SimpleNamespace(prompt_tokens=15, completion_tokens=2, total_tokens=17),
            ),
            SimpleNamespace(
                id="lite-answer",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Claude grounded answer", tool_calls=[]))],
                usage=SimpleNamespace(prompt_tokens=16, completion_tokens=5, total_tokens=21),
            ),
        ]

        def completion(**kwargs):
            requests.append(kwargs)
            return responses.pop(0)

        fake_litellm = SimpleNamespace(completion=completion)
        tools = [
            ToolSpec("search", "Search.", {"type": "object"}, lambda args: executed.append(args) or {"hits": [{"id": "hit-a"}]}),
        ]
        config = ModelConfig(provider="anthropic", model="anthropic/claude-haiku", options={"temperature": 0, "max_tokens": 300})

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = build_model(config).run_with_tools("Use native tools", tools, max_rounds=3)

        self.assertEqual(result.output, "Claude grounded answer")
        self.assertEqual(executed[0]["query"], "signals")
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "search")
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "litellm-search")
        self.assertEqual(result.raw["usage"], {"input_tokens": 31, "output_tokens": 7, "total_tokens": 38})
        self.assertEqual(result.raw["api"], "litellm")
        self.assertTrue(result.raw["native_tool_calls"])

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
