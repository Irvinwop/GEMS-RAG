from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .config import ModelConfig
from .types import ModelResult


OPENAI_COMPAT_DEFAULTS = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None, "api": "chat_completions"},
    "openai_compatible": {"api_key_env": "OPENAI_API_KEY", "base_url": None, "api": "chat_completions"},
    "xai": {"api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1", "api": "chat_completions"},
    "grok": {"api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1", "api": "chat_completions"},
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "api": "chat_completions",
    },
    "qwen_dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "api": "chat_completions",
    },
    "local_openai": {
        "api_key_env": "LOCAL_OPENAI_API_KEY",
        "base_url": "http://localhost:8000/v1",
        "allow_missing_api_key": True,
        "api": "chat_completions",
    },
}
LITELLM_PROVIDERS = {"litellm", "anthropic"}
KNOWN_MODEL_PROVIDERS = {"dry_run", *LITELLM_PROVIDERS, *OPENAI_COMPAT_DEFAULTS}
LLM_MODEL_PROVIDERS = KNOWN_MODEL_PROVIDERS - {"dry_run"}
PLACEHOLDER_MODEL_MARKERS = ("replace-with", "placeholder", "or-successor")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], Any]


class ModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> ModelResult:
        raise NotImplementedError

    def run_with_tools(self, prompt: str, tools: list[ToolSpec], *, max_rounds: int = 4) -> ModelResult:
        return ModelResult(
            provider=str(getattr(getattr(self, "config", None), "provider", "unknown")),
            model=str(getattr(getattr(self, "config", None), "model", "unknown")),
            output="",
            error="native tool calls are not supported by this model client",
        )


class DryRunModel(ModelClient):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> ModelResult:
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output=(
                "Direct Answer: DRY RUN - no paid model was called.\n"
                "Standards:\nGuidance:\nOptions:\nSupport:\n"
                "Citations:"
            ),
            raw={"dry_run": True, "prompt_chars": len(prompt)},
        )

    def run_with_tools(self, prompt: str, tools: list[ToolSpec], *, max_rounds: int = 4) -> ModelResult:
        available = {tool.name: tool for tool in tools}
        trace = []
        search_result: Any = None
        if max_rounds > 0 and "search" in available:
            arguments = {"query": prompt, "top_k": 6}
            search_result = available["search"].execute(arguments)
            trace.append({"id": "dry-search", "name": "search", "arguments": arguments, "result": search_result})
        if max_rounds > 1 and "open" in available:
            arguments = {"hit_ids": _tool_result_ids(search_result)}
            opened = available["open"].execute(arguments)
            trace.append({"id": "dry-open", "name": "open", "arguments": arguments, "result": opened})
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output=(
                "Direct Answer: DRY RUN - no paid model was called.\n"
                "Standards:\nGuidance:\nOptions:\nSupport:\n"
                "Citations:"
            ),
            raw={
                "dry_run": True,
                "prompt_chars": len(prompt),
                "native_tool_calls": True,
                "tool_calls": trace,
            },
        )


class OpenAICompatibleModel(ModelClient):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> ModelResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            return ModelResult(self.config.provider, self.config.model, "", error=f"openai package not installed: {exc}")

        api_key_env = _openai_compat_option(self.config, "api_key_env")
        api_key = self.config.options.get("api_key") or os.environ.get(str(api_key_env))
        if not api_key:
            if _allow_missing_api_key(self.config):
                api_key = "local"
            else:
                return ModelResult(self.config.provider, self.config.model, "", error=f"missing API key env var: {api_key_env}")
        base_url = _openai_compatible_base_url(self.config)
        client = OpenAI(api_key=api_key, base_url=base_url)
        api = _openai_compatible_api(self.config)
        try:
            if api == "responses":
                return self._generate_responses(client, prompt)
            return self._generate_chat_completions(client, prompt)
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return ModelResult(self.config.provider, self.config.model, "", error=repr(exc))

    def run_with_tools(self, prompt: str, tools: list[ToolSpec], *, max_rounds: int = 4) -> ModelResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            return ModelResult(self.config.provider, self.config.model, "", error=f"openai package not installed: {exc}")

        api_key_env = _openai_compat_option(self.config, "api_key_env")
        api_key = self.config.options.get("api_key") or os.environ.get(str(api_key_env))
        if not api_key:
            if _allow_missing_api_key(self.config):
                api_key = "local"
            else:
                return ModelResult(self.config.provider, self.config.model, "", error=f"missing API key env var: {api_key_env}")
        client = OpenAI(api_key=api_key, base_url=_openai_compatible_base_url(self.config))
        try:
            if _openai_compatible_api(self.config) == "responses":
                return self._run_responses_tools(client, prompt, tools, max_rounds=max_rounds)
            return self._run_chat_tools(client, prompt, tools, max_rounds=max_rounds)
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return ModelResult(self.config.provider, self.config.model, "", error=repr(exc))

    def _run_chat_tools(self, client, prompt: str, tools: list[ToolSpec], *, max_rounds: int) -> ModelResult:
        kwargs: dict[str, Any] = {"model": self.config.model}
        temperature = self.config.options.get("temperature", 0)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        max_tokens = self.config.options.get("max_tokens", 900)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        return _run_chat_tool_loop(
            self.config,
            prompt,
            tools,
            max_rounds=max_rounds,
            api="chat_completions",
            create=client.chat.completions.create,
            base_kwargs=kwargs,
        )

    def _run_responses_tools(self, client, prompt: str, tools: list[ToolSpec], *, max_rounds: int) -> ModelResult:
        tool_map = {tool.name: tool for tool in tools}
        schemas = [_responses_tool_schema(tool) for tool in tools]
        trace = []
        provider_calls = []
        next_input: Any = prompt
        previous_response_id = None
        tool_rounds = 0
        while True:
            force_final = tool_rounds >= max_rounds
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": next_input,
                "tools": schemas,
                "tool_choice": "none" if force_final else "auto",
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            max_output_tokens = self.config.options.get("max_output_tokens", self.config.options.get("max_tokens", 900))
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = int(max_output_tokens)
            temperature = self.config.options.get("temperature")
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            reasoning_effort = self.config.options.get("reasoning_effort", self.config.options.get("effort"))
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": str(reasoning_effort)}
            response = client.responses.create(**kwargs)
            calls = _responses_tool_calls(response)
            usage = _usage_payload(response)
            provider_calls.append(
                {
                    "id": getattr(response, "id", None),
                    "status": getattr(response, "status", None),
                    "api": "responses",
                    "usage": usage,
                }
            )
            if not calls:
                return _native_tool_result(
                    self.config,
                    output=_responses_output_text(response),
                    trace=trace,
                    provider_calls=provider_calls,
                )
            if force_final:
                return _native_tool_result(
                    self.config,
                    output=_responses_output_text(response),
                    trace=trace,
                    provider_calls=provider_calls,
                    error=f"native tool loop exceeded max_rounds={max_rounds}",
                )
            outputs = []
            for call in calls:
                record = _execute_tool_call(call, tool_map)
                trace.append(record)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": json.dumps(record["result"], ensure_ascii=False, default=str),
                    }
                )
            next_input = outputs
            previous_response_id = getattr(response, "id", None)
            tool_rounds += 1

    def _generate_chat_completions(self, client, prompt: str) -> ModelResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        temperature = self.config.options.get("temperature", 0)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        max_tokens = self.config.options.get("max_tokens", 900)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        response = client.chat.completions.create(**kwargs)
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output=response.choices[0].message.content or "",
            raw={"id": getattr(response, "id", None), "api": "chat_completions", "usage": _usage_payload(response)},
        )

    def _generate_responses(self, client, prompt: str) -> ModelResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": prompt,
        }
        max_output_tokens = self.config.options.get("max_output_tokens", self.config.options.get("max_tokens", 900))
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        temperature = self.config.options.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        reasoning_effort = self.config.options.get("reasoning_effort", self.config.options.get("effort"))
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": str(reasoning_effort)}
        response = client.responses.create(**kwargs)
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output=_responses_output_text(response),
            raw={
                "id": getattr(response, "id", None),
                "status": getattr(response, "status", None),
                "api": "responses",
                "usage": _usage_payload(response),
            },
        )


class LiteLLMModel(ModelClient):
    """Optional multi-provider model client for provider/model sweeps."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> ModelResult:
        try:
            import litellm
        except ImportError as exc:
            return ModelResult(self.config.provider, self.config.model, "", error=f"litellm package not installed: {exc}")

        kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.config.options.get("temperature", 0)),
            "max_tokens": int(self.config.options.get("max_tokens", 900)),
        }
        api_key_env = self.config.options.get("api_key_env")
        if api_key_env and "api_key" not in self.config.options:
            api_key = os.environ.get(str(api_key_env))
            if not api_key:
                return ModelResult(self.config.provider, self.config.model, "", error=f"missing API key env var: {api_key_env}")
            kwargs["api_key"] = api_key
        for key in ["api_base", "api_key", "custom_llm_provider"]:
            if key in self.config.options:
                kwargs[key] = self.config.options[key]
        try:
            response = litellm.completion(**kwargs)
            return ModelResult(
                provider=self.config.provider,
                model=self.config.model,
                output=response.choices[0].message.content or "",
                raw={"id": getattr(response, "id", None), "api": "litellm", "usage": _usage_payload(response)},
            )
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return ModelResult(self.config.provider, self.config.model, "", error=repr(exc))

    def run_with_tools(self, prompt: str, tools: list[ToolSpec], *, max_rounds: int = 4) -> ModelResult:
        try:
            import litellm
        except ImportError as exc:
            return ModelResult(self.config.provider, self.config.model, "", error=f"litellm package not installed: {exc}")

        base_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": float(self.config.options.get("temperature", 0)),
            "max_tokens": int(self.config.options.get("max_tokens", 900)),
        }
        api_key_env = self.config.options.get("api_key_env")
        if api_key_env and "api_key" not in self.config.options:
            api_key = os.environ.get(str(api_key_env))
            if not api_key:
                return ModelResult(self.config.provider, self.config.model, "", error=f"missing API key env var: {api_key_env}")
            base_kwargs["api_key"] = api_key
        for key in ["api_base", "api_key", "custom_llm_provider"]:
            if key in self.config.options:
                base_kwargs[key] = self.config.options[key]

        try:
            return _run_chat_tool_loop(
                self.config,
                prompt,
                tools,
                max_rounds=max_rounds,
                api="litellm",
                create=litellm.completion,
                base_kwargs=base_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return ModelResult(self.config.provider, self.config.model, "", error=repr(exc))


def build_model(config: ModelConfig) -> ModelClient:
    if config.provider == "dry_run":
        return DryRunModel(config)
    if config.provider in LITELLM_PROVIDERS:
        return LiteLLMModel(config)
    if config.provider in OPENAI_COMPAT_DEFAULTS:
        return OpenAICompatibleModel(config)
    raise ValueError(f"unknown model provider: {config.provider}")


def model_backend(config: ModelConfig) -> str:
    if config.provider == "dry_run":
        return "dry_run"
    if config.provider in LITELLM_PROVIDERS:
        return "litellm"
    if config.provider in OPENAI_COMPAT_DEFAULTS:
        return "openai_compatible"
    return "unknown"


def model_required_package(config: ModelConfig) -> str | None:
    backend = model_backend(config)
    if backend == "litellm":
        return "litellm"
    if backend == "openai_compatible":
        return "openai"
    return None


def model_api(config: ModelConfig) -> str:
    if config.provider == "dry_run":
        return "dry_run"
    if config.provider in OPENAI_COMPAT_DEFAULTS:
        return _openai_compatible_api(config)
    return model_backend(config)


def is_placeholder_model_name(model: str) -> bool:
    normalized = str(model).strip().lower()
    return any(marker in normalized for marker in PLACEHOLDER_MODEL_MARKERS)


def model_api_key_envs(config: ModelConfig) -> list[str]:
    if config.provider == "dry_run" or config.options.get("api_key"):
        return []
    if config.options.get("api_key_env"):
        return [str(config.options["api_key_env"])]
    if _allow_missing_api_key(config):
        return []
    if config.provider in OPENAI_COMPAT_DEFAULTS:
        return [str(_openai_compat_option(config, "api_key_env"))]
    if config.provider == "anthropic":
        return ["ANTHROPIC_API_KEY"]
    if config.provider == "litellm":
        return _litellm_key_envs(config)
    return []


def _litellm_key_envs(config: ModelConfig) -> list[str]:
    model = config.model.lower()
    api_base = str(config.options.get("api_base") or config.options.get("base_url") or "").lower()
    if model.startswith(("openai/", "gpt-", "o1", "o3", "o4", "o5")):
        return ["OPENAI_API_KEY"]
    if model.startswith(("anthropic/", "claude")):
        return ["ANTHROPIC_API_KEY"]
    if model.startswith(("xai/", "grok")):
        return ["XAI_API_KEY"]
    if "dashscope" in api_base or model.startswith(("qwen/", "dashscope/")):
        return ["DASHSCOPE_API_KEY"]
    if model.startswith(("hosted_vllm/", "openai/")) and api_base:
        return []
    return ["OPENAI_API_KEY"]


def _openai_compat_option(config: ModelConfig, key: str) -> Any:
    default = OPENAI_COMPAT_DEFAULTS.get(config.provider, OPENAI_COMPAT_DEFAULTS["openai_compatible"])
    return config.options.get(key, default.get(key))


def _openai_compatible_api(config: ModelConfig) -> str:
    api = config.options.get("api") or config.options.get("api_type") or _openai_compat_option(config, "api")
    normalized = str(api or "chat_completions").lower().replace("-", "_")
    if normalized in {"response", "responses_api"}:
        return "responses"
    return normalized


def _openai_compatible_base_url(config: ModelConfig) -> str | None:
    explicit = config.options.get("base_url") or config.options.get("api_base")
    if explicit:
        return str(explicit)
    base_url_env = config.options.get("base_url_env") or _openai_compat_option(config, "base_url_env")
    if base_url_env:
        value = os.environ.get(str(base_url_env))
        if value:
            return value
    base_url = _openai_compat_option(config, "base_url")
    return str(base_url) if base_url else None


def _allow_missing_api_key(config: ModelConfig) -> bool:
    default = OPENAI_COMPAT_DEFAULTS.get(config.provider, {})
    return bool(config.options.get("allow_missing_api_key", default.get("allow_missing_api_key", False)))


def _usage_payload(response: Any) -> dict[str, int] | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    input_tokens = _int_field(usage, "input_tokens", "prompt_tokens")
    output_tokens = _int_field(usage, "output_tokens", "completion_tokens")
    total_tokens = _int_field(usage, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    payload = {
        key: value
        for key, value in {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }.items()
        if value is not None
    }
    return payload or None


def _int_field(value: Any, *names: str) -> int | None:
    for name in names:
        raw = _field(value, name)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _responses_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    parts = []
    for item in getattr(response, "output", []) or []:
        content_items = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", []) or []
        for content in content_items:
            text = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _run_chat_tool_loop(
    config: ModelConfig,
    prompt: str,
    tools: list[ToolSpec],
    *,
    max_rounds: int,
    api: str,
    create: Callable[..., Any],
    base_kwargs: dict[str, Any],
) -> ModelResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_map = {tool.name: tool for tool in tools}
    schemas = [_chat_tool_schema(tool) for tool in tools]
    trace = []
    provider_calls = []
    tool_rounds = 0
    while True:
        force_final = tool_rounds >= max_rounds
        response = create(
            **base_kwargs,
            messages=list(messages),
            tools=schemas,
            tool_choice="none" if force_final else "auto",
        )
        message = response.choices[0].message
        calls = _chat_tool_calls(message)
        provider_calls.append(
            {
                "id": getattr(response, "id", None),
                "api": api,
                "usage": _usage_payload(response),
            }
        )
        if not calls:
            return _native_tool_result(
                config,
                output=str(_field(message, "content") or ""),
                trace=trace,
                provider_calls=provider_calls,
            )
        if force_final:
            return _native_tool_result(
                config,
                output=str(_field(message, "content") or ""),
                trace=trace,
                provider_calls=provider_calls,
                error=f"native tool loop exceeded max_rounds={max_rounds}",
            )
        messages.append(
            {
                "role": "assistant",
                "content": _field(message, "content"),
                "tool_calls": [_chat_tool_call_message(call) for call in calls],
            }
        )
        for call in calls:
            record = _execute_tool_call(call, tool_map)
            trace.append(record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(record["result"], ensure_ascii=False, default=str),
                }
            )
        tool_rounds += 1


def _chat_tool_schema(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _responses_tool_schema(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _responses_tool_calls(response: Any) -> list[dict[str, Any]]:
    calls = []
    for item in _field(response, "output") or []:
        if str(_field(item, "type") or "") != "function_call":
            continue
        calls.append(
            {
                "id": str(_field(item, "call_id") or _field(item, "id") or f"tool-call-{len(calls) + 1}"),
                "name": str(_field(item, "name") or ""),
                "arguments_json": _field(item, "arguments") or "{}",
            }
        )
    return calls


def _chat_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = []
    for item in _field(message, "tool_calls") or []:
        function = _field(item, "function")
        calls.append(
            {
                "id": str(_field(item, "id") or f"tool-call-{len(calls) + 1}"),
                "name": str(_field(function, "name") or ""),
                "arguments_json": _field(function, "arguments") or "{}",
            }
        )
    return calls


def _chat_tool_call_message(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": call["arguments_json"],
        },
    }


def _execute_tool_call(call: dict[str, Any], tools: dict[str, ToolSpec]) -> dict[str, Any]:
    error = None
    try:
        arguments = call.get("arguments_json", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must decode to an object")
        tool = tools.get(str(call.get("name") or ""))
        if tool is None:
            raise ValueError(f"unknown tool: {call.get('name')}")
        result = tool.execute(arguments)
    except Exception as exc:
        arguments = arguments if isinstance(locals().get("arguments"), dict) else {}
        error = repr(exc)
        result = {"error": error}
    return {
        "id": str(call.get("id") or ""),
        "name": str(call.get("name") or ""),
        "arguments": arguments,
        "result": result,
        "error": error,
    }


def _native_tool_result(
    config: ModelConfig,
    *,
    output: str,
    trace: list[dict[str, Any]],
    provider_calls: list[dict[str, Any]],
    error: str | None = None,
) -> ModelResult:
    usage_payloads = [call.get("usage") for call in provider_calls]
    usage, observed_calls = _aggregate_usage_payloads(usage_payloads)
    return ModelResult(
        provider=config.provider,
        model=config.model,
        output=output,
        raw={
            "api": provider_calls[-1].get("api") if provider_calls else None,
            "native_tool_calls": True,
            "tool_calls": trace,
            "model_calls": provider_calls,
            **({"usage": usage} if usage else {}),
            "usage_coverage": {
                "expected_calls": len(provider_calls),
                "observed_calls": observed_calls,
                "complete": observed_calls == len(provider_calls),
            },
        },
        error=error,
    )


def _aggregate_usage_payloads(payloads: list[Any]) -> tuple[dict[str, int], int]:
    fields = ["input_tokens", "output_tokens", "total_tokens"]
    totals = {field: 0 for field in fields}
    observed_fields: set[str] = set()
    observed_calls = 0
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        observed = False
        for field in fields:
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            totals[field] += int(value)
            observed_fields.add(field)
            observed = True
        if observed:
            observed_calls += 1
    usage = {field: totals[field] for field in fields if field in observed_fields}
    if "total_tokens" not in usage and "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage, observed_calls


def _tool_result_ids(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    hits = result.get("hits")
    if not isinstance(hits, list):
        return []
    return [str(hit["id"]) for hit in hits if isinstance(hit, dict) and hit.get("id") is not None]
