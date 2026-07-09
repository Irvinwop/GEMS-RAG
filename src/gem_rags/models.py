from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

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


class ModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> ModelResult:
        raise NotImplementedError


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
