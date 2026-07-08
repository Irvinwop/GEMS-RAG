from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .config import ModelConfig
from .types import ModelResult


OPENAI_COMPAT_DEFAULTS = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "openai_compatible": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "xai": {"api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "grok": {"api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "qwen": {"api_key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    "qwen_dashscope": {"api_key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    "local_openai": {"api_key_env": "LOCAL_OPENAI_API_KEY", "base_url": "http://localhost:8000/v1", "allow_missing_api_key": True},
}
LITELLM_PROVIDERS = {"litellm", "anthropic"}
KNOWN_MODEL_PROVIDERS = {"dry_run", *LITELLM_PROVIDERS, *OPENAI_COMPAT_DEFAULTS}
LLM_MODEL_PROVIDERS = KNOWN_MODEL_PROVIDERS - {"dry_run"}


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
        base_url = self.config.options.get("base_url") or self.config.options.get("api_base") or _openai_compat_option(self.config, "base_url")
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(self.config.options.get("temperature", 0)),
                max_tokens=int(self.config.options.get("max_tokens", 900)),
            )
            return ModelResult(
                provider=self.config.provider,
                model=self.config.model,
                output=response.choices[0].message.content or "",
                raw={"id": getattr(response, "id", None)},
            )
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return ModelResult(self.config.provider, self.config.model, "", error=repr(exc))


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
                raw={"id": getattr(response, "id", None)},
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


def _allow_missing_api_key(config: ModelConfig) -> bool:
    default = OPENAI_COMPAT_DEFAULTS.get(config.provider, {})
    return bool(config.options.get("allow_missing_api_key", default.get("allow_missing_api_key", False)))
