from __future__ import annotations

import re
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from .config import RagBackendConfig, RetrieverConfig, rag_backend_to_dict

RAG_BACKEND_FAMILIES = {
    "graphrag",
    "hipporag",
    "lightrag",
    "megarag",
    "paperqa2",
    "raganything",
}

RAG_BACKEND_PRESETS: dict[str, RagBackendConfig] = {
    "openai": RagBackendConfig(),
    "local_openai": RagBackendConfig(
        provider="local_openai",
        api_key_env="LOCAL_OPENAI_API_KEY",
        base_url="http://localhost:8000/v1",
        allow_missing_api_key=True,
        chat_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        vision_model="qwen2.5vl:7b",
        reasoning_effort="none",
    ),
}
RAG_BACKEND_LABELS = {"openai": "OpenAI", "local_openai": "Local / compatible"}

_VALUE_FLAGS = {
    "--api-key-env",
    "--base-url",
    "--llm-model",
    "--embedding-model",
    "--embedding-dim",
    "--vision-model",
    "--embedding",
    "--llm",
    "--summary-llm",
    "--reasoning-effort",
}
_BOOLEAN_FLAGS = {
    "--allow-missing-api-key",
    "--entity-extraction-json",
    "--no-entity-extraction-json",
}
_GLOBAL_OPTION_FAMILIES = {"graphrag", "hipporag", "megarag", "paperqa2"}
_REASONING_EFFORT_FAMILIES = {"graphrag", "hipporag", "lightrag", "megarag", "raganything"}


def rag_backend_presets_payload() -> list[dict[str, Any]]:
    return [
        {**rag_backend_to_dict(profile), "label": RAG_BACKEND_LABELS[provider]}
        for provider, profile in RAG_BACKEND_PRESETS.items()
    ]


def rag_backend_from_payload(value: Any) -> RagBackendConfig:
    raw = value if isinstance(value, dict) else {}
    provider = str(raw.get("provider") or "openai").strip()
    try:
        preset = RAG_BACKEND_PRESETS[provider]
    except KeyError as exc:
        raise ValueError(f"unsupported RAG backend provider: {provider}") from exc

    base_url_value = raw.get("base_url", preset.base_url)
    base_url = str(base_url_value).strip() if base_url_value not in {None, ""} else None
    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RAG backend URL must be an absolute http:// or https:// URL")

    backend = replace(
        preset,
        base_url=base_url,
        chat_model=_model_name(raw.get("chat_model"), preset.chat_model, "chat_model"),
        embedding_model=_model_name(
            raw.get("embedding_model"), preset.embedding_model, "embedding_model"
        ),
        embedding_dim=_bounded_int(
            raw.get("embedding_dim", preset.embedding_dim), 1, 65536, "embedding_dim"
        ),
        vision_model=_model_name(raw.get("vision_model"), preset.vision_model, "vision_model"),
        reasoning_effort=_reasoning_effort(
            raw.get("reasoning_effort", preset.reasoning_effort)
        ),
    )
    if backend.provider == "local_openai" and not backend.base_url:
        raise ValueError("local RAG backend requires a base URL")
    return backend


def configure_retriever_backend(
    config: RetrieverConfig,
    family: str,
    backend: RagBackendConfig,
) -> RetrieverConfig:
    if config.kind != "external_command" or family not in RAG_BACKEND_FAMILIES:
        return config
    options = dict(config.options)
    for key in ("command", "check_command"):
        command = options.get(key)
        if isinstance(command, list | tuple):
            options[key] = backend_command([str(part) for part in command], family, backend)
    return replace(config, options=options)


def backend_command(command: list[str], family: str, backend: RagBackendConfig) -> list[str]:
    if family not in RAG_BACKEND_FAMILIES:
        return list(command)
    cleaned = _strip_backend_options(command)
    script_index = next(
        (index for index, part in enumerate(cleaned) if part.endswith(f"query_{family}_index.py")),
        None,
    )
    if family == "raganything":
        script_index = next(
            (index for index, part in enumerate(cleaned) if part.endswith("query_raganything_index.py")),
            script_index,
        )
    if family == "paperqa2":
        script_index = next(
            (index for index, part in enumerate(cleaned) if part.endswith("query_paperqa_index.py")),
            script_index,
        )
    if script_index is None:
        return cleaned

    common = ["--api-key-env", backend.api_key_env]
    if backend.base_url:
        common.extend(["--base-url", backend.base_url])
    if backend.allow_missing_api_key:
        common.append("--allow-missing-api-key")
    if family in _REASONING_EFFORT_FAMILIES and backend.reasoning_effort:
        common.extend(["--reasoning-effort", backend.reasoning_effort])
    if family in {"lightrag", "raganything"} and backend.provider == "local_openai":
        common.append("--entity-extraction-json")

    subcommand_index = script_index + 1
    subcommand = cleaned[subcommand_index] if subcommand_index < len(cleaned) else ""
    model_options: list[str] = []
    if family in {"lightrag", "raganything"}:
        model_options.extend(
            [
                "--llm-model",
                backend.chat_model,
                "--embedding-model",
                backend.embedding_model,
                "--embedding-dim",
                str(backend.embedding_dim),
            ]
        )
    if family == "raganything":
        model_options.extend(["--vision-model", backend.vision_model])
    if family == "hipporag":
        model_options.extend(
            ["--llm-model", backend.chat_model, "--embedding-model", backend.embedding_model]
        )
    if family == "megarag":
        model_options.extend(
            [
                "--llm-model",
                backend.chat_model,
                "--vision-model",
                backend.vision_model,
            ]
        )
    if family == "paperqa2":
        model_options.extend(["--embedding", backend.embedding_model])
        if subcommand == "query":
            model_options.extend(
                [
                    "--llm",
                    backend.chat_model,
                    "--summary-llm",
                    backend.chat_model,
                ]
            )

    if family == "paperqa2":
        return [*cleaned[: subcommand_index], *common, *cleaned[subcommand_index:], *model_options]
    if family in _GLOBAL_OPTION_FAMILIES:
        return [*cleaned[: subcommand_index], *common, *model_options, *cleaned[subcommand_index:]]
    return [*cleaned, *common, *model_options]


def _strip_backend_options(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(command):
        part = command[index]
        if part in _BOOLEAN_FLAGS:
            index += 1
            continue
        if part in _VALUE_FLAGS:
            index += 2
            continue
        if any(part.startswith(f"{flag}=") for flag in _VALUE_FLAGS):
            index += 1
            continue
        cleaned.append(part)
        index += 1
    return cleaned


def _model_name(value: Any, default: str, field: str) -> str:
    model = str(default if value in {None, ""} else value).strip()
    if not model or len(model) > 200 or re.search(r"[\r\n\x00]", model):
        raise ValueError(f"invalid RAG backend {field}")
    return model


def _bounded_int(value: Any, minimum: int, maximum: int, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return number


def _reasoning_effort(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    effort = str(value).strip().lower()
    if effort not in {"none", "low", "medium", "high"}:
        raise ValueError("RAG backend reasoning_effort must be none, low, medium, or high")
    return effort
