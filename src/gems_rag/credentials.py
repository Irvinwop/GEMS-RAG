from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"

CREDENTIAL_SPECS: dict[str, dict[str, Any]] = {
    "OPENAI_API_KEY": {"label": "OpenAI", "kind": "secret", "providers": ["openai", "paperqa2", "raganything"]},
    "ANTHROPIC_API_KEY": {"label": "Anthropic", "kind": "secret", "providers": ["anthropic"]},
    "XAI_API_KEY": {"label": "xAI / Grok", "kind": "secret", "providers": ["xai", "grok"]},
    "DASHSCOPE_API_KEY": {"label": "Qwen / DashScope", "kind": "secret", "providers": ["qwen"]},
    "LOCAL_OPENAI_API_KEY": {"label": "Local model API key (optional)", "kind": "secret", "providers": ["local_openai"]},
    "GRAPHRAG_API_KEY": {"label": "GraphRAG", "kind": "secret", "providers": ["graphrag"]},
    "OPENAI_BASE_URL": {"label": "OpenAI-compatible base URL", "kind": "url", "providers": ["openai", "paperqa2", "raganything"]},
    "DASHSCOPE_BASE_URL": {"label": "DashScope base URL", "kind": "url", "providers": ["qwen"]},
    "LOCAL_OPENAI_BASE_URL": {"label": "Local model base URL", "kind": "url", "providers": ["local_openai"]},
}


def load_local_env(path: Path = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    values = _read_env(path)
    for name, value in values.items():
        if name in CREDENTIAL_SPECS and (override or name not in os.environ):
            os.environ[name] = value
    return values


def credential_status(path: Path = DEFAULT_ENV_PATH) -> list[dict[str, Any]]:
    file_values = _read_env(path)
    rows = []
    for name, spec in CREDENTIAL_SPECS.items():
        process_value = os.getenv(name)
        file_value = file_values.get(name)
        source = "local_file" if file_value and process_value == file_value else ("environment" if process_value else ("local_file" if file_value else "unset"))
        rows.append(
            {
                "name": name,
                "label": spec["label"],
                "kind": spec["kind"],
                "providers": list(spec["providers"]),
                "configured": bool(process_value or file_values.get(name)),
                "source": source,
            }
        )
    return rows


def set_credential(name: str, value: str, path: Path = DEFAULT_ENV_PATH) -> dict[str, Any]:
    _validate_name(name)
    value = str(value).strip()
    if not value:
        return clear_credential(name, path)
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("credential values cannot contain newlines or NUL bytes")
    if CREDENTIAL_SPECS[name]["kind"] == "url" and not re.match(r"^https?://", value):
        raise ValueError("base URLs must start with http:// or https://")
    _update_env_file(path, name, value)
    os.environ[name] = value
    return next(row for row in credential_status(path) if row["name"] == name)


def clear_credential(name: str, path: Path = DEFAULT_ENV_PATH) -> dict[str, Any]:
    _validate_name(name)
    _update_env_file(path, name, None)
    os.environ.pop(name, None)
    return next(row for row in credential_status(path) if row["name"] == name)


def _validate_name(name: str) -> None:
    if name not in CREDENTIAL_SPECS:
        raise ValueError(f"unsupported credential: {name}")


def _read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        raw_value = raw_value.strip()
        try:
            value = json.loads(raw_value) if raw_value.startswith(('"', "'")) else raw_value
        except json.JSONDecodeError:
            value = raw_value.strip('"\'')
        values[name] = str(value)
    return values


def _update_env_file(path: Path, name: str, value: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    replacement = f"{name}={json.dumps(value)}" if value is not None else None
    updated = []
    replaced = False
    for line in lines:
        if pattern.match(line):
            if replacement is not None and not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(line)
    if replacement is not None and not replaced:
        updated.append(replacement)
    text = "\n".join(updated).rstrip() + ("\n" if updated else "")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)
