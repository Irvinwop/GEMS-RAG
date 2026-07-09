from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ModelConfig


@dataclass(frozen=True)
class ModelCatalogEntry:
    config: ModelConfig
    size: str
    roles: tuple[str, ...]
    tags: tuple[str, ...] = ()
    pricing: dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    name: str | None = None
    notes: str | None = None


def load_model_catalog(path: Path) -> list[ModelCatalogEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        defaults: dict[str, Any] = {}
        models = raw
    elif isinstance(raw, dict):
        defaults = dict(raw.get("defaults", {}))
        models = raw.get("models", [])
    else:
        raise ValueError("model catalog must be a JSON object or list")
    if not isinstance(models, list):
        raise ValueError("model catalog must contain a models list")

    entries = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError(f"model catalog entry must be an object: {item!r}")
        entries.append(_catalog_entry(item, defaults))
    return entries


def select_model_catalog(
    entries: list[ModelCatalogEntry],
    *,
    providers: list[str] | None = None,
    sizes: list[str] | None = None,
    roles: list[str] | None = None,
    tags: list[str] | None = None,
    include_disabled: bool = False,
) -> list[ModelCatalogEntry]:
    provider_set = set(providers or [])
    size_set = set(sizes or [])
    role_set = set(roles or [])
    tag_set = set(tags or [])
    selected = []
    for entry in entries:
        if not include_disabled and not entry.enabled:
            continue
        if provider_set and entry.config.provider not in provider_set:
            continue
        if size_set and entry.size not in size_set:
            continue
        if role_set and role_set.isdisjoint(entry.roles):
            continue
        if tag_set and not tag_set.issubset(set(entry.tags)):
            continue
        selected.append(entry)
    return selected


def model_config_to_spec(config: ModelConfig) -> str:
    parts = [f"{config.provider}:{config.model}"]
    for key in sorted(config.options):
        parts.append(f"{key}={_format_option_value(config.options[key])}")
    return ",".join(parts)


def render_model_specs(entries: list[ModelCatalogEntry], *, include_comments: bool = True) -> str:
    lines = []
    for entry in entries:
        line = model_config_to_spec(entry.config)
        if include_comments:
            labels = [f"size={entry.size}", f"roles={','.join(entry.roles)}"]
            if entry.tags:
                labels.append(f"tags={','.join(entry.tags)}")
            line = f"{line}  # {'; '.join(labels)}"
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def catalog_entries_to_models_payload(entries: list[ModelCatalogEntry]) -> dict[str, Any]:
    return {
        "models": [
            {
                "provider": entry.config.provider,
                "model": entry.config.model,
                "options": entry.config.options,
                **({"pricing": entry.pricing} if entry.pricing else {}),
                "metadata": {
                    "name": entry.name,
                    "size": entry.size,
                    "roles": list(entry.roles),
                    "tags": list(entry.tags),
                    "enabled": entry.enabled,
                    "notes": entry.notes,
                },
            }
            for entry in entries
        ]
    }


def catalog_pricing_payload(entries: list[ModelCatalogEntry]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[dict[str, float]]] = {}
    pricing: dict[str, dict[str, float]] = {}
    for entry in entries:
        if not entry.pricing:
            continue
        model_price = dict(entry.pricing)
        pricing[f"{entry.config.provider}:{entry.config.model}"] = model_price
        by_model.setdefault(entry.config.model, []).append(model_price)
    for model, prices in by_model.items():
        if len(prices) == 1:
            pricing.setdefault(model, prices[0])
    return pricing


def pricing_coverage_for_config(
    config: ExperimentConfig,
    model_pricing: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    requirements: list[tuple[str, str, str]] = []
    if not config.dry_run:
        requirements.extend(
            ("answer", model.provider, model.model)
            for model in config.models
            if model.provider != "dry_run"
        )
        if config.grader.provider != "heuristic":
            requirements.append(("judge", config.grader.provider, config.grader.model))

    checks = []
    missing = []
    seen: set[tuple[str, str, str]] = set()
    for role, provider, model in requirements:
        key = (role, provider, model)
        if key in seen:
            continue
        seen.add(key)
        pricing = _resolve_pricing(model_pricing or {}, provider=provider, model=model)
        ok = _complete_pricing(pricing)
        check = {
            "role": role,
            "provider": provider,
            "model": model,
            "pricing": pricing,
            "ok": ok,
        }
        checks.append(check)
        if not ok:
            missing.append(check)
    return {
        "ok": not missing,
        "required_models": len(checks),
        "priced_models": len(checks) - len(missing),
        "checks": checks,
        "missing": missing,
    }


def _resolve_pricing(
    model_pricing: dict[str, dict[str, float]],
    *,
    provider: str,
    model: str,
) -> dict[str, float] | None:
    return model_pricing.get(f"{provider}:{model}") or model_pricing.get(model)


def _complete_pricing(pricing: dict[str, float] | None) -> bool:
    if not pricing:
        return False
    total = _numeric_value(pricing, "total_per_1m", "total_usd_per_1m")
    if total is not None:
        return True
    input_price = _numeric_value(pricing, "input_per_1m", "input_usd_per_1m", "prompt_per_1m", "prompt_usd_per_1m")
    output_price = _numeric_value(pricing, "output_per_1m", "output_usd_per_1m", "completion_per_1m", "completion_usd_per_1m")
    return input_price is not None and output_price is not None


def _numeric_value(pricing: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = pricing.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)) and float(value) >= 0:
            return float(value)
    return None


def _catalog_entry(item: dict[str, Any], defaults: dict[str, Any]) -> ModelCatalogEntry:
    provider = str(item.get("provider") or "").strip()
    model = str(item.get("model") or "").strip()
    if not provider or not model:
        raise ValueError(f"model catalog entry must include provider and model: {item!r}")
    default_options = dict(defaults.get("options", {}))
    provider_options = _provider_options(defaults, provider)
    options = {**default_options, **provider_options, **dict(item.get("options", {}))}
    return ModelCatalogEntry(
        config=ModelConfig(provider=provider, model=model, options=options),
        size=str(item.get("size") or "unspecified"),
        roles=_string_tuple(item.get("roles", item.get("role", "answer"))),
        tags=_string_tuple(item.get("tags", ())),
        pricing=_pricing(item, defaults, provider),
        enabled=bool(item.get("enabled", True)),
        name=str(item["name"]) if item.get("name") is not None else None,
        notes=str(item["notes"]) if item.get("notes") is not None else None,
    )


def _provider_options(defaults: dict[str, Any], provider: str) -> dict[str, Any]:
    provider_defaults = defaults.get("provider_options", defaults.get("providers", {}))
    if not isinstance(provider_defaults, dict):
        return {}
    raw = provider_defaults.get(provider, {})
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("options"), dict):
        return dict(raw["options"])
    return {key: value for key, value in raw.items() if key != "pricing"}


def _pricing(item: dict[str, Any], defaults: dict[str, Any], provider: str) -> dict[str, float]:
    pricing: dict[str, float] = {}
    default_pricing = defaults.get("pricing")
    if isinstance(default_pricing, dict):
        pricing.update(_numeric_pricing(default_pricing))
    provider_defaults = defaults.get("provider_options", defaults.get("providers", {}))
    if isinstance(provider_defaults, dict):
        raw_provider = provider_defaults.get(provider, {})
        if isinstance(raw_provider, dict) and isinstance(raw_provider.get("pricing"), dict):
            pricing.update(_numeric_pricing(raw_provider["pricing"]))
    if isinstance(item.get("pricing"), dict):
        pricing.update(_numeric_pricing(item["pricing"]))
    return pricing


def _numeric_pricing(raw: dict[str, Any]) -> dict[str, float]:
    pricing = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"pricing value must be a finite non-negative number: {key}={value!r}")
            pricing[str(key)] = numeric
    return pricing


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list | tuple | set):
        items = value
    else:
        items = [value]
    return tuple(str(item).strip() for item in items if str(item).strip())


def _format_option_value(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)
