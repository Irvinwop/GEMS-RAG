from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RetrieverConfig


@dataclass(frozen=True)
class RetrieverCatalogEntry:
    config: RetrieverConfig
    family: str
    modes: tuple[str, ...]
    tags: tuple[str, ...] = ()
    enabled: bool = True
    notes: str | None = None


def load_retriever_catalog(path: Path) -> list[RetrieverCatalogEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        defaults: dict[str, Any] = {}
        retrievers = raw
    elif isinstance(raw, dict):
        defaults = dict(raw.get("defaults", {}))
        retrievers = raw.get("retrievers", [])
    else:
        raise ValueError("retriever catalog must be a JSON object or list")
    if not isinstance(retrievers, list):
        raise ValueError("retriever catalog must contain a retrievers list")

    entries = []
    for item in retrievers:
        if not isinstance(item, dict):
            raise ValueError(f"retriever catalog entry must be an object: {item!r}")
        entries.append(_catalog_entry(item, defaults))
    return entries


def select_retriever_catalog(
    entries: list[RetrieverCatalogEntry],
    *,
    families: list[str] | None = None,
    modes: list[str] | None = None,
    tags: list[str] | None = None,
    include_disabled: bool = False,
) -> list[RetrieverCatalogEntry]:
    family_set = set(families or [])
    mode_set = set(modes or [])
    tag_set = set(tags or [])
    selected = []
    for entry in entries:
        if not include_disabled and not entry.enabled:
            continue
        if family_set and entry.family not in family_set:
            continue
        if mode_set and mode_set.isdisjoint(entry.modes):
            continue
        if tag_set and not tag_set.issubset(set(entry.tags)):
            continue
        selected.append(entry)
    return selected


def catalog_entries_to_retrievers_payload(entries: list[RetrieverCatalogEntry]) -> dict[str, Any]:
    return {
        "retrievers": [
            {
                "name": entry.config.name,
                "kind": entry.config.kind,
                "top_k": entry.config.top_k,
                "options": entry.config.options,
                "metadata": {
                    "family": entry.family,
                    "modes": list(entry.modes),
                    "tags": list(entry.tags),
                    "enabled": entry.enabled,
                    "notes": entry.notes,
                },
            }
            for entry in entries
        ]
    }


def load_retriever_specs_file(path: Path) -> list[RetrieverConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("retrievers")
    if not isinstance(payload, list):
        raise ValueError("retrievers file must be a JSON list or a JSON object with a retrievers list")

    retrievers = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"retriever entry must be an object: {item!r}")
        if "name" not in item or "kind" not in item:
            raise ValueError(f"retriever entry must include name and kind: {item!r}")
        retrievers.append(
            RetrieverConfig(
                name=str(item["name"]),
                kind=str(item["kind"]),
                top_k=int(item.get("top_k", 6)),
                options=dict(item.get("options", {})),
            )
        )
    return retrievers


def _catalog_entry(item: dict[str, Any], defaults: dict[str, Any]) -> RetrieverCatalogEntry:
    name = str(item.get("name") or "").strip()
    kind = str(item.get("kind") or "").strip()
    if not name or not kind:
        raise ValueError(f"retriever catalog entry must include name and kind: {item!r}")
    default_options = dict(defaults.get("options", {}))
    family = str(item.get("family") or kind)
    family_options = _family_options(defaults, family)
    options = {**default_options, **family_options, **dict(item.get("options", {}))}
    return RetrieverCatalogEntry(
        config=RetrieverConfig(
            name=name,
            kind=kind,
            top_k=int(item.get("top_k", defaults.get("top_k", 6))),
            options=options,
        ),
        family=family,
        modes=_string_tuple(item.get("modes", item.get("mode", ()))),
        tags=_string_tuple(item.get("tags", ())),
        enabled=bool(item.get("enabled", True)),
        notes=str(item["notes"]) if item.get("notes") is not None else None,
    )


def _family_options(defaults: dict[str, Any], family: str) -> dict[str, Any]:
    family_defaults = defaults.get("family_options", defaults.get("families", {}))
    if not isinstance(family_defaults, dict):
        return {}
    raw = family_defaults.get(family, {})
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("options"), dict):
        return dict(raw["options"])
    return dict(raw)


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
