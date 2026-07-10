from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENTRY_KINDS = {"baseline", "rag_system", "survey"}
INTEGRATION_STATUSES = {
    "integrated",
    "acquired_adapter_pending",
    "paper_spec_pending",
}


@dataclass(frozen=True)
class UpstreamSource:
    repository: str
    local_path: str
    commit: str
    availability: str


@dataclass(frozen=True)
class ManuscriptRagEntry:
    method_id: str
    label: str
    kind: str
    citation_keys: tuple[str, ...]
    manuscript_roles: tuple[str, ...]
    retrievers: tuple[str, ...]
    coverage_required: bool
    integration_status: str
    implementation: str
    upstream: UpstreamSource | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ManuscriptRagCatalog:
    schema_version: int
    manuscript: str
    entries: tuple[ManuscriptRagEntry, ...]


def load_manuscript_rag_catalog(path: Path) -> ManuscriptRagCatalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manuscript RAG catalog must be a JSON object")
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError("manuscript RAG catalog must contain an entries list")

    entries = tuple(_load_entry(item) for item in entries_raw)
    method_ids = [entry.method_id for entry in entries]
    duplicates = sorted({method_id for method_id in method_ids if method_ids.count(method_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate manuscript RAG method_id values: {', '.join(duplicates)}")

    return ManuscriptRagCatalog(
        schema_version=int(raw.get("schema_version", 0)),
        manuscript=str(raw.get("manuscript") or ""),
        entries=entries,
    )


def _load_entry(raw: Any) -> ManuscriptRagEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"manuscript RAG entry must be an object: {raw!r}")
    method_id = str(raw.get("method_id") or "").strip()
    label = str(raw.get("label") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    status = str(raw.get("integration_status") or "").strip()
    if not method_id or not label:
        raise ValueError(f"manuscript RAG entry requires method_id and label: {raw!r}")
    if kind not in ENTRY_KINDS:
        raise ValueError(f"unknown manuscript RAG kind for {method_id}: {kind!r}")
    if status not in INTEGRATION_STATUSES:
        raise ValueError(f"unknown integration_status for {method_id}: {status!r}")

    upstream_raw = raw.get("upstream")
    upstream = None
    if upstream_raw is not None:
        if not isinstance(upstream_raw, dict):
            raise ValueError(f"upstream source for {method_id} must be an object")
        upstream = UpstreamSource(
            repository=str(upstream_raw.get("repository") or ""),
            local_path=str(upstream_raw.get("local_path") or ""),
            commit=str(upstream_raw.get("commit") or ""),
            availability=str(upstream_raw.get("availability") or ""),
        )

    return ManuscriptRagEntry(
        method_id=method_id,
        label=label,
        kind=kind,
        citation_keys=_string_tuple(raw.get("citation_keys")),
        manuscript_roles=_string_tuple(raw.get("manuscript_roles")),
        retrievers=_string_tuple(raw.get("retrievers")),
        coverage_required=bool(raw.get("coverage_required", kind != "survey")),
        integration_status=status,
        implementation=str(raw.get("implementation") or ""),
        upstream=upstream,
        notes=str(raw["notes"]) if raw.get("notes") is not None else None,
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"expected string or list of strings, got {value!r}")
    return tuple(str(item).strip() for item in values if str(item).strip())
