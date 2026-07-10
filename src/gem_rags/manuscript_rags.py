from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retriever_catalog import RetrieverCatalogEntry


ENTRY_KINDS = {"baseline", "rag_system", "survey"}
INTEGRATION_STATUSES = {
    "integrated",
    "acquired_adapter_pending",
    "paper_spec_pending",
}
REQUIRED_MANUSCRIPT_METHOD_IDS = frozenset(
    {
        "bm25",
        "canonical_rag",
        "crag",
        "dense_rag",
        "dpr",
        "gems_rag",
        "gfm_rag",
        "graphrag",
        "hybrid_rag",
        "kg2rag",
        "lpkg",
        "m3kg_rag",
        "megarag",
        "mm_rag",
        "okh_rag",
        "paperqa",
        "sam_rag",
        "self_rag",
        "visrag",
    }
)


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


def validate_manuscript_rag_coverage(
    catalog: ManuscriptRagCatalog,
    retriever_catalog: list[RetrieverCatalogEntry],
) -> dict[str, Any]:
    required_entries = {entry.method_id: entry for entry in catalog.entries if entry.coverage_required}
    required_ids = set(required_entries)
    retriever_names = [entry.config.name for entry in retriever_catalog]
    duplicate_retrievers = sorted(
        {name for name in retriever_names if retriever_names.count(name) > 1}
    )
    retrievers_by_name = {entry.config.name: entry for entry in retriever_catalog}
    referenced_retrievers = {
        retriever
        for entry in required_entries.values()
        for retriever in entry.retrievers
    }
    tagged_manuscript_retrievers = {
        entry.config.name
        for entry in retriever_catalog
        if "manuscript-system" in entry.tags
    }

    missing_required = sorted(REQUIRED_MANUSCRIPT_METHOD_IDS - required_ids)
    unexpected_required = sorted(required_ids - REQUIRED_MANUSCRIPT_METHOD_IDS)
    pending = sorted(
        entry.method_id
        for entry in required_entries.values()
        if entry.integration_status != "integrated"
    )
    without_retrievers = sorted(
        entry.method_id for entry in required_entries.values() if not entry.retrievers
    )
    without_implementation = sorted(
        entry.method_id for entry in required_entries.values() if not entry.implementation.strip()
    )
    missing_retrievers = sorted(referenced_retrievers - set(retrievers_by_name))
    disabled_retrievers = sorted(
        name
        for name in referenced_retrievers
        if name in retrievers_by_name and not retrievers_by_name[name].enabled
    )
    orphan_tagged_retrievers = sorted(tagged_manuscript_retrievers - referenced_retrievers)
    incomplete_upstream = sorted(
        entry.method_id
        for entry in catalog.entries
        if entry.upstream is not None
        and not all(
            [
                entry.upstream.repository.strip(),
                entry.upstream.local_path.strip(),
                entry.upstream.commit.strip(),
                entry.upstream.availability.strip(),
            ]
        )
    )

    problems = []
    for label, values in [
        ("missing required manuscript methods", missing_required),
        ("unexpected coverage-required methods", unexpected_required),
        ("methods not marked integrated", pending),
        ("methods without retrievers", without_retrievers),
        ("methods without implementation descriptions", without_implementation),
        ("retriever names missing from catalog", missing_retrievers),
        ("disabled manuscript retrievers", disabled_retrievers),
        ("unreferenced manuscript-system retrievers", orphan_tagged_retrievers),
        ("entries with incomplete upstream provenance", incomplete_upstream),
        ("duplicate retriever catalog names", duplicate_retrievers),
    ]:
        if values:
            problems.append({"problem": label, "values": values})

    return {
        "ok": not problems,
        "status": "complete" if not problems else "blocked",
        "schema_version": catalog.schema_version,
        "manuscript": catalog.manuscript,
        "required_method_count": len(REQUIRED_MANUSCRIPT_METHOD_IDS),
        "integrated_method_count": sum(
            entry.integration_status == "integrated" for entry in required_entries.values()
        ),
        "referenced_retriever_count": len(referenced_retrievers),
        "required_method_ids": sorted(required_ids),
        "missing_required_method_ids": missing_required,
        "unexpected_required_method_ids": unexpected_required,
        "pending_method_ids": pending,
        "methods_without_retrievers": without_retrievers,
        "methods_without_implementation": without_implementation,
        "missing_retriever_names": missing_retrievers,
        "disabled_retriever_names": disabled_retrievers,
        "orphan_manuscript_retriever_names": orphan_tagged_retrievers,
        "incomplete_upstream_method_ids": incomplete_upstream,
        "duplicate_retriever_names": duplicate_retrievers,
        "problems": problems,
    }


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
