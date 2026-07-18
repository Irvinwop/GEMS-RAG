from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any


def cap_completion_tokens(kwargs: MutableMapping[str, Any], limit: int | None) -> None:
    """Apply a hard ceiling without overriding a caller's smaller token budget."""
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("completion token limit must be positive")

    fields = [
        field
        for field in ("max_tokens", "max_completion_tokens")
        if field in kwargs
    ]
    if not fields:
        fields = ["max_tokens"]
    for field in fields:
        requested = kwargs.get(field)
        kwargs[field] = limit if requested is None else min(int(requested), limit)


async def lightrag_document_status_report(
    rag: Any,
    *,
    doc_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return whether all indexed LightRAG documents reached processed state."""
    storage = getattr(rag, "doc_status", None)
    if storage is None:
        return {
            "complete": False,
            "document_count": 0,
            "status_counts": {"missing_status_storage": 1},
        }

    if doc_ids is None:
        get_counts = getattr(storage, "get_status_counts", None)
        if get_counts is None:
            return {
                "complete": False,
                "document_count": 0,
                "status_counts": {"missing_status_reader": 1},
            }
        raw_counts = await get_counts()
        counts = {
            _status_value(status): int(count)
            for status, count in raw_counts.items()
            if int(count) > 0
        }
        document_count = sum(counts.values())
    else:
        ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))
        get_by_id = getattr(storage, "get_by_id", None)
        if get_by_id is None:
            return {
                "complete": False,
                "document_count": len(ids),
                "status_counts": {"missing_status_reader": len(ids) or 1},
            }
        counts: dict[str, int] = {}
        for doc_id in ids:
            record = await get_by_id(doc_id)
            if isinstance(record, Mapping):
                status = record.get("status")
            else:
                status = getattr(record, "status", None)
            key = _status_value(status) if record is not None else "missing"
            counts[key] = counts.get(key, 0) + 1
        document_count = len(ids)

    return {
        "complete": document_count > 0
        and counts.get("processed", 0) == document_count,
        "document_count": document_count,
        "status_counts": dict(sorted(counts.items())),
    }


def _status_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower() if raw not in {None, ""} else "missing"
