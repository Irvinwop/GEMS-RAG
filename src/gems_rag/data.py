from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .types import QAItem


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_qa_items(path: Path, limit: int | None = None, qa_ids: list[str] | None = None) -> list[QAItem]:
    allow = set(qa_ids or [])
    items: list[QAItem] = []
    for row in read_jsonl(path):
        qa_id = str(row.get("qa_id") or row.get("question_id") or f"qa_{len(items) + 1:04d}")
        if allow and qa_id not in allow:
            continue
        items.append(
            QAItem(
                qa_id=qa_id,
                question=row["question"],
                question_type=row.get("question_type"),
                expected_refusal=bool(row.get("expected_refusal", False)),
                gold_answer=row.get("gold_answer", {}),
                references=list(row.get("references", [])),
                gold_figures=list(row.get("gold_figures", [])),
                raw=row,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def load_chunks(mrag_dir: Path) -> list[dict]:
    chunks, _report = canonicalize_chunks(read_jsonl(mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"))
    return chunks


def load_figures(mrag_dir: Path) -> list[dict]:
    return [
        localize_visual_record(mrag_dir, row, kind=str(row.get("kind") or "figure"))
        for row in read_jsonl(mrag_dir / "mmrag_cache_v3" / "figures.jsonl")
    ]


def localize_visual_record(mrag_dir: Path, record: dict, *, kind: str | None = None) -> dict:
    localized = dict(record)
    raw_path = str(localized.get("image_path") or "").strip()
    if not raw_path:
        return localized
    local_path = resolve_visual_path(mrag_dir, raw_path, kind=kind)
    if local_path is not None:
        localized["image_path"] = str(local_path)
    return localized


def resolve_visual_path(mrag_dir: Path, raw_path: str | Path, *, kind: str | None = None) -> Path | None:
    raw = Path(raw_path).expanduser()
    if raw.is_file():
        return raw.resolve()

    filename = raw.name
    normalized_kind = str(kind or "").lower()
    page_first = normalized_kind == "page" or filename.lower().startswith("page_")
    directories = (
        [mrag_dir / "page_images", mrag_dir / "figures"]
        if page_first
        else [mrag_dir / "figures", mrag_dir / "page_images"]
    )
    candidates = [mrag_dir / raw, *(directory / filename for directory in directories)]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def canonicalize_chunks(rows: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """Select one deterministic, information-rich record for each chunk ID."""
    selected: dict[str, dict] = {}
    order: list[str] = []
    counts: dict[str, int] = {}
    raw_rows = 0
    for index, row in enumerate(rows):
        raw_rows += 1
        chunk_id = str(row.get("chunk_id") or f"__missing_chunk_id_{index}")
        counts[chunk_id] = counts.get(chunk_id, 0) + 1
        if chunk_id not in selected:
            selected[chunk_id] = row
            order.append(chunk_id)
            continue
        if _chunk_quality(row) > _chunk_quality(selected[chunk_id]):
            selected[chunk_id] = row
    collision_rows = sum(count - 1 for count in counts.values() if count > 1)
    return [selected[chunk_id] for chunk_id in order], {
        "raw_rows": raw_rows,
        "unique_chunks": len(selected),
        "collision_rows": collision_rows,
        "colliding_ids": sum(count > 1 for count in counts.values()),
    }


def _chunk_quality(row: dict) -> tuple[int, int, int, int]:
    text = str(row.get("text") or "").strip()
    words = re.findall(r"[A-Za-z]{2,}", text)
    references = sum(
        len(row.get(key) or [])
        for key in ["figure_refs", "table_refs", "section_refs", "sign_codes", "modal_verbs"]
    )
    return len(words), sum(character.isalpha() for character in text), len(text), references
