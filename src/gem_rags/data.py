from __future__ import annotations

import json
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
        qa_id = row.get("qa_id") or f"qa_{len(items) + 1:04d}"
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
    return list(read_jsonl(mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"))


def load_figures(mrag_dir: Path) -> list[dict]:
    return list(read_jsonl(mrag_dir / "mmrag_cache_v3" / "figures.jsonl"))
