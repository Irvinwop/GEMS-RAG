from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import load_qa_items
from .types import QAItem


def summarize_qa_items(items: list[QAItem]) -> dict[str, Any]:
    reference_counts = Counter(len(item.references) for item in items)
    content_types: Counter[str] = Counter()
    section_parts: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    for item in items:
        if item.question_type:
            question_types[str(item.question_type)] += 1
        for ref in item.references:
            content_type = str(ref.get("content_type") or "unknown")
            content_types[content_type] += 1
            section_id = str(ref.get("section_id") or "")
            if section_id:
                section_parts[_section_part(section_id)] += 1
    return {
        "total": len(items),
        "expected_refusal": _bool_counts(item.expected_refusal for item in items),
        "has_references": _bool_counts(bool(item.references) for item in items),
        "has_gold_figures": _bool_counts(bool(item.gold_figures) for item in items),
        "reference_count_distribution": {str(key): reference_counts[key] for key in sorted(reference_counts)},
        "reference_content_types": dict(sorted(content_types.items())),
        "reference_parts": dict(sorted(section_parts.items())),
        "question_type_count": len(question_types),
        "top_question_types": [
            {"question_type": question_type, "count": count}
            for question_type, count in question_types.most_common(20)
        ],
        "strata": _strata_counts(items),
    }


def make_qa_split(items: list[QAItem], *, size: int, seed: int = 0, strategy: str = "balanced") -> dict[str, Any]:
    if size <= 0:
        raise ValueError("split size must be positive")
    if not items:
        raise ValueError("cannot split an empty QA set")
    if strategy not in {"balanced", "proportional"}:
        raise ValueError(f"unknown split strategy: {strategy}")
    selected = _balanced_select(items, size=size, seed=seed) if strategy == "balanced" else _proportional_select(items, size=size, seed=seed)
    return {
        "strategy": strategy,
        "seed": seed,
        "requested_size": size,
        "size": len(selected),
        "qa_ids": [item.qa_id for item in selected],
        "summary": summarize_qa_items(selected),
    }


def load_qa_ids_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("qa_ids"), list):
        return [str(item) for item in payload["qa_ids"]]
    raise ValueError(f"QA IDs file must be a JSON list, JSON object with qa_ids, or newline text: {path}")


def summarize_qa_path(path: Path, *, limit: int | None = None, qa_ids: list[str] | None = None) -> dict[str, Any]:
    items = load_qa_items(path, limit=limit, qa_ids=qa_ids)
    return {"qa_path": str(path), **summarize_qa_items(items)}


def write_qa_split(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _balanced_select(items: list[QAItem], *, size: int, seed: int) -> list[QAItem]:
    rng = random.Random(seed)
    groups = _group_by_stratum(items)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[QAItem] = []
    seen: set[str] = set()
    ordered_keys = sorted(groups, key=_stratum_priority)
    while len(selected) < min(size, len(items)):
        progressed = False
        for key in ordered_keys:
            group = groups[key]
            while group and group[0].qa_id in seen:
                group.pop(0)
            if not group:
                continue
            item = group.pop(0)
            selected.append(item)
            seen.add(item.qa_id)
            progressed = True
            if len(selected) >= min(size, len(items)):
                break
        if not progressed:
            break
    return selected


def _proportional_select(items: list[QAItem], *, size: int, seed: int) -> list[QAItem]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[: min(size, len(shuffled))]


def _group_by_stratum(items: list[QAItem]) -> dict[tuple[bool, bool, bool], list[QAItem]]:
    grouped: dict[tuple[bool, bool, bool], list[QAItem]] = defaultdict(list)
    for item in items:
        grouped[_stratum_key(item)].append(item)
    return dict(grouped)


def _strata_counts(items: list[QAItem]) -> list[dict[str, Any]]:
    grouped = _group_by_stratum(items)
    rows = []
    for key, group in sorted(grouped.items()):
        expected_refusal, has_figures, has_references = key
        rows.append(
            {
                "expected_refusal": expected_refusal,
                "has_gold_figures": has_figures,
                "has_references": has_references,
                "count": len(group),
            }
        )
    return rows


def _stratum_key(item: QAItem) -> tuple[bool, bool, bool]:
    return (item.expected_refusal, bool(item.gold_figures), bool(item.references))


def _stratum_priority(key: tuple[bool, bool, bool]) -> tuple[bool, bool, bool, tuple[bool, bool, bool]]:
    expected_refusal, has_figures, has_references = key
    return (not expected_refusal, not has_figures, not has_references, key)


def _bool_counts(values) -> dict[str, int]:
    counts = Counter(bool(value) for value in values)
    return {"true": counts[True], "false": counts[False]}


def _section_part(section_id: str) -> str:
    first = section_id.strip()[:1]
    return f"Part {first}" if first.isdigit() else "unknown"
