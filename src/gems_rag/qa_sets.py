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


def qa_coverage_report(available_items: list[QAItem], selected_items: list[QAItem]) -> dict[str, Any]:
    rows = qa_coverage_rows(available_items, selected_items)
    covered = sum(1 for row in rows if row["selected_count"] > 0)
    selected_total = len(selected_items)
    available_total = len(available_items)
    return {
        "available": summarize_qa_items(available_items),
        "selected": summarize_qa_items(selected_items),
        "coverage": {
            "selected_fraction": round(selected_total / available_total, 4) if available_total else None,
            "strata_total": len(rows),
            "strata_covered": covered,
            "strata_missing": len(rows) - covered,
        },
        "strata": rows,
    }


def qa_coverage_for_selection(
    path: Path,
    *,
    limit: int | None = None,
    qa_ids: list[str] | None = None,
) -> dict[str, Any]:
    available = load_qa_items(path)
    selected = load_qa_items(path, limit=limit, qa_ids=qa_ids)
    return {
        "qa_path": str(path),
        "selection": {
            "limit": limit,
            "qa_ids": qa_ids,
        },
        **qa_coverage_report(available, selected),
    }


def evaluate_qa_coverage(
    report: dict[str, Any],
    *,
    min_selected_per_stratum: int | None = None,
) -> dict[str, Any] | None:
    if min_selected_per_stratum is None:
        return None
    if min_selected_per_stratum < 1:
        raise ValueError("minimum selected QA items per stratum must be positive")

    checks = []
    failed = []
    for row in report.get("strata", []):
        selected_count = int(row.get("selected_count") or 0)
        available_count = int(row.get("available_count") or 0)
        ok = selected_count >= min_selected_per_stratum
        check = {
            "expected_refusal": bool(row.get("expected_refusal")),
            "has_gold_figures": bool(row.get("has_gold_figures")),
            "has_references": bool(row.get("has_references")),
            "available_count": available_count,
            "selected_count": selected_count,
            "minimum_selected": min_selected_per_stratum,
            "shortfall": max(min_selected_per_stratum - selected_count, 0),
            "ok": ok,
        }
        checks.append(check)
        if not ok:
            failed.append(check)
    return {
        "ok": not failed,
        "minimum_selected_per_stratum": min_selected_per_stratum,
        "checks": checks,
        "failed": failed,
    }


def qa_coverage_rows(available_items: list[QAItem], selected_items: list[QAItem]) -> list[dict[str, Any]]:
    available = _group_by_stratum(available_items)
    selected = _group_by_stratum(selected_items)
    available_total = max(len(available_items), 1)
    selected_total = max(len(selected_items), 1)
    rows = []
    for key in sorted(available):
        expected_refusal, has_gold_figures, has_references = key
        available_count = len(available.get(key, []))
        selected_count = len(selected.get(key, []))
        rows.append(
            {
                "expected_refusal": expected_refusal,
                "has_gold_figures": has_gold_figures,
                "has_references": has_references,
                "available_count": available_count,
                "selected_count": selected_count,
                "covered": selected_count > 0,
                "available_share": round(available_count / available_total, 4),
                "selected_share": round(selected_count / selected_total, 4) if selected_items else 0.0,
            }
        )
    return rows


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
