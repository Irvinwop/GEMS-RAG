from __future__ import annotations

import json
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ExperimentConfig
from .data import load_qa_items

RUBRIC_KEYS = [
    "factual_accuracy",
    "category_correctness",
    "citation_validity",
    "verbatim_faithfulness",
    "completeness",
    "refusal_appropriateness",
    "figure_relevance",
    "figure_grounding",
]
DEFAULT_METRICS = [
    *RUBRIC_KEYS,
    "gold_section_recall",
    "gold_reference_recall",
    "evidence_count",
    "retrieval_failed",
    "latency_s",
]
DEFAULT_MATCH_FIELDS = ["qa_id", "retriever", "context_mode", "model_provider", "model", "grader"]
RUN_KEY_FIELDS = ["qa_id", "retriever", "context_mode", "model_provider", "model"]


def load_run_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_run(
    config: ExperimentConfig,
    runs_path: Path | None = None,
    *,
    allow_errors: bool = False,
    sample_size: int = 20,
) -> dict[str, Any]:
    runs_path = runs_path or config.output_dir / config.name / "runs.jsonl"
    parsed = _load_run_rows_lenient(runs_path)
    rows = parsed["rows"]
    expected_keys = _expected_run_keys(config)
    row_keys = [_run_key(row) for row in rows]
    counts = defaultdict(int)
    for key in row_keys:
        counts[key] += 1
    actual_unique_keys = set(counts)
    missing_keys = sorted(expected_keys - actual_unique_keys)
    unexpected_keys = sorted(actual_unique_keys - expected_keys)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    incomplete_judge_score_rows = [
        _incomplete_judge_score_record(row)
        for row in rows
        if _row_has_incomplete_judge_scores(row, config)
    ]
    error_counts = {
        "retrieval_errors": sum(1 for row in rows if row.get("retrieval_error")),
        "model_errors": sum(1 for row in rows if row.get("model_error")),
        "judge_errors": sum(1 for row in rows if row.get("judge_error")),
        "incomplete_judge_scores": len(incomplete_judge_score_rows),
        "invalid_json_lines": len(parsed["invalid_json_lines"]),
    }
    structural_ok = not missing_keys and not unexpected_keys and not duplicate_keys and not parsed["invalid_json_lines"]
    error_free = not any(error_counts[key] for key in ["retrieval_errors", "model_errors", "judge_errors", "incomplete_judge_scores"])
    ok = structural_ok and (allow_errors or error_free)
    problems = []
    if missing_keys:
        problems.append(f"missing expected rows: {len(missing_keys)}")
    if unexpected_keys:
        problems.append(f"unexpected rows: {len(unexpected_keys)}")
    if duplicate_keys:
        problems.append(f"duplicate row keys: {len(duplicate_keys)}")
    if parsed["invalid_json_lines"]:
        problems.append(f"invalid JSON lines: {len(parsed['invalid_json_lines'])}")
    if not allow_errors and not error_free:
        problems.append(
            "run contains errors: "
            + ", ".join(f"{key}={value}" for key, value in error_counts.items() if key != "invalid_json_lines" and value)
        )
    return {
        "ok": ok,
        "status": "ready" if ok else "failed",
        "runs": str(runs_path),
        "experiment": config.name,
        "expected_rows": len(expected_keys),
        "actual_rows": len(rows),
        "actual_unique_rows": len(actual_unique_keys),
        "structural_ok": structural_ok,
        "allow_errors": allow_errors,
        **error_counts,
        "missing_rows": len(missing_keys),
        "unexpected_rows": len(unexpected_keys),
        "duplicate_rows": sum(counts[key] - 1 for key in duplicate_keys),
        "duplicate_keys": len(duplicate_keys),
        "missing_rows_sample": [_key_record(RUN_KEY_FIELDS, key) for key in missing_keys[:sample_size]],
        "unexpected_rows_sample": [_key_record(RUN_KEY_FIELDS, key) for key in unexpected_keys[:sample_size]],
        "duplicate_keys_sample": [_key_record(RUN_KEY_FIELDS, key) for key in duplicate_keys[:sample_size]],
        "invalid_json_lines_sample": parsed["invalid_json_lines"][:sample_size],
        "incomplete_judge_scores_sample": incomplete_judge_score_rows[:sample_size],
        "problems": problems,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cfg = row.get("config", {})
        groups[
            (
                cfg.get("retriever", ""),
                cfg.get("context_mode", ""),
                cfg.get("model_provider", ""),
                cfg.get("model", ""),
                cfg.get("grader", ""),
            )
        ].append(row)
    return [_summarize_group(key, value) for key, value in sorted(groups.items())]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "match_key"}
        flat.update({f"match_{key}": value for key, value in row.get("match_key", {}).items()})
        flattened.append(flat)
    return flattened


def parse_filter(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"filter must be field=value, got {value!r}")
        field, expected = value.split("=", 1)
        filters[field.strip()] = expected.strip()
    return filters


def analyze_run(
    runs_path: Path,
    *,
    output_dir: Path | None = None,
    filters: dict[str, str] | None = None,
    qa_path: Path | None = None,
    axis: str | None = None,
    baseline: str | None = None,
    candidates: list[str] | None = None,
    metrics: list[str] | None = None,
    match_fields: list[str] | None = None,
    write_pairs: bool = True,
) -> dict[str, Any]:
    if bool(axis) != bool(baseline):
        raise ValueError("axis and baseline must be provided together")
    filters = filters or {}
    output_dir = output_dir or runs_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_run_rows(runs_path)
    filtered_rows = [row for row in rows if _matches(row, filters)]
    qa_lookup = _load_qa_lookup(qa_path) if qa_path else {}
    summary = {
        "runs": str(runs_path),
        "rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "filters": filters,
        "groups": summarize_rows(filtered_rows),
    }
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    _write_json(summary_json, summary)
    write_csv(summary_csv, summary["groups"])

    report: dict[str, Any] = {
        "status": "complete",
        "runs": str(runs_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "filters": filters,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "comparisons": [],
    }
    if qa_path:
        report["qa_path"] = str(qa_path)
        report["qa_rows_loaded"] = len(qa_lookup)
        strata_summary = summarize_rows_by_strata(filtered_rows, qa_lookup=qa_lookup)
        strata_summary_csv = output_dir / "strata-summary.csv"
        write_csv(strata_summary_csv, strata_summary)
        report["strata_summary_csv"] = str(strata_summary_csv)
    if axis and baseline:
        candidate_values = candidates or _observed_axis_values(filtered_rows, axis=axis, baseline=baseline)
        report["axis"] = axis
        report["baseline"] = baseline
        report["candidate_values"] = candidate_values
        strata_comparison_rows = []
        for candidate in candidate_values:
            comparison = compare_conditions(
                filtered_rows,
                baseline_filter={axis: baseline},
                candidate_filter={axis: candidate},
                metrics=metrics,
                match_fields=match_fields,
            )
            comparison_without_pairs = {key: value for key, value in comparison.items() if key != "pairs"}
            comparison_without_pairs["axis"] = axis
            comparison_without_pairs["baseline"] = baseline
            comparison_without_pairs["candidate"] = candidate
            comparison_without_pairs["filters"] = filters

            stem = f"compare-{_slug(axis)}-{_slug(baseline)}-vs-{_slug(candidate)}"
            comparison_json = output_dir / f"{stem}.json"
            metrics_csv = output_dir / f"{stem}.csv"
            pairs_csv = output_dir / f"{stem}-pairs.csv"
            _write_json(comparison_json, comparison_without_pairs)
            write_csv(metrics_csv, comparison_without_pairs["metrics"])

            comparison_record = {
                "axis": axis,
                "baseline": baseline,
                "candidate": candidate,
                "matched_pairs": comparison_without_pairs["matched_pairs"],
                "baseline_rows": comparison_without_pairs["baseline_rows"],
                "candidate_rows": comparison_without_pairs["candidate_rows"],
                "comparison_json": str(comparison_json),
                "metrics_csv": str(metrics_csv),
            }
            if write_pairs:
                write_csv(pairs_csv, flatten_pairs(comparison["pairs"]))
                comparison_record["pairs_csv"] = str(pairs_csv)
            report["comparisons"].append(comparison_record)
            if qa_path:
                strata_comparison_rows.extend(
                    compare_conditions_by_strata(
                        filtered_rows,
                        baseline_filter={axis: baseline},
                        candidate_filter={axis: candidate},
                        metrics=metrics,
                        match_fields=match_fields,
                        qa_lookup=qa_lookup,
                        axis=axis,
                        baseline=baseline,
                        candidate=candidate,
                    )
                )
        if qa_path:
            strata_comparisons_csv = output_dir / "strata-comparisons.csv"
            write_csv(strata_comparisons_csv, strata_comparison_rows)
            report["strata_comparisons_csv"] = str(strata_comparisons_csv)

    manifest_json = output_dir / "analysis.json"
    report["manifest_json"] = str(manifest_json)
    _write_json(manifest_json, report)
    return report


def summarize_rows_by_strata(rows: list[dict[str, Any]], *, qa_lookup: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cfg = row.get("config", {})
        for facet, value in _row_strata(row, qa_lookup or {}).items():
            grouped[
                (
                    facet,
                    value,
                    cfg.get("retriever", ""),
                    cfg.get("context_mode", ""),
                    cfg.get("model_provider", ""),
                    cfg.get("model", ""),
                    cfg.get("grader", ""),
                )
            ].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        facet, value, retriever, context_mode, model_provider, model, grader = key
        summary = _summarize_group((retriever, context_mode, model_provider, model, grader), group)
        out.append({"facet": facet, "value": value, **summary})
    return out


def compare_conditions_by_strata(
    rows: list[dict[str, Any]],
    *,
    baseline_filter: dict[str, str],
    candidate_filter: dict[str, str],
    metrics: list[str] | None = None,
    match_fields: list[str] | None = None,
    qa_lookup: dict[str, Any] | None = None,
    axis: str | None = None,
    baseline: str | None = None,
    candidate: str | None = None,
) -> list[dict[str, Any]]:
    strata = sorted({item for row in rows for item in _row_strata(row, qa_lookup or {}).items()})
    out = []
    for facet, value in strata:
        subset = [row for row in rows if _row_strata(row, qa_lookup or {}).get(facet) == value]
        comparison = compare_conditions(
            subset,
            baseline_filter=baseline_filter,
            candidate_filter=candidate_filter,
            metrics=metrics,
            match_fields=match_fields,
        )
        for metric in comparison["metrics"]:
            out.append(
                {
                    "facet": facet,
                    "value": value,
                    "axis": axis,
                    "baseline": baseline,
                    "candidate": candidate,
                    "baseline_rows": comparison["baseline_rows"],
                    "candidate_rows": comparison["candidate_rows"],
                    "matched_pairs": comparison["matched_pairs"],
                    **metric,
                }
            )
    return out


def compare_conditions(
    rows: list[dict[str, Any]],
    *,
    baseline_filter: dict[str, str],
    candidate_filter: dict[str, str],
    metrics: list[str] | None = None,
    match_fields: list[str] | None = None,
) -> dict[str, Any]:
    metrics = metrics or DEFAULT_METRICS
    changed_fields = set(baseline_filter) | set(candidate_filter)
    if match_fields is None:
        match_fields = [field for field in DEFAULT_MATCH_FIELDS if field not in changed_fields]
    baseline_rows = [_row for _row in rows if _matches(_row, baseline_filter)]
    candidate_rows = [_row for _row in rows if _matches(_row, candidate_filter)]
    baseline_index = _index_rows(baseline_rows, match_fields)
    candidate_index = _index_rows(candidate_rows, match_fields)
    matched_keys = sorted(set(baseline_index) & set(candidate_index))
    pair_rows = []
    for key in matched_keys:
        base = baseline_index[key][0]
        cand = candidate_index[key][0]
        pair = {"match_key": _key_record(match_fields, key)}
        for metric in metrics:
            base_value = metric_value(base, metric)
            cand_value = metric_value(cand, metric)
            pair[f"baseline_{metric}"] = base_value
            pair[f"candidate_{metric}"] = cand_value
            pair[f"delta_{metric}"] = None if base_value is None or cand_value is None else round(cand_value - base_value, 4)
        pair_rows.append(pair)
    return {
        "rows": len(rows),
        "baseline_filter": baseline_filter,
        "candidate_filter": candidate_filter,
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "match_fields": match_fields,
        "matched_pairs": len(pair_rows),
        "duplicate_baseline_keys": _duplicate_count(baseline_index),
        "duplicate_candidate_keys": _duplicate_count(candidate_index),
        "metrics": [_metric_summary(metric, pair_rows) for metric in metrics],
        "pairs": pair_rows,
    }


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric in RUBRIC_KEYS:
        value = (row.get("judge_scores") or {}).get(metric)
        if isinstance(value, dict):
            value = value.get("score")
        return _to_float(value)
    diagnostics = ((row.get("grader_raw") or {}).get("diagnostics") or {})
    if metric == "gold_section_recall":
        return _to_float(diagnostics.get("gold_section_recall"))
    if metric == "gold_reference_recall":
        return _to_float(diagnostics.get("gold_reference_recall"))
    if metric == "gold_category_recall":
        return _to_float(diagnostics.get("gold_category_recall"))
    if metric == "evidence_count":
        return float(len(row.get("evidence", [])))
    if metric == "retrieval_failed":
        return 1.0 if row.get("retrieval_error") else 0.0
    if metric == "latency_s":
        return _to_float(row.get("latency_s"))
    raise ValueError(f"unknown metric: {metric}")


def _expected_run_keys(config: ExperimentConfig) -> set[tuple[str, str, str, str, str]]:
    items = load_qa_items(config.dataset.qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids)
    return {
        (
            item.qa_id,
            retriever.name,
            context_mode,
            model.provider,
            model.model,
        )
        for item in items
        for retriever in config.retrievers
        for context_mode in config.context_modes
        for model in config.models
    }


def _run_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    cfg = row.get("config") or {}
    return (
        str(row.get("qa_id", "")),
        str(cfg.get("retriever", "")),
        str(cfg.get("context_mode", "")),
        str(cfg.get("model_provider", "")),
        str(cfg.get("model", "")),
    )


def _row_has_incomplete_judge_scores(row: dict[str, Any], config: ExperimentConfig) -> bool:
    if row.get("judge_error"):
        return False
    if config.dry_run and config.grader.provider != "heuristic":
        return False
    return bool(_missing_judge_score_keys(row))


def _incomplete_judge_score_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **_key_record(RUN_KEY_FIELDS, _run_key(row)),
        "missing_score_keys": _missing_judge_score_keys(row),
    }


def _missing_judge_score_keys(row: dict[str, Any]) -> list[str]:
    scores = row.get("judge_scores")
    if not isinstance(scores, dict):
        return list(RUBRIC_KEYS)
    return [key for key in RUBRIC_KEYS if key not in scores]


def _load_run_rows_lenient(path: Path) -> dict[str, Any]:
    rows = []
    invalid = []
    if not path.exists():
        return {"rows": rows, "invalid_json_lines": [{"line_number": None, "error": f"missing run file: {path}"}]}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                invalid.append({"line_number": line_number, "error": str(exc), "text": line[:500]})
    return {"rows": rows, "invalid_json_lines": invalid}


def _matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
    return all(str(_field_value(row, field)) == expected for field, expected in filters.items())


def _field_value(row: dict[str, Any], field: str) -> Any:
    if field in {"retriever", "context_mode", "model_provider", "model", "grader", "experiment"}:
        return (row.get("config") or {}).get(field)
    if "." in field:
        current: Any = row
        for part in field.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
    return row.get(field)


def _load_qa_lookup(qa_path: Path) -> dict[str, Any]:
    return {item.qa_id: item for item in load_qa_items(qa_path)}


def _row_strata(row: dict[str, Any], qa_lookup: dict[str, Any]) -> dict[str, str]:
    qa_id = str(row.get("qa_id", ""))
    item = qa_lookup.get(qa_id)
    expected_refusal = item.expected_refusal if item else bool(row.get("expected_refusal", False))
    question_type = (item.question_type if item else row.get("question_type")) or "unknown"
    references = item.references if item else []
    gold_figures = item.gold_figures if item else []
    strata = {
        "expected_refusal": _bool_label(expected_refusal),
        "question_type": str(question_type),
    }
    if item is not None:
        strata["has_gold_figures"] = _bool_label(bool(gold_figures))
        strata["has_references"] = _bool_label(bool(references))
        strata["reference_count"] = str(len(references))
        content_types = sorted({str(ref.get("content_type") or "unknown") for ref in references})
        strata["reference_content_types"] = "+".join(content_types) if content_types else "none"
    return strata


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _observed_axis_values(rows: list[dict[str, Any]], *, axis: str, baseline: str) -> list[str]:
    values = set()
    for row in rows:
        value = _field_value(row, axis)
        if value is None:
            continue
        text = str(value)
        if text and text != baseline:
            values.add(text)
    return sorted(values)


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-").lower()
    text = text[:80].strip("-")
    return text or "blank"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _index_rows(rows: list[dict[str, Any]], match_fields: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        index[tuple(_field_value(row, field) for field in match_fields)].append(row)
    return dict(index)


def _duplicate_count(index: dict[tuple[Any, ...], list[dict[str, Any]]]) -> int:
    return sum(len(rows) - 1 for rows in index.values() if len(rows) > 1)


def _key_record(match_fields: list[str], key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(match_fields, key, strict=False))


def _metric_summary(metric: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [pair[f"delta_{metric}"] for pair in pairs if isinstance(pair.get(f"delta_{metric}"), int | float)]
    baseline_values = [pair[f"baseline_{metric}"] for pair in pairs if isinstance(pair.get(f"baseline_{metric}"), int | float)]
    candidate_values = [pair[f"candidate_{metric}"] for pair in pairs if isinstance(pair.get(f"candidate_{metric}"), int | float)]
    wins = sum(1 for delta in deltas if delta > 0)
    losses = sum(1 for delta in deltas if delta < 0)
    ties = sum(1 for delta in deltas if delta == 0)
    return {
        "metric": metric,
        "pairs": len(deltas),
        "baseline_mean": _mean(baseline_values),
        "candidate_mean": _mean(candidate_values),
        "mean_delta": _mean(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / len(deltas), 4) if deltas else None,
    }


def _summarize_group(key: tuple[str, str, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    retriever, context_mode, model_provider, model, grader = key
    out: dict[str, Any] = {
        "retriever": retriever,
        "context_mode": context_mode,
        "model_provider": model_provider,
        "model": model,
        "grader": grader,
        "rows": len(rows),
        "retrieval_errors": sum(1 for row in rows if row.get("retrieval_error")),
        "model_errors": sum(1 for row in rows if row.get("model_error")),
        "judge_errors": sum(1 for row in rows if row.get("judge_error")),
        "mean_latency_s": _mean([row.get("latency_s") for row in rows]),
        "mean_evidence": _mean([len(row.get("evidence", [])) for row in rows]),
    }
    for rubric in RUBRIC_KEYS:
        out[f"mean_{rubric}"] = _mean([_rubric_score(row, rubric) for row in rows])
    out["mean_gold_section_recall"] = _mean(
        [((row.get("grader_raw") or {}).get("diagnostics") or {}).get("gold_section_recall") for row in rows]
    )
    out["mean_gold_reference_recall"] = _mean(
        [((row.get("grader_raw") or {}).get("diagnostics") or {}).get("gold_reference_recall") for row in rows]
    )
    return out


def _rubric_score(row: dict[str, Any], rubric: str) -> float | None:
    value = (row.get("judge_scores") or {}).get(rubric)
    if isinstance(value, dict):
        value = value.get("score")
    return _to_float(value)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(nums), 4) if nums else None
