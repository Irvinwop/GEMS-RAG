from __future__ import annotations

import csv
import json
import math
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
    "tool_selected_count",
    "tool_opened_count",
    "tool_call_count",
    "tool_selection_parse_failed",
    "tool_search_query_count",
    "tool_search_result_count",
    "tool_search_error_count",
    "tool_search_parse_failed",
    "retrieval_failed",
    "answer_input_tokens",
    "answer_output_tokens",
    "answer_total_tokens",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_total_tokens",
    "total_tokens",
    "answer_cost_usd",
    "judge_cost_usd",
    "total_cost_usd",
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
    max_total_tokens: int | None = None,
    max_total_cost_usd: float | None = None,
    model_pricing: dict[str, dict[str, float]] | None = None,
    pricing_source: str | None = None,
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
    grader_mismatch_rows = [
        _grader_mismatch_record(row, config)
        for row in rows
        if _row_has_grader_mismatch(row, config)
    ]
    error_counts = {
        "retrieval_errors": sum(1 for row in rows if row.get("retrieval_error")),
        "model_errors": sum(1 for row in rows if row.get("model_error")),
        "judge_errors": sum(1 for row in rows if row.get("judge_error")),
        "incomplete_judge_scores": len(incomplete_judge_score_rows),
        "grader_mismatches": len(grader_mismatch_rows),
        "invalid_json_lines": len(parsed["invalid_json_lines"]),
    }
    structural_ok = not missing_keys and not unexpected_keys and not duplicate_keys and not parsed["invalid_json_lines"]
    error_free = not any(
        error_counts[key]
        for key in ["retrieval_errors", "model_errors", "judge_errors", "incomplete_judge_scores", "grader_mismatches"]
    )
    token_usage = _token_usage_summary(rows)
    cost_summary = _cost_usage_summary(rows, config, model_pricing, sample_size=sample_size)
    budget_checks = _validation_budget_checks(
        token_usage,
        cost_summary,
        max_total_tokens=max_total_tokens,
        max_total_cost_usd=max_total_cost_usd,
    )
    budget_ok = all(check["ok"] for check in budget_checks)
    ok = structural_ok and (allow_errors or error_free) and budget_ok
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
    for check in budget_checks:
        if not check["ok"]:
            if check.get("reason") == "incomplete_cost_coverage":
                problems.append(
                    "cost budget unavailable: "
                    f"missing pricing or usage for {check['missing_cost_calls']} paid model calls"
                )
            else:
                problems.append(f"budget exceeded: {check['name']}={check['actual']} limit={check['limit']}")
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
        "budget_ok": budget_ok,
        "budget_checks": budget_checks,
        "token_usage": token_usage,
        "cost": cost_summary,
        "pricing_models": len(model_pricing or {}),
        "pricing_source": pricing_source,
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
        "grader_mismatches_sample": grader_mismatch_rows[:sample_size],
        "problems": problems,
    }


def summarize_rows(rows: list[dict[str, Any]], *, model_pricing: dict[str, dict[str, float]] | None = None) -> list[dict[str, Any]]:
    rows = _annotate_costs(rows, model_pricing) if model_pricing is not None else rows
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


def leaderboard_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in sorted(groups, key=_leaderboard_sort_key):
        rows.append(
            {
                "rank": len(rows) + 1,
                "retriever": group.get("retriever"),
                "context_mode": group.get("context_mode"),
                "model_provider": group.get("model_provider"),
                "model": group.get("model"),
                "grader": group.get("grader"),
                "rows": group.get("rows"),
                "mean_judge_score": group.get("mean_judge_score"),
                "rubric_score_count": group.get("rubric_score_count"),
                "scored_rows": group.get("scored_rows"),
                "row_errors": group.get("row_errors"),
                "row_error_rate": group.get("row_error_rate"),
                "mean_total_tokens": group.get("mean_total_tokens"),
                "total_cost_usd": group.get("total_cost_usd"),
                "mean_total_cost_usd": group.get("mean_total_cost_usd"),
                "mean_latency_s": group.get("mean_latency_s"),
                "mean_evidence": group.get("mean_evidence"),
                "retrieval_errors": group.get("retrieval_errors"),
                "model_errors": group.get("model_errors"),
                "judge_errors": group.get("judge_errors"),
            }
        )
    return rows


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
    model_pricing: dict[str, dict[str, float]] | None = None,
    pricing_source: str | None = None,
) -> dict[str, Any]:
    if bool(axis) != bool(baseline):
        raise ValueError("axis and baseline must be provided together")
    filters = filters or {}
    output_dir = output_dir or runs_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_run_rows(runs_path)
    filtered_rows = [row for row in rows if _matches(row, filters)]
    if model_pricing is not None:
        filtered_rows = _annotate_costs(filtered_rows, model_pricing)
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
    leaderboard = leaderboard_rows(summary["groups"])
    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard_csv = output_dir / "leaderboard.csv"
    _write_json(leaderboard_json, {"runs": str(runs_path), "rows": leaderboard})
    write_csv(leaderboard_csv, leaderboard)

    report: dict[str, Any] = {
        "status": "complete",
        "runs": str(runs_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "filters": filters,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_csv": str(leaderboard_csv),
        "comparisons": [],
    }
    if model_pricing is not None:
        report["pricing_models"] = len(model_pricing)
        if pricing_source:
            report["pricing_source"] = pricing_source
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


def summarize_rows_by_strata(
    rows: list[dict[str, Any]],
    *,
    qa_lookup: dict[str, Any] | None = None,
    model_pricing: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows = _annotate_costs(rows, model_pricing) if model_pricing is not None else rows
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
    model_pricing: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    rows = _annotate_costs(rows, model_pricing) if model_pricing is not None else rows
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
    context_debug = _context_debug(row)
    if metric == "tool_selected_count":
        return float(len(context_debug.get("selected_ids") or []))
    if metric == "tool_opened_count":
        return float(len(context_debug.get("opened_ids") or []))
    if metric == "tool_call_count":
        tool_calls = (row.get("model_raw") or {}).get("tool_calls")
        return float(len(tool_calls)) if isinstance(tool_calls, list) else 0.0
    if metric == "tool_selection_parse_failed":
        return _bool_metric(context_debug.get("selection_parse_failed"))
    if metric == "tool_search_query_count":
        searches = context_debug.get("search_queries")
        if not isinstance(searches, list):
            searches = context_debug.get("search_results")
        return float(len(searches)) if isinstance(searches, list) else 0.0
    if metric == "tool_search_result_count":
        return float(len(_tool_search_result_ids(context_debug)))
    if metric == "tool_search_error_count":
        return float(len(context_debug.get("search_errors") or []))
    if metric == "tool_search_parse_failed":
        return _bool_metric(context_debug.get("search_parse_failed"))
    if metric == "retrieval_failed":
        return 1.0 if row.get("retrieval_error") else 0.0
    if metric == "answer_input_tokens":
        return _usage_metric(row.get("model_raw"), "input_tokens")
    if metric == "answer_output_tokens":
        return _usage_metric(row.get("model_raw"), "output_tokens")
    if metric == "answer_total_tokens":
        return _usage_metric(row.get("model_raw"), "total_tokens")
    if metric == "judge_input_tokens":
        return _usage_metric(_grader_model_raw(row), "input_tokens")
    if metric == "judge_output_tokens":
        return _usage_metric(_grader_model_raw(row), "output_tokens")
    if metric == "judge_total_tokens":
        return _usage_metric(_grader_model_raw(row), "total_tokens")
    if metric == "total_tokens":
        return _sum_optional(metric_value(row, "answer_total_tokens"), metric_value(row, "judge_total_tokens"))
    cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
    if metric == "answer_cost_usd":
        return _to_float(cost.get("answer_usd"))
    if metric == "judge_cost_usd":
        return _to_float(cost.get("judge_usd"))
    if metric == "total_cost_usd":
        return _to_float(cost.get("total_usd"))
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


def _row_has_grader_mismatch(row: dict[str, Any], config: ExperimentConfig) -> bool:
    return str((row.get("config") or {}).get("grader", "")) != config.grader.model


def _grader_mismatch_record(row: dict[str, Any], config: ExperimentConfig) -> dict[str, Any]:
    return {
        **_key_record(RUN_KEY_FIELDS, _run_key(row)),
        "expected_grader": config.grader.model,
        "actual_grader": str((row.get("config") or {}).get("grader", "")),
    }


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


def _token_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "answer_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "judge_input_tokens",
        "judge_output_tokens",
        "judge_total_tokens",
        "total_tokens",
    ]
    summary: dict[str, Any] = {"rows": len(rows)}
    for metric in metrics:
        values = [metric_value(row, metric) for row in rows]
        numeric = [value for value in values if isinstance(value, int | float)]
        summary[metric] = _clean_number(sum(numeric))
        summary[f"rows_with_{metric}"] = len(numeric)
    summary["rows_with_any_token_usage"] = summary["rows_with_total_tokens"]
    summary["rows_missing_token_usage"] = len(rows) - summary["rows_with_any_token_usage"]
    return summary


def _annotate_costs(rows: list[dict[str, Any]], model_pricing: dict[str, dict[str, float]] | None) -> list[dict[str, Any]]:
    if not model_pricing:
        return rows
    annotated = []
    for row in rows:
        cost = _row_cost(row, model_pricing)
        if cost is None:
            annotated.append(row)
            continue
        updated = dict(row)
        updated["cost"] = cost
        annotated.append(updated)
    return annotated


def _row_cost(row: dict[str, Any], model_pricing: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
    answer_price = _resolve_model_pricing(
        model_pricing,
        provider=str(cfg.get("model_provider") or ""),
        model=str(cfg.get("model") or ""),
    )
    judge_price = _resolve_model_pricing(
        model_pricing,
        provider=str(cfg.get("grader_provider") or ""),
        model=str(cfg.get("grader") or ""),
    )
    answer_raw = row.get("model_raw")
    answer_cost = _usage_cost(answer_raw, answer_price)
    expected_answer_calls = _answer_model_call_count(str(cfg.get("context_mode") or ""), answer_raw)
    if (
        expected_answer_calls > 1
        and not _pricing_is_explicit_zero(answer_price)
        and not _usage_coverage_complete(answer_raw, expected_answer_calls)
    ):
        answer_cost = None
    judge_cost = _usage_cost(_grader_model_raw(row), judge_price)
    total_cost = _sum_optional(answer_cost, judge_cost)
    if total_cost is None:
        return None
    return {
        "currency": "USD",
        "answer_usd": _round_cost(answer_cost) if answer_cost is not None else None,
        "judge_usd": _round_cost(judge_cost) if judge_cost is not None else None,
        "total_usd": _round_cost(total_cost),
        "answer_priced": answer_cost is not None,
        "judge_priced": judge_cost is not None,
    }


def _resolve_model_pricing(
    model_pricing: dict[str, dict[str, float]],
    *,
    provider: str,
    model: str,
) -> dict[str, float] | None:
    if not model:
        return None
    if provider:
        match = model_pricing.get(f"{provider}:{model}")
        if match:
            return match
    return model_pricing.get(model)


def _usage_cost(raw: Any, pricing: dict[str, float] | None) -> float | None:
    if not pricing:
        return None
    input_tokens = _usage_metric(raw, "input_tokens")
    output_tokens = _usage_metric(raw, "output_tokens")
    total_tokens = _usage_metric(raw, "total_tokens")
    input_price = _price_value(pricing, "input_per_1m", "input_usd_per_1m", "prompt_per_1m", "prompt_usd_per_1m")
    output_price = _price_value(pricing, "output_per_1m", "output_usd_per_1m", "completion_per_1m", "completion_usd_per_1m")
    total_price = _price_value(pricing, "total_per_1m", "total_usd_per_1m")
    if input_price == 0 and output_price == 0:
        return 0.0
    if total_price == 0:
        return 0.0
    usage_coverage = raw.get("usage_coverage") if isinstance(raw, dict) else None
    if isinstance(usage_coverage, dict) and usage_coverage.get("complete") is False:
        return None
    if input_price is not None or output_price is not None:
        if input_price is None or output_price is None:
            return None
        if input_price != 0 and input_tokens is None:
            return None
        if output_price != 0 and output_tokens is None:
            return None
        return ((input_tokens or 0) * input_price + (output_tokens or 0) * output_price) / 1_000_000
    if total_tokens is not None and total_price is not None:
        return total_tokens * total_price / 1_000_000
    return None


def _pricing_is_explicit_zero(pricing: dict[str, float] | None) -> bool:
    if not pricing:
        return False
    input_price = _price_value(pricing, "input_per_1m", "input_usd_per_1m", "prompt_per_1m", "prompt_usd_per_1m")
    output_price = _price_value(pricing, "output_per_1m", "output_usd_per_1m", "completion_per_1m", "completion_usd_per_1m")
    total_price = _price_value(pricing, "total_per_1m", "total_usd_per_1m")
    return total_price == 0 or (input_price == 0 and output_price == 0)


def _usage_coverage_complete(raw: Any, expected_calls: int) -> bool:
    if expected_calls <= 1:
        return _usage_metric(raw, "total_tokens") is not None
    coverage = raw.get("usage_coverage") if isinstance(raw, dict) else None
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        return False
    observed_calls = coverage.get("observed_calls")
    reported_expected = coverage.get("expected_calls")
    return (
        isinstance(observed_calls, int)
        and not isinstance(observed_calls, bool)
        and observed_calls >= expected_calls
        and isinstance(reported_expected, int)
        and not isinstance(reported_expected, bool)
        and reported_expected >= expected_calls
    )


def _observed_usage_calls(raw: Any, expected_calls: int) -> int:
    coverage = raw.get("usage_coverage") if isinstance(raw, dict) else None
    if isinstance(coverage, dict):
        value = coverage.get("observed_calls")
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, min(value, expected_calls))
    return 1 if _usage_metric(raw, "total_tokens") is not None else 0


def _answer_model_call_count(context_mode: str, raw: Any = None) -> int:
    if context_mode == "tool_explore":
        return 2
    if context_mode == "tool_search":
        return 3
    if context_mode == "tool_native":
        coverage = raw.get("usage_coverage") if isinstance(raw, dict) else None
        reported = coverage.get("expected_calls") if isinstance(coverage, dict) else None
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            return reported
        model_calls = raw.get("model_calls") if isinstance(raw, dict) else None
        if isinstance(model_calls, list) and model_calls:
            return len(model_calls)
        return 5
    return 1


def _price_value(pricing: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = pricing.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)) and float(value) >= 0:
            return float(value)
    return None


def _cost_usage_summary(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    model_pricing: dict[str, dict[str, float]] | None,
    *,
    sample_size: int,
) -> dict[str, Any]:
    expected_answer_calls = 0
    expected_judge_calls = 0
    priced_answer_calls = 0
    priced_judge_calls = 0
    answer_cost = 0.0
    judge_cost = 0.0
    complete_rows = 0
    missing = []
    missing_cost_calls = 0
    judge_is_paid = not config.dry_run and config.grader.provider != "heuristic"
    judge_price = _resolve_model_pricing(
        model_pricing or {},
        provider=config.grader.provider,
        model=config.grader.model,
    )
    for row in rows:
        cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
        answer_provider = str(cfg.get("model_provider") or "")
        answer_model = str(cfg.get("model") or "")
        answer_is_paid = not config.dry_run and answer_provider != "dry_run"
        answer_raw = row.get("model_raw")
        answer_calls = _answer_model_call_count(str(cfg.get("context_mode") or ""), answer_raw)
        answer_price = _resolve_model_pricing(
            model_pricing or {},
            provider=answer_provider,
            model=answer_model,
        )
        answer_usage_complete = _usage_coverage_complete(answer_raw, answer_calls)
        answer_zero_priced = _pricing_is_explicit_zero(answer_price)
        observed_answer_calls = _observed_usage_calls(answer_raw, answer_calls)
        answer_value = _usage_cost(answer_raw, answer_price)
        if answer_calls > 1 and not answer_zero_priced and not answer_usage_complete:
            answer_value = None
        judge_value = _usage_cost(_grader_model_raw(row), judge_price)
        row_complete = True
        if answer_is_paid:
            expected_answer_calls += answer_calls
            if answer_value is None or (not answer_zero_priced and not answer_usage_complete):
                row_complete = False
                missing_calls = (
                    max(answer_calls - observed_answer_calls, 1)
                    if not answer_usage_complete
                    else answer_calls
                )
                missing_cost_calls += missing_calls
                missing.append(
                    {
                        **_key_record(RUN_KEY_FIELDS, _run_key(row)),
                        "component": "answer",
                        "provider": answer_provider,
                        "model": answer_model,
                        "expected_calls": answer_calls,
                        "observed_usage_calls": observed_answer_calls,
                        "missing_calls": missing_calls,
                    }
                )
            else:
                priced_answer_calls += answer_calls
                answer_cost += answer_value
        if judge_is_paid:
            expected_judge_calls += 1
            if judge_value is None:
                row_complete = False
                missing_cost_calls += 1
                missing.append(
                    {
                        **_key_record(RUN_KEY_FIELDS, _run_key(row)),
                        "component": "judge",
                        "provider": config.grader.provider,
                        "model": config.grader.model,
                        "expected_calls": 1,
                        "observed_usage_calls": 0,
                        "missing_calls": 1,
                    }
                )
            else:
                priced_judge_calls += 1
                judge_cost += judge_value
        if row_complete:
            complete_rows += 1

    known_total = answer_cost + judge_cost
    coverage_ok = not missing
    return {
        "currency": "USD",
        "rows": len(rows),
        "rows_with_complete_cost": complete_rows,
        "rows_missing_cost": len(rows) - complete_rows,
        "expected_answer_calls": expected_answer_calls,
        "expected_judge_calls": expected_judge_calls,
        "expected_paid_calls": expected_answer_calls + expected_judge_calls,
        "priced_answer_calls": priced_answer_calls,
        "priced_judge_calls": priced_judge_calls,
        "priced_paid_calls": priced_answer_calls + priced_judge_calls,
        "missing_cost_components": len(missing),
        "missing_cost_calls": missing_cost_calls,
        "coverage_ok": coverage_ok,
        "known_answer_cost_usd": _round_cost(answer_cost),
        "known_judge_cost_usd": _round_cost(judge_cost),
        "known_total_cost_usd": _round_cost(known_total),
        "total_cost_usd": _round_cost(known_total) if coverage_ok else None,
        "missing_cost_sample": missing[:sample_size],
    }


def _validation_budget_checks(
    token_usage: dict[str, Any],
    cost_summary: dict[str, Any],
    *,
    max_total_tokens: int | None,
    max_total_cost_usd: float | None,
) -> list[dict[str, Any]]:
    checks = []
    if max_total_tokens is not None:
        actual = token_usage.get("total_tokens", 0)
        checks.append(
            {
                "name": "total_tokens",
                "actual": actual,
                "limit": max_total_tokens,
                "ok": actual <= max_total_tokens,
            }
        )
    if max_total_cost_usd is not None:
        coverage_ok = bool(cost_summary.get("coverage_ok"))
        actual = cost_summary.get("total_cost_usd")
        checks.append(
            {
                "name": "total_cost_usd",
                "actual": actual,
                "known_actual": cost_summary.get("known_total_cost_usd"),
                "limit": max_total_cost_usd,
                "cost_coverage_ok": coverage_ok,
                "missing_cost_components": cost_summary.get("missing_cost_components", 0),
                "missing_cost_calls": cost_summary.get("missing_cost_calls", 0),
                "reason": None if coverage_ok else "incomplete_cost_coverage",
                "ok": coverage_ok and isinstance(actual, int | float) and actual <= max_total_cost_usd,
            }
        )
    return checks


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


def _leaderboard_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    score = group.get("mean_judge_score")
    error_rate = group.get("row_error_rate")
    cost = group.get("total_cost_usd")
    tokens = group.get("mean_total_tokens")
    return (
        1 if score is None else 0,
        -(float(score) if isinstance(score, int | float) else -1.0),
        float(error_rate) if isinstance(error_rate, int | float) else 1.0,
        float(cost) if isinstance(cost, int | float) else float("inf"),
        float(tokens) if isinstance(tokens, int | float) else float("inf"),
        str(group.get("retriever") or ""),
        str(group.get("context_mode") or ""),
        str(group.get("model_provider") or ""),
        str(group.get("model") or ""),
        str(group.get("grader") or ""),
    )


def _summarize_group(key: tuple[str, str, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    retriever, context_mode, model_provider, model, grader = key
    row_errors = sum(1 for row in rows if _row_has_error(row))
    rubric_values = [_rubric_score(row, rubric) for row in rows for rubric in RUBRIC_KEYS]
    scored_rows = sum(1 for row in rows if any(_rubric_score(row, rubric) is not None for rubric in RUBRIC_KEYS))
    out: dict[str, Any] = {
        "retriever": retriever,
        "context_mode": context_mode,
        "model_provider": model_provider,
        "model": model,
        "grader": grader,
        "rows": len(rows),
        "row_errors": row_errors,
        "row_error_rate": round(row_errors / len(rows), 4) if rows else None,
        "scored_rows": scored_rows,
        "rubric_score_count": sum(1 for value in rubric_values if isinstance(value, int | float)),
        "mean_judge_score": _mean(rubric_values),
        "retrieval_errors": sum(1 for row in rows if row.get("retrieval_error")),
        "model_errors": sum(1 for row in rows if row.get("model_error")),
        "judge_errors": sum(1 for row in rows if row.get("judge_error")),
        "mean_latency_s": _mean([row.get("latency_s") for row in rows]),
        "mean_evidence": _mean([len(row.get("evidence", [])) for row in rows]),
        "mean_tool_selected": _mean([metric_value(row, "tool_selected_count") for row in rows]),
        "mean_tool_opened": _mean([metric_value(row, "tool_opened_count") for row in rows]),
        "mean_tool_calls": _mean([metric_value(row, "tool_call_count") for row in rows]),
        "tool_selection_parse_failures": int(sum(metric_value(row, "tool_selection_parse_failed") or 0 for row in rows)),
        "mean_tool_search_queries": _mean([metric_value(row, "tool_search_query_count") for row in rows]),
        "mean_tool_search_results": _mean([metric_value(row, "tool_search_result_count") for row in rows]),
        "mean_tool_search_errors": _mean([metric_value(row, "tool_search_error_count") for row in rows]),
        "tool_search_parse_failures": int(sum(metric_value(row, "tool_search_parse_failed") or 0 for row in rows)),
        "mean_answer_input_tokens": _mean([metric_value(row, "answer_input_tokens") for row in rows]),
        "mean_answer_output_tokens": _mean([metric_value(row, "answer_output_tokens") for row in rows]),
        "mean_answer_total_tokens": _mean([metric_value(row, "answer_total_tokens") for row in rows]),
        "mean_judge_input_tokens": _mean([metric_value(row, "judge_input_tokens") for row in rows]),
        "mean_judge_output_tokens": _mean([metric_value(row, "judge_output_tokens") for row in rows]),
        "mean_judge_total_tokens": _mean([metric_value(row, "judge_total_tokens") for row in rows]),
        "mean_total_tokens": _mean([metric_value(row, "total_tokens") for row in rows]),
        "mean_answer_cost_usd": _mean([metric_value(row, "answer_cost_usd") for row in rows], digits=8),
        "mean_judge_cost_usd": _mean([metric_value(row, "judge_cost_usd") for row in rows], digits=8),
        "mean_total_cost_usd": _mean([metric_value(row, "total_cost_usd") for row in rows], digits=8),
        "total_answer_cost_usd": _sum_metric([metric_value(row, "answer_cost_usd") for row in rows], digits=8),
        "total_judge_cost_usd": _sum_metric([metric_value(row, "judge_cost_usd") for row in rows], digits=8),
        "total_cost_usd": _sum_metric([metric_value(row, "total_cost_usd") for row in rows], digits=8),
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


def _row_has_error(row: dict[str, Any]) -> bool:
    return bool(row.get("retrieval_error") or row.get("model_error") or row.get("judge_error"))


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _bool_metric(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _context_debug(row: dict[str, Any]) -> dict[str, Any]:
    debug = ((row.get("retrieval_debug") or {}).get("context_debug") or {})
    return debug if isinstance(debug, dict) else {}


def _tool_search_result_ids(context_debug: dict[str, Any]) -> set[str]:
    result_ids: set[str] = set()
    for result in context_debug.get("search_results") or []:
        if not isinstance(result, dict):
            continue
        for raw_id in result.get("result_ids") or []:
            hit_id = str(raw_id).strip()
            if hit_id:
                result_ids.add(hit_id)
    return result_ids


def _usage_metric(raw: Any, key: str) -> float | None:
    if not isinstance(raw, dict):
        return None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    return _to_float(usage.get(key))


def _grader_model_raw(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("grader_raw")
    if not isinstance(raw, dict):
        return {}
    model_raw = raw.get("model_raw")
    if isinstance(model_raw, dict):
        return model_raw
    return raw


def _sum_optional(*values: float | None) -> float | None:
    nums = [value for value in values if isinstance(value, int | float)]
    return float(sum(nums)) if nums else None


def _clean_number(value: int | float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _round_cost(value: float) -> float:
    return round(value, 8)


def _sum_metric(values: list[Any], *, digits: int = 4) -> float | None:
    nums = [float(value) for value in values if isinstance(value, int | float)]
    return round(sum(nums), digits) if nums else None


def _mean(values: list[Any], *, digits: int = 4) -> float | None:
    nums = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(nums), digits) if nums else None
