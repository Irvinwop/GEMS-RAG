from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .data import load_qa_items


def plan_experiment(config: ExperimentConfig, *, preflight_report: dict[str, Any] | None = None) -> dict[str, Any]:
    items = load_qa_items(config.dataset.qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids)
    retriever_status = _retriever_statuses(preflight_report)
    model_status = _model_statuses(preflight_report)
    grader_status = ((preflight_report or {}).get("sections") or {}).get("grader", {}).get("status")
    conditions = []
    total_rows = 0
    answer_model_calls = 0
    judge_model_calls = 0
    paid_model_calls = 0
    for retriever in config.retrievers:
        for context_mode in config.context_modes:
            for model in config.models:
                rows = len(items)
                answer_calls_per_row = _answer_calls_per_row(context_mode, model.options)
                judge_calls_per_row = 0 if config.grader.provider == "heuristic" else 1
                paid_answer_model_calls = 0 if config.dry_run or model.provider == "dry_run" else rows * answer_calls_per_row
                paid_judge_model_calls = 0 if config.dry_run or config.grader.provider == "heuristic" else rows * judge_calls_per_row
                condition = {
                    "retriever": retriever.name,
                    "retriever_kind": retriever.kind,
                    "retriever_status": retriever_status.get(retriever.name),
                    "context_mode": context_mode,
                    "model_provider": model.provider,
                    "model": model.model,
                    "model_status": model_status.get((model.provider, model.model)),
                    "grader_provider": config.grader.provider,
                    "grader_model": config.grader.model,
                    "grader_status": grader_status,
                    "qa_rows": rows,
                    "answer_model_calls_per_row": answer_calls_per_row,
                    "answer_model_calls": rows * answer_calls_per_row,
                    "judge_model_calls": rows * judge_calls_per_row,
                    "total_model_calls": rows * (answer_calls_per_row + judge_calls_per_row),
                    "paid_model_calls": paid_answer_model_calls + paid_judge_model_calls,
                }
                conditions.append(condition)
                total_rows += condition["qa_rows"]
                answer_model_calls += condition["answer_model_calls"]
                judge_model_calls += condition["judge_model_calls"]
                paid_model_calls += condition["paid_model_calls"]
    return {
        "experiment": config.name,
        "dry_run": config.dry_run,
        "dataset": {
            "qa_path": str(config.dataset.qa_path),
            "mrag_dir": str(config.dataset.mrag_dir),
            "limit": config.dataset.limit,
            "qa_ids": config.dataset.qa_ids,
            "qa_count": len(items),
            "qa_sample": [item.qa_id for item in items[:10]],
        },
        "dimensions": {
            "retrievers": len(config.retrievers),
            "context_modes": len(config.context_modes),
            "models": len(config.models),
            "conditions": len(conditions),
        },
        "estimates": {
            "rows": total_rows,
            "answer_model_calls": answer_model_calls,
            "judge_model_calls": judge_model_calls,
            "total_model_calls": answer_model_calls + judge_model_calls,
            "paid_model_calls": paid_model_calls,
        },
        "preflight": _preflight_summary(preflight_report),
        "conditions": conditions,
    }


def evaluate_plan_budget(
    plan: dict[str, Any],
    *,
    max_rows: int | None = None,
    max_total_model_calls: int | None = None,
    max_paid_model_calls: int | None = None,
) -> dict[str, Any] | None:
    limits = {
        "rows": max_rows,
        "total_model_calls": max_total_model_calls,
        "paid_model_calls": max_paid_model_calls,
    }
    active_limits = {name: limit for name, limit in limits.items() if limit is not None}
    if not active_limits:
        return None

    estimates = plan.get("estimates") or {}
    checks = []
    exceeded = []
    for name, limit in active_limits.items():
        actual = int(estimates.get(name) or 0)
        ok = actual <= int(limit)
        check = {"name": name, "actual": actual, "limit": int(limit), "ok": ok}
        checks.append(check)
        if not ok:
            exceeded.append(check)
    return {
        "ok": not exceeded,
        "checks": checks,
        "exceeded": exceeded,
    }


def _retriever_statuses(preflight_report: dict[str, Any] | None) -> dict[str, str]:
    sections = ((preflight_report or {}).get("sections") or {}).get("retrievers") or []
    return {str(section.get("name")): str(section.get("status")) for section in sections}


def _answer_calls_per_row(context_mode: str, model_options: dict[str, Any] | None = None) -> int:
    if context_mode == "tool_explore":
        return 2
    if context_mode == "tool_search":
        return 3
    if context_mode == "tool_native":
        try:
            max_rounds = int((model_options or {}).get("tool_max_rounds", 4))
        except (TypeError, ValueError):
            max_rounds = 4
        return max(1, min(max_rounds, 20)) + 1
    return 1


def _model_statuses(preflight_report: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    sections = ((preflight_report or {}).get("sections") or {}).get("models") or []
    return {
        (str(section.get("provider")), str(section.get("model"))): str(section.get("status"))
        for section in sections
    }


def _preflight_summary(preflight_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if preflight_report is None:
        return None
    return {
        "ok": preflight_report.get("ok"),
        "status": preflight_report.get("status"),
        "row_estimate": preflight_report.get("row_estimate"),
        "blocking": preflight_report.get("blocking", []),
    }
