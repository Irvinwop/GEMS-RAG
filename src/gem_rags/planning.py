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
    for retriever in config.retrievers:
        for context_mode in config.context_modes:
            for model in config.models:
                rows = len(items)
                answer_calls_per_row = _answer_calls_per_row(context_mode)
                judge_calls_per_row = 0 if config.grader.provider == "heuristic" else 1
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
                    "answer_model_calls": rows * answer_calls_per_row,
                    "judge_model_calls": rows * judge_calls_per_row,
                    "total_model_calls": rows * (answer_calls_per_row + judge_calls_per_row),
                }
                conditions.append(condition)
                total_rows += condition["qa_rows"]
                answer_model_calls += condition["answer_model_calls"]
                judge_model_calls += condition["judge_model_calls"]
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
            "paid_model_calls": 0 if config.dry_run else answer_model_calls + judge_model_calls,
        },
        "preflight": _preflight_summary(preflight_report),
        "conditions": conditions,
    }


def _retriever_statuses(preflight_report: dict[str, Any] | None) -> dict[str, str]:
    sections = ((preflight_report or {}).get("sections") or {}).get("retrievers") or []
    return {str(section.get("name")): str(section.get("status")) for section in sections}


def _answer_calls_per_row(context_mode: str) -> int:
    if context_mode == "tool_explore":
        return 2
    if context_mode == "tool_search":
        return 3
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
