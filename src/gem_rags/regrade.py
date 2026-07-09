from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, GraderConfig, ModelConfig
from .data import load_qa_items
from .grading import RUBRIC_KEYS, grade_answer
from .models import LLM_MODEL_PROVIDERS, ModelClient, build_model
from .types import Evidence, ModelResult, QAItem, RetrievalResult


def regrade_run(
    config: ExperimentConfig,
    *,
    runs_path: Path | None = None,
    output_path: Path | None = None,
    grader: GraderConfig | None = None,
    only_missing: bool = False,
) -> dict[str, Any]:
    runs_path = runs_path or config.output_dir / config.name / "runs.jsonl"
    output_path = output_path or config.output_dir / config.name / "regraded-runs.jsonl"
    if runs_path.resolve() == output_path.resolve():
        raise ValueError("regrade output path must differ from input runs path")
    grader = grader or config.grader
    grader_client, grader_build_error = _safe_build_grader(grader)
    qa_by_id = {item.qa_id: item for item in load_qa_items(config.dataset.qa_path)}
    stats = {
        "input": str(runs_path),
        "output": str(output_path),
        "grader_provider": grader.provider,
        "grader_model": grader.model,
        "rows": 0,
        "rows_regraded": 0,
        "rows_copied": 0,
        "missing_qa": 0,
        "judge_errors": 0,
        "only_missing": only_missing,
        "grader_build_error": grader_build_error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regraded_at = datetime.now(UTC).isoformat()
    with runs_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            if only_missing and _has_complete_judge_scores(row) and not row.get("judge_error"):
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows_copied"] += 1
                continue
            item = qa_by_id.get(str(row.get("qa_id", "")))
            if item is None:
                row = deepcopy(row)
                row["judge_error"] = f"missing QA item for regrade: {row.get('qa_id')}"
                row.setdefault("regrade_debug", {})
                row["regrade_debug"].update({"regraded_at": regraded_at, "error": row["judge_error"]})
                stats["missing_qa"] += 1
                stats["judge_errors"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if grader_build_error:
                updated = _regrade_grader_build_error_row(row, grader, regraded_at, grader_build_error)
            else:
                try:
                    updated = regrade_row(row, item, grader, regraded_at=regraded_at, model_client=grader_client)
                except Exception as exc:
                    updated = _regrade_error_row(row, grader, regraded_at, exc)
            if updated.get("judge_error"):
                stats["judge_errors"] += 1
            stats["rows_regraded"] += 1
            dst.write(json.dumps(updated, ensure_ascii=False) + "\n")
    stats["ok"] = stats["missing_qa"] == 0 and stats["judge_errors"] == 0
    return stats


def regrade_row(
    row: dict[str, Any],
    item: QAItem,
    grader: GraderConfig,
    *,
    regraded_at: str | None = None,
    model_client: ModelClient | None = None,
) -> dict[str, Any]:
    row = deepcopy(row)
    retrieval = _retrieval_from_row(row)
    model_result = _model_result_from_row(row)
    grade = grade_answer(grader, item, model_result, retrieval, model_client=model_client)
    row.setdefault("config", {})
    row["config"]["grader_provider"] = grader.provider
    row["config"]["grader"] = grade.grader
    row["judge_scores"] = grade.scores
    row["judge_confidence"] = grade.confidence
    row["judge_explanation"] = grade.explanation
    row["figure_metrics"] = grade.figure_metrics
    row["system_confidence_breakdown"] = grade.system_confidence_breakdown
    row["grader_raw"] = grade.raw
    row["judge_error"] = grade.error
    row["regrade_debug"] = {
        "regraded_at": regraded_at or datetime.now(UTC).isoformat(),
        "grader_provider": grader.provider,
        "grader_model": grader.model,
        "evidence_count": len(retrieval.evidence),
    }
    return row


def _safe_build_grader(grader: GraderConfig) -> tuple[ModelClient | None, str | None]:
    if grader.provider == "heuristic" or grader.provider not in LLM_MODEL_PROVIDERS:
        return None, None
    try:
        return build_model(ModelConfig(provider=grader.provider, model=grader.model, options=grader.options)), None
    except Exception as exc:
        return None, f"grader_build_failed: {type(exc).__name__}: {exc}"


def _has_complete_judge_scores(row: dict[str, Any]) -> bool:
    scores = row.get("judge_scores")
    return isinstance(scores, dict) and all(key in scores for key in RUBRIC_KEYS)


def _regrade_grader_build_error_row(row: dict[str, Any], grader: GraderConfig, regraded_at: str, error: str) -> dict[str, Any]:
    row = deepcopy(row)
    row["judge_error"] = error
    row.setdefault("regrade_debug", {})
    row["regrade_debug"].update(
        {
            "regraded_at": regraded_at,
            "grader_provider": grader.provider,
            "grader_model": grader.model,
            "error": error,
        }
    )
    return row


def _regrade_error_row(row: dict[str, Any], grader: GraderConfig, regraded_at: str, exc: Exception) -> dict[str, Any]:
    row = deepcopy(row)
    row["judge_error"] = f"regrade_failed: {type(exc).__name__}: {exc}"
    row.setdefault("regrade_debug", {})
    row["regrade_debug"].update(
        {
            "regraded_at": regraded_at,
            "grader_provider": grader.provider,
            "grader_model": grader.model,
            "error": row["judge_error"],
        }
    )
    return row


def _retrieval_from_row(row: dict[str, Any]) -> RetrievalResult:
    cfg = row.get("config") or {}
    evidence = []
    for record in row.get("evidence") or []:
        evidence.append(
            Evidence(
                evidence_id=str(record.get("evidence_id") or ""),
                kind=record.get("kind") or "tool_trace",
                text=str(record.get("text") or ""),
                metadata=dict(record.get("metadata") or {}),
                score=float(record.get("score") or 0.0),
            )
        )
    return RetrievalResult(
        adapter=str(cfg.get("retriever") or ""),
        query=str(row.get("question") or ""),
        evidence=evidence,
        debug=dict(row.get("retrieval_debug") or {}),
        error=row.get("retrieval_error"),
    )


def _model_result_from_row(row: dict[str, Any]) -> ModelResult:
    cfg = row.get("config") or {}
    raw = row.get("model_raw") if isinstance(row.get("model_raw"), dict) else {}
    return ModelResult(
        provider=str(cfg.get("model_provider") or ""),
        model=str(cfg.get("model") or ""),
        output=str(row.get("answer") or ""),
        raw={**raw, "regraded_from_row": True},
        error=row.get("model_error"),
    )
