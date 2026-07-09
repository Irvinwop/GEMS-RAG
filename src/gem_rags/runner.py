from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import load_qa_items
from .grading import grade_answer
from .models import build_model
from .prompts import (
    build_injected_prompt,
    build_tool_answer_prompt,
    build_tool_search_answer_prompt,
    build_tool_search_query_prompt,
    build_tool_search_selection_prompt,
    build_tool_selection_prompt,
    parse_open_hit_ids,
    parse_search_queries,
)
from .retrieval import build_retriever
from .types import ContextMode, Evidence, GradingResult, ModelResult, QAItem, RetrievalResult


def run_experiment(config: ExperimentConfig, *, overwrite: bool = False, resume: bool = False, retry_errors: bool = False) -> Path:
    if sum(bool(value) for value in [overwrite, resume, retry_errors]) > 1:
        raise ValueError("--overwrite, --resume, and --retry-errors are mutually exclusive")

    output_dir = config.output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs.jsonl"
    manifest_path = output_dir / "manifest.json"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    items = load_qa_items(config.dataset.qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids)
    retrievers = [(ret, *_safe_build_retriever(ret, config.dataset.mrag_dir)) for ret in config.retrievers]
    models = [(model, _safe_build_model(model)) for model in config.models]
    if overwrite and output_path.exists():
        output_path.unlink()
    retry_stats = {}
    if retry_errors:
        completed, retry_stats = _prepare_retry_errors(output_path)
    else:
        completed = _load_completed_keys(output_path) if resume else set()

    summary = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "overwrite" if overwrite else "retry_errors" if retry_errors else "resume" if resume else "append",
        "rows_written": 0,
        "rows_skipped": 0,
        "items": len(items),
        "retrievers": len(retrievers),
        "context_modes": len(config.context_modes),
        "models": len(models),
        "retriever_build_errors": sum(1 for _, _, error in retrievers if error),
        "model_build_errors": sum(1 for _, client in models if getattr(client, "build_error", None)),
        **retry_stats,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        for item in items:
            for retriever_config, retriever, retriever_build_error in retrievers:
                retrieval_cache: RetrievalResult | None = None
                for context_mode in config.context_modes:
                    if context_mode == "tool_search":
                        retrieval = _deferred_retrieval(retriever_config, item, retriever_build_error)
                    else:
                        if retrieval_cache is None:
                            retrieval_cache = _safe_retrieve(retriever_config, retriever, item, retriever_build_error)
                        retrieval = retrieval_cache
                    for model_config, model_client in models:
                        key = _completed_key(item.qa_id, retriever_config.name, context_mode, model_config.provider, model_config.model)
                        if key in completed:
                            summary["rows_skipped"] += 1
                            continue
                        started = time.time()
                        model_result, context_retrieval, context_debug = _generate_for_context(
                            context_mode,
                            model_client,
                            item,
                            retrieval,
                            config.max_evidence_chars,
                            retriever=retriever,
                            retriever_build_error=retriever_build_error,
                        )
                        latency_s = time.time() - started
                        grade = _safe_grade(config.grader, item, model_result, context_retrieval)
                        row = {
                            "qa_id": item.qa_id,
                            "question": item.question,
                            "question_type": item.question_type,
                            "expected_refusal": item.expected_refusal,
                            "config": {
                                "experiment": config.name,
                                "retriever": retrieval.adapter,
                                "context_mode": context_mode,
                                "model_provider": model_result.provider,
                                "model": model_result.model,
                                "grader": grade.grader,
                            },
                            "run": {
                                "run_id": run_id,
                                "started_at": summary["started_at"],
                            },
                            "answer": model_result.output,
                            "retrieval_error": context_retrieval.error or retrieval.error,
                            "model_error": model_result.error,
                            "latency_s": round(latency_s, 3),
                            "evidence": [_evidence_record(ev) for ev in context_retrieval.evidence],
                            "retrieval_debug": {
                                **retrieval.debug,
                                "retrieved_evidence_count": len(retrieval.evidence),
                                "provided_evidence_count": len(context_retrieval.evidence),
                                "context_debug": context_debug,
                            },
                            "judge_scores": grade.scores,
                            "judge_confidence": grade.confidence,
                            "judge_explanation": grade.explanation,
                            "figure_metrics": grade.figure_metrics,
                            "system_confidence_breakdown": grade.system_confidence_breakdown,
                            "grader_raw": grade.raw,
                            "judge_error": grade.error,
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed.add(key)
                        summary["rows_written"] += 1
    summary["finished_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps({"config": _json_safe(asdict(config)), "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _safe_build_retriever(retriever_config, mrag_dir: Path):
    try:
        return build_retriever(retriever_config, mrag_dir), None
    except Exception as exc:
        return None, _error("retriever_build_failed", exc)


def _safe_build_model(model_config):
    try:
        return build_model(model_config)
    except Exception as exc:
        return _UnavailableModelClient(model_config, _error("model_build_failed", exc))


def _safe_retrieve(retriever_config, retriever, item: QAItem, build_error: str | None) -> RetrievalResult:
    if build_error:
        return RetrievalResult(
            adapter=retriever_config.name,
            query=item.question,
            evidence=[],
            debug={"retriever_build_error": build_error},
            error=build_error,
        )
    try:
        return retriever.retrieve(item)
    except Exception as exc:
        error = _error("retriever_failed", exc)
        return RetrievalResult(
            adapter=retriever_config.name,
            query=item.question,
            evidence=[],
            debug={"retriever_error": error},
            error=error,
        )


def _deferred_retrieval(retriever_config, item: QAItem, build_error: str | None) -> RetrievalResult:
    return RetrievalResult(
        adapter=retriever_config.name,
        query=item.question,
        evidence=[],
        debug={"deferred_retrieval": True, "retriever_build_error": build_error},
        error=build_error,
    )


def _safe_grade(config, item: QAItem, model_result: ModelResult, retrieval: RetrievalResult) -> GradingResult:
    try:
        return grade_answer(config, item, model_result, retrieval)
    except Exception as exc:
        error = _error("grade_failed", exc)
        return GradingResult(
            grader=config.model,
            scores={},
            raw={"error": error},
            error=error,
        )


def _build_prompt(context_mode: ContextMode, item, evidence: list[Evidence], max_evidence_chars: int) -> str:
    if context_mode == "injected":
        return build_injected_prompt(item, evidence, max_evidence_chars)
    if context_mode == "tool_explore":
        return build_tool_answer_prompt(item, evidence, max_evidence_chars)
    if context_mode == "tool_search":
        return build_tool_search_answer_prompt(item, evidence, max_evidence_chars)
    raise ValueError(f"unknown context mode: {context_mode}")


def _generate_for_context(
    context_mode: ContextMode,
    model_client,
    item: QAItem,
    retrieval: RetrievalResult,
    max_evidence_chars: int,
    *,
    retriever=None,
    retriever_build_error: str | None = None,
) -> tuple[ModelResult, RetrievalResult, dict[str, Any]]:
    try:
        if context_mode == "injected":
            prompt = build_injected_prompt(item, retrieval.evidence, max_evidence_chars)
            result = model_client.generate(prompt)
            return result, retrieval, {"mode": "injected", "prompt_chars": len(prompt)}
        if context_mode == "tool_explore":
            return _generate_tool_explore(model_client, item, retrieval, max_evidence_chars)
        if context_mode == "tool_search":
            return _generate_tool_search(model_client, item, retriever, retrieval, max_evidence_chars, retriever_build_error)
        raise ValueError(f"unknown context mode: {context_mode}")
    except Exception as exc:
        error = _error("model_generate_failed", exc)
        model_config = getattr(model_client, "config", None)
        return (
            ModelResult(
                provider=str(getattr(model_config, "provider", "unknown")),
                model=str(getattr(model_config, "model", "unknown")),
                output="",
                raw={"error": error},
                error=error,
            ),
            retrieval,
            {"mode": context_mode, "error": error},
        )


def _generate_tool_explore(
    model_client,
    item: QAItem,
    retrieval: RetrievalResult,
    max_evidence_chars: int,
    max_open: int = 5,
) -> tuple[ModelResult, RetrievalResult, dict[str, Any]]:
    selection_prompt = build_tool_selection_prompt(item, retrieval.evidence, max_evidence_chars)
    selection = model_client.generate(selection_prompt)
    selected_ids = parse_open_hit_ids(selection.output)
    selection_parse_failed = not selected_ids
    if selection_parse_failed and selection.raw.get("dry_run"):
        selected_ids = [ev.evidence_id for ev in retrieval.evidence[:max_open]]
    allowed = {ev.evidence_id: ev for ev in retrieval.evidence}
    opened: list[Evidence] = []
    opened_ids: set[str] = set()
    for hit_id in selected_ids:
        ev = allowed.get(hit_id)
        if ev and ev.evidence_id not in opened_ids:
            opened.append(ev)
            opened_ids.add(ev.evidence_id)
        if len(opened) >= max_open:
            break
    answer_prompt = build_tool_answer_prompt(item, opened, max_evidence_chars)
    answer = model_client.generate(answer_prompt)
    raw = {
        **answer.raw,
        "tool_explore": {
            "selection_prompt_chars": len(selection_prompt),
            "answer_prompt_chars": len(answer_prompt),
            "selection_output": selection.output,
            "selection_error": selection.error,
            "selected_ids": selected_ids,
            "opened_ids": [ev.evidence_id for ev in opened],
            "selection_parse_failed": selection_parse_failed,
        },
    }
    result = ModelResult(
        provider=answer.provider,
        model=answer.model,
        output=answer.output,
        raw=raw,
        error=answer.error or selection.error,
    )
    context_retrieval = RetrievalResult(
        adapter=retrieval.adapter,
        query=retrieval.query,
        evidence=opened,
        debug=retrieval.debug,
        error=retrieval.error,
    )
    return result, context_retrieval, raw["tool_explore"]


def _generate_tool_search(
    model_client,
    item: QAItem,
    retriever,
    initial_retrieval: RetrievalResult,
    max_evidence_chars: int,
    retriever_build_error: str | None = None,
    max_searches: int = 2,
    max_open: int = 5,
) -> tuple[ModelResult, RetrievalResult, dict[str, Any]]:
    if retriever_build_error or retriever is None:
        error = retriever_build_error or "retriever is unavailable for tool_search"
        model_config = getattr(model_client, "config", None)
        return (
            ModelResult(
                provider=str(getattr(model_config, "provider", "unknown")),
                model=str(getattr(model_config, "model", "unknown")),
                output="",
                raw={"tool_search": {"error": error}},
                error=error,
            ),
            RetrievalResult(
                adapter=initial_retrieval.adapter,
                query=item.question,
                evidence=[],
                debug={**initial_retrieval.debug, "tool_search": True, "error": error},
                error=error,
            ),
            {"mode": "tool_search", "error": error},
        )

    search_prompt = build_tool_search_query_prompt(item, max_searches=max_searches)
    search_plan = model_client.generate(search_prompt)
    search_queries = parse_search_queries(search_plan.output, max_queries=max_searches)
    search_parse_failed = not search_queries
    if search_parse_failed and search_plan.raw.get("dry_run"):
        search_queries = [{"query": item.question, "top_k": max_open}]

    catalog: list[Evidence] = []
    search_debug: list[dict[str, Any]] = []
    search_errors: list[str] = []
    seen: set[str] = set()
    for query_spec in search_queries:
        query = str(query_spec["query"])
        top_k = int(query_spec["top_k"])
        search_item = QAItem(
            qa_id=item.qa_id,
            question=query,
            question_type=item.question_type,
            expected_refusal=item.expected_refusal,
            gold_answer=item.gold_answer,
            references=item.references,
            gold_figures=item.gold_figures,
            raw=item.raw,
        )
        result = _safe_retrieve_for_tool_search(retriever, initial_retrieval.adapter, search_item)
        if result.error:
            search_errors.append(result.error)
        result_ids = []
        for rank, ev in enumerate(result.evidence[:top_k], 1):
            result_ids.append(ev.evidence_id)
            if ev.evidence_id in seen:
                continue
            catalog.append(
                Evidence(
                    evidence_id=ev.evidence_id,
                    kind=ev.kind,
                    text=ev.text,
                    metadata={**ev.metadata, "tool_search_query": query, "tool_search_rank": rank},
                    score=ev.score,
                )
            )
            seen.add(ev.evidence_id)
        search_debug.append(
            {
                "query": query,
                "top_k": top_k,
                "adapter": result.adapter,
                "result_ids": result_ids,
                "error": result.error,
            }
        )

    selection_prompt = build_tool_search_selection_prompt(item, catalog, max_evidence_chars)
    selection = model_client.generate(selection_prompt)
    selected_ids = parse_open_hit_ids(selection.output)
    selection_parse_failed = not selected_ids
    if selection_parse_failed and selection.raw.get("dry_run"):
        selected_ids = [ev.evidence_id for ev in catalog[:max_open]]

    allowed = {ev.evidence_id: ev for ev in catalog}
    opened: list[Evidence] = []
    opened_ids: set[str] = set()
    for hit_id in selected_ids:
        ev = allowed.get(hit_id)
        if ev and ev.evidence_id not in opened_ids:
            opened.append(ev)
            opened_ids.add(ev.evidence_id)
        if len(opened) >= max_open:
            break

    answer_prompt = build_tool_search_answer_prompt(item, opened, max_evidence_chars)
    answer = model_client.generate(answer_prompt)
    raw = {
        **answer.raw,
        "tool_search": {
            "search_prompt_chars": len(search_prompt),
            "selection_prompt_chars": len(selection_prompt),
            "answer_prompt_chars": len(answer_prompt),
            "search_plan_output": search_plan.output,
            "search_plan_error": search_plan.error,
            "search_queries": search_queries,
            "search_parse_failed": search_parse_failed,
            "search_results": search_debug,
            "search_errors": search_errors,
            "selection_output": selection.output,
            "selection_error": selection.error,
            "selected_ids": selected_ids,
            "opened_ids": [ev.evidence_id for ev in opened],
            "selection_parse_failed": selection_parse_failed,
        },
    }
    model_error = answer.error or selection.error or search_plan.error
    retrieval_error = "; ".join(search_errors) if search_errors else initial_retrieval.error
    result = ModelResult(
        provider=answer.provider,
        model=answer.model,
        output=answer.output,
        raw=raw,
        error=model_error,
    )
    context_retrieval = RetrievalResult(
        adapter=initial_retrieval.adapter,
        query=item.question,
        evidence=opened,
        debug={**initial_retrieval.debug, "tool_search": True, "search_results": search_debug},
        error=retrieval_error,
    )
    return result, context_retrieval, raw["tool_search"]


def _safe_retrieve_for_tool_search(retriever, adapter: str, item: QAItem) -> RetrievalResult:
    try:
        return retriever.retrieve(item)
    except Exception as exc:
        error = _error("tool_search_retriever_failed", exc)
        return RetrievalResult(
            adapter=adapter,
            query=item.question,
            evidence=[],
            debug={"tool_search_retriever_error": error},
            error=error,
        )


def _evidence_record(ev: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": ev.evidence_id,
        "kind": ev.kind,
        "score": ev.score,
        "metadata": ev.metadata,
        "text": ev.text,
    }


def _completed_key(qa_id: str, retriever: str, context_mode: str, provider: str, model: str) -> tuple[str, str, str, str, str]:
    return (qa_id, retriever, context_mode, provider, model)


def _load_completed_keys(output_path: Path) -> set[tuple[str, str, str, str, str]]:
    completed: set[tuple[str, str, str, str, str]] = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            config = row.get("config", {})
            completed.add(
                _completed_key(
                    str(row.get("qa_id", "")),
                    str(config.get("retriever", "")),
                    str(config.get("context_mode", "")),
                    str(config.get("model_provider", "")),
                    str(config.get("model", "")),
                )
            )
    return completed


def _prepare_retry_errors(output_path: Path) -> tuple[set[tuple[str, str, str, str, str]], dict[str, Any]]:
    if not output_path.exists():
        return set(), {
            "rows_kept_for_retry": 0,
            "rows_pruned_for_retry": 0,
            "duplicate_clean_rows_pruned_for_retry": 0,
            "invalid_lines_preserved_for_retry": 0,
        }
    clean_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    invalid_lines: list[str] = []
    rows_pruned = 0
    duplicate_clean = 0
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line)
                continue
            key = _row_key(row)
            if _row_has_error(row):
                rows_pruned += 1
                continue
            if key in clean_by_key:
                duplicate_clean += 1
            clean_by_key[key] = row
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in clean_by_key.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for line in invalid_lines:
            handle.write(line if line.endswith("\n") else line + "\n")
    tmp_path.replace(output_path)
    return set(clean_by_key), {
        "rows_kept_for_retry": len(clean_by_key),
        "rows_pruned_for_retry": rows_pruned,
        "duplicate_clean_rows_pruned_for_retry": duplicate_clean,
        "invalid_lines_preserved_for_retry": len(invalid_lines),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    config = row.get("config", {})
    return _completed_key(
        str(row.get("qa_id", "")),
        str(config.get("retriever", "")),
        str(config.get("context_mode", "")),
        str(config.get("model_provider", "")),
        str(config.get("model", "")),
    )


def _row_has_error(row: dict[str, Any]) -> bool:
    return bool(row.get("retrieval_error") or row.get("model_error") or row.get("judge_error"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class _UnavailableModelClient:
    def __init__(self, config, build_error: str) -> None:
        self.config = config
        self.build_error = build_error

    def generate(self, prompt: str) -> ModelResult:
        return ModelResult(
            provider=self.config.provider,
            model=self.config.model,
            output="",
            raw={"prompt_chars": len(prompt), "model_build_error": self.build_error},
            error=self.build_error,
        )


def _error(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"
