from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ModelConfig, RetrieverConfig
from .data import load_qa_items
from .models import DryRunModel
from .preflight import _check_retriever
from .retrieval import build_retriever
from .runner import _deferred_retrieval, _generate_for_context, _safe_retrieve


def audit_retrievers(
    config: ExperimentConfig,
    *,
    check_external: bool = True,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Preflight and smoke-test each retriever in every compatible context mode."""

    items = load_qa_items(
        config.dataset.qa_path,
        limit=1,
        qa_ids=config.dataset.qa_ids,
    )
    if not items:
        raise ValueError(f"no audit question found in {config.dataset.qa_path}")
    item = items[0]
    model = DryRunModel(ModelConfig(provider="dry_run", model="rag-audit"))
    rows = [
        _audit_retriever(
            retriever_config,
            config=config,
            item=item,
            model=model,
            check_external=check_external,
            timeout_s=timeout_s,
        )
        for retriever_config in config.retrievers
    ]
    status_counts = Counter(row["status"] for row in rows)
    mode_checks = [check for row in rows for check in row["context_checks"]]
    ready_modes = sum(check["status"] == "ready" for check in mode_checks)
    failed_modes = len(mode_checks) - ready_modes
    all_ready = bool(rows) and status_counts.get("ready", 0) == len(rows)
    return {
        "schema_version": 1,
        "experiment": config.name,
        "question": {"qa_id": item.qa_id, "question": item.question},
        "retrievers": rows,
        "summary": {
            "retrievers": len(rows),
            "ready": status_counts.get("ready", 0),
            "blocked": status_counts.get("blocked", 0),
            "blocked_by_credentials": status_counts.get("blocked_by_credentials", 0),
            "blocked_by_model_service": status_counts.get("blocked_by_model_service", 0),
            "not_checked": status_counts.get("not_checked", 0),
            "failed": status_counts.get("failed", 0),
            "compatible_modes_tested": len(mode_checks),
            "compatible_modes_ready": ready_modes,
            "compatible_modes_failed": failed_modes,
        },
        "ok": all_ready,
        "status": "ready" if all_ready else "incomplete",
    }


def write_rag_audit(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit_retriever(
    retriever_config: RetrieverConfig,
    *,
    config: ExperimentConfig,
    item,
    model: DryRunModel,
    check_external: bool,
    timeout_s: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _check_retriever(
        retriever_config,
        check_external=check_external,
        timeout_s=timeout_s,
        requested_context_modes=list(retriever_config.context_modes),
    )
    row: dict[str, Any] = {
        "name": retriever_config.name,
        "kind": retriever_config.kind,
        "interaction": retriever_config.interaction,
        "supported_context_modes": list(retriever_config.context_modes),
        "status": preflight["status"],
        "problems": list(preflight.get("problems", [])),
        "context_checks": [],
        "preflight": preflight,
    }
    if preflight["status"] != "ready":
        row["duration_s"] = round(time.perf_counter() - started, 3)
        return row

    try:
        retriever = build_retriever(retriever_config, config.dataset.mrag_dir)
    except Exception as exc:
        row["status"] = "failed"
        row["problems"].append(f"retriever build failed: {type(exc).__name__}: {exc}")
        row["duration_s"] = round(time.perf_counter() - started, 3)
        return row

    retrieval_cache = None
    for context_mode in retriever_config.context_modes:
        mode_started = time.perf_counter()
        if context_mode in {"tool_search", "tool_native"}:
            initial = _deferred_retrieval(retriever_config, item, None)
        else:
            if retrieval_cache is None:
                retrieval_cache = _safe_retrieve(retriever_config, retriever, item, None)
            initial = retrieval_cache
        model_result, context_retrieval, _debug = _generate_for_context(
            context_mode,
            model,
            item,
            initial,
            config.max_evidence_chars,
            retriever=retriever,
        )
        errors = [error for error in [initial.error, context_retrieval.error, model_result.error] if error]
        evidence_count = len(context_retrieval.evidence)
        expects_evidence = retriever_config.interaction != "no_retrieval"
        if expects_evidence and evidence_count == 0:
            errors.append("compatible mode returned no evidence")
        check_status = "ready" if not errors else "failed"
        row["context_checks"].append(
            {
                "context_mode": context_mode,
                "status": check_status,
                "evidence_count": evidence_count,
                "errors": errors,
                "duration_s": round(time.perf_counter() - mode_started, 3),
            }
        )

    failed_checks = [check for check in row["context_checks"] if check["status"] != "ready"]
    if failed_checks:
        row["status"] = "failed"
        row["problems"].extend(
            f"{check['context_mode']}: {'; '.join(check['errors'])}" for check in failed_checks
        )
    row["duration_s"] = round(time.perf_counter() - started, 3)
    return row
