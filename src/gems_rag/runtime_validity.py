from __future__ import annotations

import re
from typing import Any


SUCCESS_STATUSES = {"success", "successful", "complete", "completed", "ok"}
FAILED_PROVIDER_STATUSES = {"cancelled", "canceled", "failed", "incomplete", "error"}
TRUNCATION_REASONS = {
    "length",
    "max_token",
    "max_tokens",
    "max_output_token",
    "max_output_tokens",
    "token_limit",
}
_HEADING_ONLY = re.compile(
    r"^(?:#{1,6}\s*)?(?:answer|direct answer|evidence|retrieved evidence|"
    r"standards|guidance|options|support|citations?)\s*:?\s*$",
    flags=re.IGNORECASE,
)
_CAPTION_ONLY = re.compile(
    r"^(?:figure|fig\.?|table|evidence|image)\s+[A-Za-z0-9_.-]+\s*:?\s*$",
    flags=re.IGNORECASE,
)


def operational_row_problems(
    row: dict[str, Any],
    *,
    require_serialized_return: bool = True,
    require_run_status: bool = True,
    require_retrieval_trace: bool = True,
) -> list[str]:
    problems: list[str] = []
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question:
        problems.append("missing question record")
    if not answer:
        problems.append("saved answer is empty")
    elif _looks_like_heading_or_caption(answer):
        problems.append("saved answer is only a heading or evidence caption")

    serialized = row.get("serialized_return")
    if require_serialized_return:
        if not isinstance(serialized, dict):
            problems.append("serialized_return is missing")
        else:
            serialized_answer = str(serialized.get("answer") or "").strip()
            if not serialized_answer:
                problems.append("serialized_return.answer is empty")
            elif answer and serialized_answer != answer:
                problems.append("serialized_return.answer does not match the saved answer")

    if require_run_status:
        status = str(row.get("run_status") or "").strip().lower()
        if status not in SUCCESS_STATUSES:
            problems.append(f"run status is not successful: {status or 'missing'}")

    for field in ("retrieval_error", "model_error", "judge_error"):
        if row.get(field):
            problems.append(f"unresolved {field}")

    provider_problems = provider_completion_problems(row.get("model_raw"))
    if provider_problems:
        problems.append("provider completion is invalid: " + ", ".join(provider_problems))

    if require_retrieval_trace:
        problems.extend(_retrieval_trace_problems(row))
    return problems


def provider_completion_problems(value: Any) -> list[str]:
    reasons: list[str] = []

    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key).lower())
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, key)
            return
        normalized = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            return
        if key in {"finish_reason", "stop_reason", "incomplete_reason"} and normalized in TRUNCATION_REASONS:
            marker = f"{key}={normalized}"
            if marker not in reasons:
                reasons.append(marker)
        if key == "status" and normalized in FAILED_PROVIDER_STATUSES:
            marker = f"status={normalized}"
            if marker not in reasons:
                reasons.append(marker)

    visit(value)
    return reasons


def _looks_like_heading_or_caption(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    return all(_HEADING_ONLY.fullmatch(line) or _CAPTION_ONLY.fullmatch(line) for line in lines)


def _retrieval_trace_problems(row: dict[str, Any]) -> list[str]:
    evidence = row.get("evidence")
    debug = row.get("retrieval_debug")
    if not isinstance(evidence, list):
        return ["retrieval evidence log is missing"]
    if not isinstance(debug, dict):
        return ["retrieval debug log is missing"]

    problems = []
    retrieved = debug.get("retrieved_evidence_count")
    provided = debug.get("provided_evidence_count")
    if not isinstance(retrieved, int) or retrieved < 0:
        problems.append("retrieved_evidence_count is missing or invalid")
    if not isinstance(provided, int) or provided < 0:
        problems.append("provided_evidence_count is missing or invalid")
    elif provided != len(evidence):
        problems.append(
            f"provided_evidence_count={provided} does not match evidence rows={len(evidence)}"
        )
    if isinstance(retrieved, int) and isinstance(provided, int) and retrieved < provided:
        problems.append("retrieved_evidence_count is smaller than provided_evidence_count")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            problems.append(f"evidence row {index} is not an object")
            continue
        if not str(item.get("evidence_id") or "").strip():
            problems.append(f"evidence row {index} has no evidence_id")
        if not str(item.get("kind") or "").strip():
            problems.append(f"evidence row {index} has no kind")
        if not str(item.get("text") or "").strip() and not _is_successful_empty_result_trace(
            row,
            evidence,
            index,
            item,
        ):
            problems.append(f"evidence row {index} has no text")
        if not isinstance(item.get("metadata"), dict):
            problems.append(f"evidence row {index} has no metadata object")
    return problems


def _is_successful_empty_result_trace(
    row: dict[str, Any],
    evidence: list[Any],
    index: int,
    item: dict[str, Any],
) -> bool:
    """Recognize an external RAG's explicit, successful no-result sentinel."""
    config = row.get("config")
    metadata = item.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        return False
    qa_id = str(row.get("qa_id") or "").strip()
    retriever = str(config.get("retriever") or "").strip()
    return (
        len(evidence) == 1
        and index == 0
        and bool(qa_id)
        and bool(retriever)
        and item.get("kind") == "tool_trace"
        and str(item.get("evidence_id") or "") == f"{retriever}:{qa_id}"
        and metadata.get("parsed_json") is True
        and metadata.get("returncode") == 0
        and not row.get("retrieval_error")
    )
