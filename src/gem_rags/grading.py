from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .config import GraderConfig, ModelConfig
from .models import LLM_MODEL_PROVIDERS, ModelClient, build_model
from .types import GradingResult, ModelResult, QAItem, RetrievalResult

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


def grade_answer(
    config: GraderConfig,
    item: QAItem,
    model_result: ModelResult,
    retrieval: RetrievalResult,
    *,
    model_client: ModelClient | None = None,
) -> GradingResult:
    if config.provider == "heuristic":
        return heuristic_grade(config, item, model_result, retrieval)
    if config.provider in LLM_MODEL_PROVIDERS:
        return llm_grade(config, item, model_result, retrieval, model_client=model_client)
    raise ValueError(f"unknown grader provider: {config.provider}")


def heuristic_grade(config: GraderConfig, item: QAItem, model_result: ModelResult, retrieval: RetrievalResult) -> GradingResult:
    answer = model_result.output.lower()
    gold = json.dumps(item.gold_answer, ensure_ascii=False).lower()
    answer_terms = Counter(_tokens(answer))
    gold_terms = set(_tokens(gold))
    overlap = sum(1 for term in gold_terms if answer_terms[term] > 0)
    lexical_recall = overlap / max(len(gold_terms), 1)

    retrieved_sections = {str(ev.metadata.get("section_id")) for ev in retrieval.evidence if ev.metadata.get("section_id")}
    gold_sections = {str(ref.get("section_id")) for ref in item.references if ref.get("section_id")}
    section_recall = len(retrieved_sections & gold_sections) / max(len(gold_sections), 1) if gold_sections else None
    retrieved_ref_keys = {_ref_key(ev.metadata) for ev in retrieval.evidence if ev.metadata.get("section_id")}
    gold_ref_keys = {_ref_key(ref) for ref in item.references if ref.get("section_id")}
    ref_recall = len(retrieved_ref_keys & gold_ref_keys) / max(len(gold_ref_keys), 1) if gold_ref_keys else None
    retrieved_types = {str(ev.metadata.get("content_type")).lower() for ev in retrieval.evidence if ev.metadata.get("content_type")}
    gold_types = {str(ref.get("content_type")).lower() for ref in item.references if ref.get("content_type")}
    category_recall = len(retrieved_types & gold_types) / max(len(gold_types), 1) if gold_types else None
    refusal_terms = {"does not answer", "not answer", "out of scope", "consult", "state standards", "state-specific"}
    refusal_detected = None if model_result.raw.get("dry_run") else any(term in answer for term in refusal_terms)
    figure_metrics = _figure_metrics(item, retrieval)
    system_confidence = _system_confidence(retrieval, section_recall)
    scores = {
        "factual_accuracy": _score_obj(
            _to_five(lexical_recall),
            f"heuristic lexical overlap with gold answer: {lexical_recall:.3f}",
        ),
        "category_correctness": _score_obj(
            _to_five(category_recall) if category_recall is not None else None,
            "retrieved MUTCD rule categories compared with gold reference categories"
            if category_recall is not None
            else "no gold reference categories",
        ),
        "citation_validity": _score_obj(
            _to_five(ref_recall) if ref_recall is not None else None,
            "retrieved exact section/content/ordinal keys compared with gold references"
            if ref_recall is not None
            else "no gold references",
        ),
        "verbatim_faithfulness": _score_obj(
            _to_five(lexical_recall),
            "heuristic token overlap proxy; use an LLM grader for real quote/paraphrase judgment",
        ),
        "completeness": _score_obj(
            _to_five(section_recall) if section_recall is not None else _to_five(lexical_recall),
            "gold section recall from retrieved evidence"
            if section_recall is not None
            else "no gold sections; fell back to lexical overlap",
        ),
        "refusal_appropriateness": _refusal_score(item.expected_refusal, refusal_detected),
        "figure_relevance": _figure_score(figure_metrics, "relevance"),
        "figure_grounding": _figure_score(figure_metrics, "grounding"),
    }
    diagnostics = {
        "lexical_gold_recall": round(lexical_recall, 4),
        "gold_section_recall": section_recall,
        "gold_reference_recall": ref_recall,
        "gold_category_recall": category_recall,
        "expected_refusal": item.expected_refusal,
        "refusal_detected": refusal_detected,
        "model_error": model_result.error,
        "n_evidence": len(retrieval.evidence),
    }
    return GradingResult(
        grader=config.model,
        scores=scores,
        raw={"heuristic": True, "diagnostics": diagnostics},
        confidence=system_confidence["system_confidence"],
        explanation="Deterministic smoke-test grader; use the configured LLM grader for final judgments.",
        figure_metrics=figure_metrics,
        system_confidence_breakdown=system_confidence,
    )


def llm_grade(
    config: GraderConfig,
    item: QAItem,
    model_result: ModelResult,
    retrieval: RetrievalResult,
    *,
    model_client: ModelClient | None = None,
) -> GradingResult:
    prompt = build_llm_grader_prompt(config, item, model_result, retrieval)
    model = model_client
    if model is None:
        model = build_model(ModelConfig(provider=config.provider, model=config.model, options=config.options))
    result = model.generate(prompt)
    if result.error:
        return GradingResult(config.model, {}, error=result.error, raw=result.raw)
    parsed, parse_error = parse_grader_output(result.output)
    scores = normalize_judge_scores(parsed)
    figure_metrics = (
        parsed.get("figure_metrics")
        if isinstance(parsed, dict) and isinstance(parsed.get("figure_metrics"), dict)
        else _figure_metrics(item, retrieval)
    )
    return GradingResult(
        config.model,
        scores,
        raw={"model_raw": result.raw, "parsed": parsed, "prompt_chars": len(prompt), "parse_error": parse_error},
        error=parse_error,
        confidence=_clamp_float(parsed.get("judge_confidence")) if isinstance(parsed, dict) else None,
        explanation=parsed.get("judge_explanation") if isinstance(parsed, dict) else None,
        figure_metrics=figure_metrics,
    )


def build_llm_grader_prompt(
    config: GraderConfig,
    item: QAItem,
    model_result: ModelResult,
    retrieval: RetrievalResult,
) -> str:
    evidence = _grader_evidence_payload(
        retrieval,
        max_items=int(config.options.get("max_evidence_items", 20)),
        max_text_chars=int(config.options.get("max_evidence_text_chars", 1200)),
    )
    schema_scores = ",\n    ".join(
        f'"{key}": {{"score": 0-5 or null, "note": "short rationale"}}'
        for key in RUBRIC_KEYS
    )
    payload = {
        "question": item.question,
        "question_type": item.question_type,
        "expected_refusal": item.expected_refusal,
        "gold_answer": item.gold_answer,
        "gold_references": item.references,
        "gold_figures": item.gold_figures,
        "retrieval_adapter": retrieval.adapter,
        "retrieved_evidence": evidence,
        "rag_answer": model_result.output,
        "model_error": model_result.error,
    }
    return f"""You are grading an MUTCD retrieval-augmented answer.
Grade the answer against the gold answer and references. Use retrieved_evidence
to judge grounding, citation validity, quote/paraphrase faithfulness, and figure
grounding. Penalize claims that are not supported by the retrieved evidence.
Do not reward an answer for merely retrieving correct evidence if the answer does
not use it correctly.

Return only compact JSON with this schema:
{{
  "judge_scores": {{
    {schema_scores}
  }},
  "judge_confidence": 0.0-1.0,
  "judge_explanation": "one paragraph",
  "figure_metrics": {{"figure_recall": null, "figure_precision": null, "n_gold_figures": 0, "n_rag_figures": 0, "n_intersection": 0}}
}}
Use null when a rubric does not apply. Every judge_scores key must be present.

Evaluation payload JSON:
{json.dumps(payload, ensure_ascii=False)}
"""


def parse_grader_output(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(text)
        return (parsed, None) if isinstance(parsed, dict) else ({"raw_value": parsed}, "grader_json_not_object")
    except json.JSONDecodeError:
        pass
    for candidate in _json_object_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return (parsed, None) if isinstance(parsed, dict) else ({"raw_value": parsed}, "grader_json_not_object")
    return {"raw_text": text}, "grader_json_parse_failed"


def normalize_judge_scores(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_scores = parsed.get("judge_scores") if isinstance(parsed.get("judge_scores"), dict) else parsed
    scores: dict[str, Any] = {}
    for key in RUBRIC_KEYS:
        raw_value = raw_scores.get(key) if isinstance(raw_scores, dict) else None
        scores[key] = _normalize_score_obj(raw_value)
    return scores


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _grader_evidence_payload(retrieval: RetrievalResult, *, max_items: int, max_text_chars: int) -> list[dict[str, Any]]:
    payload = []
    for ev in retrieval.evidence[:max_items]:
        text = ev.text
        if len(text) > max_text_chars:
            text = text[: max_text_chars - 3].rstrip() + "..."
        payload.append(
            {
                "evidence_id": ev.evidence_id,
                "kind": ev.kind,
                "score": ev.score,
                "metadata": ev.metadata,
                "text": text,
            }
        )
    return payload


def _json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _normalize_score_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        score = _normalize_score(value.get("score"))
        note = value.get("note") or value.get("rationale") or value.get("explanation") or ""
        return {"score": score, "note": str(note)}
    return {"score": _normalize_score(value), "note": ""}


def _normalize_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return max(0, min(5, int(round(float(value)))))
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "null", "none", "n/a", "na", "not applicable"}:
            return None
        try:
            return _normalize_score(float(stripped))
        except ValueError:
            return None
    return None


def _clamp_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return _clamp_float(float(value))
        except ValueError:
            return None
    return None


def _ref_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("section_id", "")),
        str(record.get("content_type", "")).lower(),
        str(record.get("ordinal", "")).zfill(2),
    )


def _score_obj(score: int | None, note: str) -> dict[str, Any]:
    return {"score": score, "note": note}


def _to_five(value: float | None) -> int | None:
    if value is None:
        return None
    return max(0, min(5, int(round(value * 5))))


def _refusal_score(expected_refusal: bool, refusal_detected: bool | None) -> dict[str, Any]:
    if refusal_detected is None:
        return _score_obj(None, "dry-run answer; refusal behavior was not evaluated")
    if expected_refusal:
        return _score_obj(5 if refusal_detected else 1, "question is a refusal trap")
    return _score_obj(None, "not a refusal trap")


def _figure_metrics(item: QAItem, retrieval: RetrievalResult) -> dict[str, Any]:
    gold = {_figure_id(value) for value in item.gold_figures}
    gold.discard("")
    rag = {ev.evidence_id for ev in retrieval.evidence if ev.kind == "figure"}
    intersection = gold & rag
    return {
        "figure_recall": (len(intersection) / len(gold)) if gold else None,
        "figure_precision": (len(intersection) / len(rag)) if rag else None,
        "n_gold_figures": len(gold),
        "n_rag_figures": len(rag),
        "n_intersection": len(intersection),
    }


def _figure_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("figure_id") or value.get("id") or "")
    return str(value)


def _figure_score(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if metrics["n_gold_figures"] == 0 and metrics["n_rag_figures"] == 0:
        return _score_obj(None, f"no figure {label} target")
    if metrics["n_gold_figures"] == 0:
        return _score_obj(0, f"retrieved figures when no figures were required")
    recall = metrics["figure_recall"] or 0.0
    precision = metrics["figure_precision"] if metrics["figure_precision"] is not None else recall
    return _score_obj(_to_five((recall + precision) / 2), f"heuristic figure {label} proxy")


def _system_confidence(retrieval: RetrievalResult, section_recall: float | None) -> dict[str, Any]:
    scores = sorted((ev.score for ev in retrieval.evidence), reverse=True)
    top = scores[0] if scores else 0.0
    mean_top3 = sum(scores[:3]) / min(len(scores), 3) if scores else 0.0
    section_component = section_recall if section_recall is not None else 0.0
    return {
        "system_confidence": round(section_component, 4),
        "sc_top_chunk_score": top,
        "sc_mean_top3": mean_top3,
        "sc_fraction_cited": section_component,
    }
