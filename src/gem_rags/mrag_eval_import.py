from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import load_chunks, load_figures, load_qa_items, read_jsonl
from .retrieval import evidence_text_from_chunk


def import_mrag_eval(
    mrag_dir: Path,
    output_path: Path,
    *,
    runs_path: Path | None = None,
    scored_path: Path | None = None,
    experiment_name: str = "mrag-prior-eval",
    retriever_name: str = "mrag_reference_prior",
    context_mode: str = "injected",
    grader_name: str = "mrag_prior_judge",
    overwrite: bool = False,
) -> dict[str, Any]:
    runs_path = runs_path or mrag_dir / "eval" / "runs.jsonl"
    scored_path = scored_path or mrag_dir / "eval" / "scored.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    chunks = load_chunks(mrag_dir)
    figures = load_figures(mrag_dir)
    qa_by_id = {item.qa_id: item for item in load_qa_items(mrag_dir / "eval" / "gold_qa.jsonl")}
    chunk_index = _chunk_index(chunks)
    figure_index = {str(figure.get("figure_id")): figure for figure in figures}
    scored_by_key = {_source_key(row): row for row in read_jsonl(scored_path)}

    stats = {
        "input_runs": str(runs_path),
        "input_scored": str(scored_path),
        "output": str(output_path),
        "experiment": experiment_name,
        "retriever": retriever_name,
        "context_mode": context_mode,
        "grader": grader_name,
        "rows_read": 0,
        "rows_written": 0,
        "missing_scores": 0,
        "missing_qa": 0,
        "chunk_evidence": 0,
        "figure_evidence": 0,
        "page_evidence": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imported_at = datetime.now(UTC).isoformat()
    with output_path.open("w", encoding="utf-8") as handle:
        for run_row in read_jsonl(runs_path):
            stats["rows_read"] += 1
            scored_row = scored_by_key.get(_source_key(run_row))
            if scored_row is None:
                stats["missing_scores"] += 1
            qa_item = qa_by_id.get(str(run_row.get("qa_id", "")))
            if qa_item is None:
                stats["missing_qa"] += 1
            row = normalize_mrag_eval_row(
                run_row,
                scored_row,
                qa_item=qa_item,
                chunk_index=chunk_index,
                figure_index=figure_index,
                mrag_dir=mrag_dir,
                experiment_name=experiment_name,
                retriever_name=retriever_name,
                context_mode=context_mode,
                grader_name=grader_name,
                imported_at=imported_at,
            )
            stats["chunk_evidence"] += sum(1 for evidence in row["evidence"] if evidence["kind"] == "chunk")
            stats["figure_evidence"] += sum(1 for evidence in row["evidence"] if evidence["kind"] == "figure")
            stats["page_evidence"] += sum(1 for evidence in row["evidence"] if evidence["kind"] == "page")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
    stats["ok"] = stats["rows_read"] == stats["rows_written"] and stats["missing_scores"] == 0 and stats["missing_qa"] == 0
    return stats


def normalize_mrag_eval_row(
    run_row: dict[str, Any],
    scored_row: dict[str, Any] | None,
    *,
    qa_item: Any | None,
    chunk_index: dict[tuple[str, str, int | None], dict[str, Any]],
    figure_index: dict[str, dict[str, Any]],
    mrag_dir: Path,
    experiment_name: str,
    retriever_name: str,
    context_mode: str,
    grader_name: str,
    imported_at: str,
) -> dict[str, Any]:
    source_config = dict(run_row.get("config") or {})
    model = str(source_config.get("vlm_model_id") or source_config.get("vlm_alias") or "unknown-prior-model")
    model_provider = "qwen" if "qwen" in model.lower() else "mrag_prior"
    evidence = _evidence_from_prior(run_row, chunk_index, figure_index, mrag_dir)
    diagnostics = _diagnostics(qa_item, evidence) if qa_item is not None else {}
    scored_row = scored_row or {}
    return {
        "qa_id": run_row.get("qa_id"),
        "question": run_row.get("question"),
        "question_type": getattr(qa_item, "question_type", None),
        "expected_refusal": getattr(qa_item, "expected_refusal", None),
        "config": {
            "experiment": experiment_name,
            "retriever": retriever_name,
            "context_mode": context_mode,
            "model_provider": model_provider,
            "model": model,
            "grader": grader_name,
            "prompt_style": source_config.get("prompt_style"),
            "vlm_alias": source_config.get("vlm_alias"),
            "source_config": source_config,
        },
        "run": {
            "source": "mrag_eval",
            "imported_at": imported_at,
        },
        "answer": run_row.get("answer") or "",
        "retrieval_error": None,
        "model_error": run_row.get("error"),
        "latency_s": _round_float(run_row.get("latency_s")),
        "evidence": evidence,
        "retrieval_debug": {
            **dict(run_row.get("debug") or {}),
            "source": "mrag_eval",
            "chunks_used": run_row.get("chunks_used", []),
            "figures_used": run_row.get("figures_used", []),
            "pages_used": run_row.get("pages_used", []),
            "provided_evidence_count": len(evidence),
        },
        "judge_scores": scored_row.get("judge_scores", {}),
        "judge_confidence": scored_row.get("judge_confidence"),
        "judge_explanation": scored_row.get("judge_explanation"),
        "figure_metrics": scored_row.get("figure_metrics", {}),
        "system_confidence_breakdown": scored_row.get("system_confidence_breakdown", {}),
        "grader_raw": {
            "imported_mrag_eval": True,
            "diagnostics": diagnostics,
            "rag_answer_length_chars": scored_row.get("rag_answer_length_chars"),
            "rag_n_figures": scored_row.get("rag_n_figures"),
            "images_attached": scored_row.get("images_attached"),
            "judge_latency_s": scored_row.get("judge_latency_s"),
        },
        "judge_error": scored_row.get("judge_error") if scored_row else "missing imported score row",
    }


def _evidence_from_prior(
    run_row: dict[str, Any],
    chunk_index: dict[tuple[str, str, int | None], dict[str, Any]],
    figure_index: dict[str, dict[str, Any]],
    mrag_dir: Path,
) -> list[dict[str, Any]]:
    evidence = []
    for idx, ref in enumerate(run_row.get("chunks_used") or [], 1):
        evidence.append(_chunk_evidence(ref, idx, chunk_index))
    for idx, ref in enumerate(run_row.get("figures_used") or [], 1):
        evidence.append(_figure_evidence(ref, idx, figure_index, mrag_dir))
    for idx, page in enumerate(run_row.get("pages_used") or [], 1):
        evidence.append(_page_evidence(page, idx, mrag_dir))
    return evidence


def _chunk_evidence(ref: dict[str, Any], idx: int, chunk_index: dict[tuple[str, str, int | None], dict[str, Any]]) -> dict[str, Any]:
    chunk = chunk_index.get(_chunk_ref_key(ref))
    score = float(ref.get("score") or 1.0)
    if chunk is None:
        section_id = ref.get("section_id")
        content_type = ref.get("content_type")
        ordinal = ref.get("ordinal")
        text = f"Prior MRAG chunk reference: Section {section_id} {content_type} {ordinal}"
        metadata = {"source": "mrag_eval", **dict(ref)}
        evidence_id = f"mrag_prior:chunk:{idx}"
    else:
        text = evidence_text_from_chunk(chunk)
        metadata = {"source": "mrag_eval", **chunk, "prior_score": score}
        evidence_id = str(chunk.get("chunk_id") or f"mrag_prior:chunk:{idx}")
    return {
        "evidence_id": evidence_id,
        "kind": "chunk",
        "text": text,
        "metadata": metadata,
        "score": score,
    }


def _figure_evidence(ref: dict[str, Any], idx: int, figure_index: dict[str, dict[str, Any]], mrag_dir: Path) -> dict[str, Any]:
    figure_id = str(ref.get("figure_id") or f"figure:{idx}")
    figure = figure_index.get(figure_id, {})
    metadata = {"source": "mrag_eval", **figure, "prior_figure_source": ref.get("source")}
    image_path = _local_figure_path(mrag_dir, metadata)
    if image_path is not None:
        metadata["image_path"] = str(image_path)
    title = figure.get("title") or figure.get("caption") or figure_id
    page = figure.get("page_printed") or figure.get("page_pdf")
    text = f"Prior MRAG visual evidence: {title}"
    if page:
        text += f" (page {page})"
    return {
        "evidence_id": figure_id,
        "kind": "figure",
        "text": text,
        "metadata": metadata,
        "score": 1.0,
    }


def _page_evidence(page: Any, idx: int, mrag_dir: Path) -> dict[str, Any]:
    page_text = str(page)
    image_path = _local_page_path(mrag_dir, page_text)
    metadata: dict[str, Any] = {"source": "mrag_eval", "page": page_text}
    if image_path is not None:
        metadata["image_path"] = str(image_path)
    return {
        "evidence_id": f"page:{page_text}",
        "kind": "page",
        "text": f"Prior MRAG page evidence: page {page_text}",
        "metadata": metadata,
        "score": max(0.0, 1.0 - idx * 0.01),
    }


def _diagnostics(qa_item: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    gold_sections = {str(ref.get("section_id")) for ref in qa_item.references if ref.get("section_id")}
    retrieved_sections = {
        str(ev.get("metadata", {}).get("section_id"))
        for ev in evidence
        if ev.get("metadata", {}).get("section_id")
    }
    gold_refs = {_ref_key(ref) for ref in qa_item.references if ref.get("section_id")}
    retrieved_refs = {_ref_key(ev.get("metadata", {})) for ev in evidence if ev.get("metadata", {}).get("section_id")}
    return {
        "gold_section_recall": len(gold_sections & retrieved_sections) / max(len(gold_sections), 1) if gold_sections else None,
        "gold_reference_recall": len(gold_refs & retrieved_refs) / max(len(gold_refs), 1) if gold_refs else None,
        "n_evidence": len(evidence),
        "n_gold_sections": len(gold_sections),
        "n_gold_references": len(gold_refs),
    }


def _source_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("qa_id")), json.dumps(row.get("config") or {}, sort_keys=True, ensure_ascii=False)


def _chunk_index(chunks: list[dict[str, Any]]) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    return {_chunk_ref_key(chunk): chunk for chunk in chunks}


def _chunk_ref_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return str(row.get("section_id")), str(row.get("content_type", "")).lower(), _optional_int(row.get("ordinal"))


def _ref_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return str(row.get("section_id")), str(row.get("content_type", "")).lower(), _optional_int(row.get("ordinal"))


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def _local_figure_path(mrag_dir: Path, figure: dict[str, Any]) -> Path | None:
    raw_path = str(figure.get("image_path") or "")
    if raw_path:
        candidate = mrag_dir / "figures" / Path(raw_path).name
        if candidate.exists():
            return candidate
    return None


def _local_page_path(mrag_dir: Path, page: str) -> Path | None:
    try:
        number = int(page)
    except ValueError:
        return None
    candidate = mrag_dir / "page_images" / f"page_{number:04d}.png"
    return candidate if candidate.exists() else None
