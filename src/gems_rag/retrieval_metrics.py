from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

from .data import canonicalize_chunks

SECTION_RANGE_RE = re.compile(
    r"^(?P<start_prefix>\d+[A-Z]+)\.(?P<start>\d+)-"
    r"(?P<end_prefix>\d+[A-Z]+)\.(?P<end>\d+)$"
)


def build_chunk_qrels(
    gold_rows: Iterable[dict[str, Any]],
    corpus_rows: Iterable[dict[str, Any]],
    *,
    require_nonempty: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus, canonicalization = canonicalize_chunks(corpus_rows)
    known_sections = {
        str(row.get("section_id") or "").strip()
        for row in corpus
        if str(row.get("section_id") or "").strip()
    }
    qrels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for gold in gold_rows:
        question_id = str(gold.get("question_id") or gold.get("qa_id") or "").strip()
        if not question_id:
            raise ValueError("every gold row requires question_id or qa_id")
        if question_id in seen_ids:
            raise ValueError(f"duplicate gold question ID: {question_id}")
        seen_ids.add(question_id)

        source_pages = {
            int(value)
            for value in gold.get("source_pdf_pages") or []
            if _integer_or_none(value) is not None
        }
        section_annotations = [
            str(value).strip()
            for value in gold.get("sections") or []
            if str(value).strip()
        ]
        sections = expand_section_annotations(section_annotations, known_sections)
        tables = _string_set(gold.get("tables") or [])
        figures = _string_set(gold.get("figures") or [])

        relevant_chunks: list[dict[str, Any]] = []
        for chunk in corpus:
            reasons = chunk_relevance_reasons(
                chunk,
                source_pages=source_pages,
                sections=sections,
                tables=tables,
                figures=figures,
            )
            if not reasons:
                continue
            relevant_chunks.append(
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "section_id": chunk.get("section_id"),
                    "page_pdf": chunk.get("page_pdf"),
                    "page_printed": chunk.get("page_printed"),
                    "figure_refs": list(chunk.get("figure_refs") or []),
                    "table_refs": list(chunk.get("table_refs") or []),
                    "relevance": 1,
                    "matching_reasons": reasons,
                }
            )

        if require_nonempty and not relevant_chunks:
            raise ValueError(f"gold annotations resolve to no canonical chunks: {question_id}")
        qrels.append(
            {
                "schema_version": 1,
                "benchmark_id": gold.get("benchmark_version"),
                "question_id": question_id,
                "question": gold.get("question"),
                "answerable": gold.get("answerable"),
                "primary_modality": gold.get("primary_modality"),
                "split": gold.get("split"),
                "gold_annotations": {
                    "source_pdf_pages": sorted(source_pages),
                    "printed_manual_pages": list(
                        gold.get("printed_manual_pages") or []
                    ),
                    "sections": section_annotations,
                    "expanded_sections": sorted(sections),
                    "tables": sorted(tables),
                    "figures": sorted(figures),
                },
                "relevant_chunk_count": len(relevant_chunks),
                "text_retrieval_evaluable": bool(relevant_chunks),
                "relevant_chunk_ids": [
                    item["chunk_id"] for item in relevant_chunks
                ],
                "relevant_chunks": relevant_chunks,
            }
        )

    report = {
        "schema_version": 1,
        "gold_questions": len(qrels),
        "canonical_corpus": canonicalization,
        "qrels_min": min(
            (row["relevant_chunk_count"] for row in qrels), default=0
        ),
        "qrels_max": max(
            (row["relevant_chunk_count"] for row in qrels), default=0
        ),
        "qrels_mean": (
            sum(row["relevant_chunk_count"] for row in qrels) / len(qrels)
            if qrels
            else 0.0
        ),
        "text_retrieval_evaluable_questions": sum(
            bool(row["relevant_chunk_count"]) for row in qrels
        ),
        "text_retrieval_unevaluable_questions": sum(
            not row["relevant_chunk_count"] for row in qrels
        ),
        "text_retrieval_unevaluable_question_ids": [
            row["question_id"]
            for row in qrels
            if not row["relevant_chunk_count"]
        ],
    }
    return qrels, report


def expand_section_annotations(
    annotations: Iterable[str],
    known_sections: set[str],
) -> set[str]:
    expanded: set[str] = set()
    for annotation in annotations:
        value = str(annotation).strip()
        if not value:
            continue
        match = SECTION_RANGE_RE.fullmatch(value)
        if match is None:
            expanded.add(value)
            continue
        start_prefix = match.group("start_prefix")
        end_prefix = match.group("end_prefix")
        if start_prefix != end_prefix:
            raise ValueError(f"section range crosses prefixes: {value}")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if end < start:
            raise ValueError(f"section range is descending: {value}")
        matches = {
            section
            for section in known_sections
            if _section_in_range(section, start_prefix, start, end)
        }
        if not matches:
            raise ValueError(f"section range does not match the corpus: {value}")
        expanded.update(matches)
    return expanded


def chunk_relevance_reasons(
    chunk: dict[str, Any],
    *,
    source_pages: set[int],
    sections: set[str],
    tables: set[str],
    figures: set[str],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    page_pdf = _integer_or_none(chunk.get("page_pdf"))
    if page_pdf is not None and page_pdf in source_pages:
        reasons.append({"field": "source_pdf_pages", "value": page_pdf})

    section_id = str(chunk.get("section_id") or "").strip()
    if section_id and section_id in sections:
        reasons.append({"field": "sections", "value": section_id})

    for table in sorted(_string_set(chunk.get("table_refs") or []) & tables):
        reasons.append({"field": "tables", "value": table})
    for figure in sorted(_string_set(chunk.get("figure_refs") or []) & figures):
        reasons.append({"field": "figures", "value": figure})
    return reasons


def score_canonical_rankings(
    rankings: Iterable[dict[str, Any]],
    qrels: Iterable[dict[str, Any]],
    *,
    depth: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    qrels_by_id = _unique_by_question_id(qrels, label="qrels")
    ranking_rows = list(rankings)
    expected_pairs = {
        (str(row.get("question_id") or ""), str(row.get("method") or ""))
        for row in ranking_rows
    }
    if len(expected_pairs) != len(ranking_rows):
        raise ValueError("duplicate (question_id, method) ranking rows")

    scores: list[dict[str, Any]] = []
    native_sensitivity: list[dict[str, Any]] = []
    for row in ranking_rows:
        question_id = str(row.get("question_id") or "")
        method = str(row.get("method") or "")
        if question_id not in qrels_by_id:
            raise ValueError(f"ranking question is missing qrels: {question_id}")
        qrel = qrels_by_id[question_id]
        relevant_ids = set(qrel["relevant_chunk_ids"])
        reasons_by_id = {
            item["chunk_id"]: item["matching_reasons"]
            for item in qrel["relevant_chunks"]
        }
        ranked_ids = [
            str(item.get("canonical_chunk_id") or "")
            for item in (row.get("canonical_chunk_ranking_top10") or [])[:depth]
            if str(item.get("canonical_chunk_id") or "")
        ]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(
                f"canonical ranking contains duplicate chunk IDs: {question_id}/{method}"
            )
        metrics = binary_ranking_metrics(
            ranked_ids,
            relevant_ids,
            depth=depth,
        )
        ranked_audit = [
            {
                "rank": rank,
                "canonical_chunk_id": chunk_id,
                "relevant": chunk_id in relevant_ids,
                "matching_reasons": reasons_by_id.get(chunk_id, []),
            }
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ]
        scores.append(
            {
                "schema_version": 1,
                "question_id": question_id,
                "method": method,
                "display_name": row.get("display_name") or method,
                "primary_modality": qrel.get("primary_modality"),
                "answerable": qrel.get("answerable"),
                "split": qrel.get("split"),
                "scoring_view": "canonical_chunk_ranking_top10",
                **metrics,
                "ranked_chunks": ranked_audit,
            }
        )
        native_sensitivity.append(
            _score_native_context(
                row,
                qrel,
                reasons_by_id=reasons_by_id,
                depth=depth,
            )
        )

    summary = aggregate_retrieval_scores(scores, depth=depth)
    return scores, summary, native_sensitivity


def binary_ranking_metrics(
    ranked_ids: Iterable[str],
    relevant_ids: set[str],
    *,
    depth: int = 10,
) -> dict[str, Any]:
    ranking = list(ranked_ids)[:depth]
    if not relevant_ids:
        return {
            "depth": depth,
            "evaluable": False,
            "retrieved_count": len(ranking),
            "relevant_count": 0,
            "retrieved_relevant_count": 0,
            "hit_at_10": None,
            "recall_at_10": None,
            "mrr_at_10": None,
            "ndcg_at_10": None,
            "first_relevant_rank": None,
            "relevant_ranks": [],
            "dcg_at_10": None,
            "ideal_dcg_at_10": None,
        }
    relevant_ranks = [
        rank
        for rank, item_id in enumerate(ranking, start=1)
        if item_id in relevant_ids
    ]
    retrieved_relevant = {
        item_id for item_id in ranking if item_id in relevant_ids
    }
    relevance = [1 if item_id in relevant_ids else 0 for item_id in ranking]
    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(relevance, start=1)
    )
    ideal_relevant = min(len(relevant_ids), depth)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    first_rank = relevant_ranks[0] if relevant_ranks else None
    return {
        "depth": depth,
        "evaluable": True,
        "retrieved_count": len(ranking),
        "relevant_count": len(relevant_ids),
        "retrieved_relevant_count": len(retrieved_relevant),
        "hit_at_10": 1.0 if relevant_ranks else 0.0,
        "recall_at_10": (
            len(retrieved_relevant) / len(relevant_ids)
            if relevant_ids
            else 0.0
        ),
        "mrr_at_10": 1.0 / first_rank if first_rank is not None else 0.0,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "first_relevant_rank": first_rank,
        "relevant_ranks": relevant_ranks,
        "dcg_at_10": dcg,
        "ideal_dcg_at_10": idcg,
    }


def aggregate_retrieval_scores(
    scores: Iterable[dict[str, Any]],
    *,
    depth: int = 10,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        grouped[str(score["method"])].append(score)

    methods = []
    for method, rows in sorted(grouped.items()):
        evaluable = [row for row in rows if row["evaluable"]]
        methods.append(
            {
                "method": method,
                "display_name": rows[0].get("display_name") or method,
                "questions": len(rows),
                "evaluable_questions": len(evaluable),
                "unevaluable_questions": len(rows) - len(evaluable),
                "recall_at_10": _mean(evaluable, "recall_at_10"),
                "mrr_at_10": _mean(evaluable, "mrr_at_10"),
                "ndcg_at_10": _mean(evaluable, "ndcg_at_10"),
                "hit_rate_at_10": _mean(evaluable, "hit_at_10"),
                "micro_recall_at_10": (
                    sum(row["retrieved_relevant_count"] for row in evaluable)
                    / sum(row["relevant_count"] for row in evaluable)
                    if evaluable
                    else 0.0
                ),
                "mean_relevant_chunks": _mean(evaluable, "relevant_count"),
                "mean_retrieved_chunks": _mean(evaluable, "retrieved_count"),
            }
        )
    return {
        "schema_version": 1,
        "scoring_view": "canonical_chunk_ranking_top10",
        "depth": depth,
        "relevance": (
            "Binary canonical-chunk relevance: PDF page OR section OR table "
            "reference OR figure reference matches locked gold."
        ),
        "recall_definition": (
            "Macro mean of true set chunk Recall@10; each question denominator "
            "is its non-empty finite canonical qrel set. Figure-only questions "
            "with no canonical text qrels are reported and excluded."
        ),
        "mrr_definition": "Macro mean reciprocal rank of the first relevant chunk through rank 10.",
        "ndcg_definition": "Macro mean binary nDCG@10.",
        "methods": methods,
    }


def _score_native_context(
    ranking: dict[str, Any],
    qrel: dict[str, Any],
    *,
    reasons_by_id: dict[str, list[dict[str, Any]]],
    depth: int,
) -> dict[str, Any]:
    relevant_ids = set(qrel["relevant_chunk_ids"])
    ranked = (ranking.get("native_context_ranking") or [])[:depth]
    contexts = []
    covered: set[str] = set()
    first_relevant_rank = None
    for rank, item in enumerate(ranked, start=1):
        mapped = [
            str(value)
            for value in item.get("canonical_chunk_ids") or []
            if str(value)
        ]
        matched = sorted(set(mapped) & relevant_ids)
        covered.update(matched)
        if matched and first_relevant_rank is None:
            first_relevant_rank = rank
        contexts.append(
            {
                "rank": rank,
                "evidence_id": item.get("evidence_id"),
                "kind": item.get("kind"),
                "group": item.get("group"),
                "mapped_canonical_chunk_ids": mapped,
                "matched_relevant_chunk_ids": matched,
                "matching_reasons": {
                    chunk_id: reasons_by_id.get(chunk_id, [])
                    for chunk_id in matched
                },
                "relevant": bool(matched),
            }
        )
    return {
        "schema_version": 1,
        "question_id": qrel["question_id"],
        "method": ranking.get("method"),
        "display_name": ranking.get("display_name"),
        "scoring_view": "native_context_ranking_top10_sensitivity",
        "depth": depth,
        "retrieved_context_count": len(contexts),
        "relevant_context_count": sum(item["relevant"] for item in contexts),
        "unique_relevant_chunks_covered": len(covered),
        "relevant_chunk_count": len(relevant_ids),
        "evaluable": bool(relevant_ids),
        "chunk_coverage_recall_at_10": (
            len(covered) / len(relevant_ids) if relevant_ids else None
        ),
        "hit_at_10": (
            1.0 if first_relevant_rank is not None else 0.0
        )
        if relevant_ids
        else None,
        "mrr_at_10": (
            (
                1.0 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )
            if relevant_ids
            else None
        ),
        "first_relevant_rank": first_relevant_rank,
        "note": (
            "Native GraphRAG units are heterogeneous and do not have a finite "
            "corpus-wide qrel universe, so native-unit nDCG is intentionally "
            "not reported."
        ),
        "contexts": contexts,
    }


def _unique_by_question_id(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"{label} row is missing question_id")
        if question_id in result:
            raise ValueError(f"duplicate {label} question ID: {question_id}")
        result[question_id] = row
    return result


def _section_in_range(
    section: str,
    prefix: str,
    start: int,
    end: int,
) -> bool:
    match = re.fullmatch(r"(?P<prefix>\d+[A-Z]+)\.(?P<number>\d+)", section)
    return bool(
        match
        and match.group("prefix") == prefix
        and start <= int(match.group("number")) <= end
    )


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_set(values: Iterable[Any]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0
