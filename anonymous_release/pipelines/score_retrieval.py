#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comparison_support.data import canonicalize_chunks
from comparison_support.retrieval_metrics import (
    build_chunk_qrels,
    score_canonical_rankings,
)


DEFAULT_RESULTS = ROOT / "runs" / "comparison" / "results.jsonl"
DEFAULT_GOLD = ROOT / "benchmark" / "gold.jsonl"
DEFAULT_CORPUS = (
    ROOT / "indexes" / "gems-rag" / "mmrag_cache_v3" / "chunks.jsonl"
)
DEFAULT_GRAPH_INPUT = (
    ROOT / "indexes" / "graphrag" / "input" / "mutcd_chunks.txt"
)
DEFAULT_OUTPUT = ROOT / "runs" / "comparison" / "retrieval_metrics"
DISPLAY_NAMES = {
    "bm25": "BM25",
    "graphrag": "GraphRAG",
    "paperqa": "PaperQA",
    "gems-rag": "GEMS-RAG",
}
MARKER_RE = re.compile(r"(?m)^---\s+(MUTCD11e_[A-Za-z0-9_]+)\s+---\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class GraphChunkResolver:
    def __init__(self, input_path: Path, valid_chunk_ids: set[str]) -> None:
        self.text = input_path.read_text(encoding="utf-8")
        self.valid_chunk_ids = valid_chunk_ids
        self.markers = [
            (match.start(), match.group(1))
            for match in MARKER_RE.finditer(self.text)
            if match.group(1) in valid_chunk_ids
        ]
        self.positions = [position for position, _ in self.markers]

    def resolve(self, context_text: str) -> tuple[list[str], str]:
        start = self.text.find(context_text)
        if start < 0:
            explicit = [
                match.group(1)
                for match in MARKER_RE.finditer(context_text)
                if match.group(1) in self.valid_chunk_ids
            ]
            return list(dict.fromkeys(explicit)), "explicit_markers_only"
        end = start + len(context_text)
        first = max(0, bisect.bisect_right(self.positions, start) - 1)
        resolved = []
        for position, chunk_id in self.markers[first:]:
            if position >= end:
                break
            resolved.append(chunk_id)
        return list(dict.fromkeys(resolved)), "exact_input_span"


def adapter_contexts(method: str, output: dict[str, Any]) -> list[dict[str, Any]]:
    if method == "gems-rag":
        return [
            {
                "name": row.get("chunk_id"),
                "kind": "chunk",
                "text": row.get("text"),
                "score": row.get("score"),
                "metadata": row,
            }
            for row in (output.get("chunks") or [])[:10]
        ]
    contexts = output.get("contexts") or []
    return [row for row in contexts[:10] if isinstance(row, dict)]


def canonical_ids(
    method: str,
    context: dict[str, Any],
    *,
    corpus_by_id: dict[str, dict[str, Any]],
    graph_resolver: GraphChunkResolver,
) -> tuple[list[str], str]:
    metadata = context.get("metadata") or {}
    if method == "graphrag":
        if metadata.get("graph_group") != "sources":
            return [], "not_a_source_text_unit"
        return graph_resolver.resolve(str(context.get("text") or ""))

    candidates = (
        metadata.get("chunk_id"),
        metadata.get("source_name"),
        context.get("name"),
        context.get("evidence_id"),
    )
    resolved = [
        str(candidate)
        for candidate in candidates
        if candidate is not None and str(candidate) in corpus_by_id
    ]
    return list(dict.fromkeys(resolved)), "direct_identifier"


def normalize_results(
    result_rows: list[dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
    graph_resolver: GraphChunkResolver,
) -> list[dict[str, Any]]:
    normalized = []
    seen: set[tuple[str, str]] = set()
    for result in result_rows:
        if result.get("status") != "ok":
            continue
        question_id = str(result.get("question_id") or "")
        method = str(result.get("method") or "")
        pair = (question_id, method)
        if not question_id or method not in DISPLAY_NAMES:
            raise ValueError(f"invalid result identity: {pair}")
        if pair in seen:
            raise ValueError(f"duplicate successful result row: {pair}")
        seen.add(pair)
        output = result.get("output") or {}
        native_context = []
        flattened_ids: list[str] = []
        source_ranks: dict[str, list[int]] = {}
        for rank, context in enumerate(adapter_contexts(method, output), start=1):
            resolved, mapping = canonical_ids(
                method,
                context,
                corpus_by_id=corpus_by_id,
                graph_resolver=graph_resolver,
            )
            native_context.append(
                {
                    "rank": rank,
                    "evidence_id": context.get("name")
                    or context.get("evidence_id"),
                    "kind": context.get("kind"),
                    "group": (context.get("metadata") or {}).get("graph_group"),
                    "canonical_chunk_ids": resolved,
                    "canonical_mapping": mapping,
                }
            )
            for chunk_id in resolved:
                source_ranks.setdefault(chunk_id, []).append(rank)
                if chunk_id not in flattened_ids:
                    flattened_ids.append(chunk_id)

        ranking = []
        for rank, chunk_id in enumerate(flattened_ids[:10], start=1):
            corpus_row = corpus_by_id[chunk_id]
            ranking.append(
                {
                    "rank": rank,
                    "canonical_chunk_id": chunk_id,
                    "source_native_ranks": source_ranks[chunk_id],
                    "section_id": corpus_row.get("section_id"),
                    "page_pdf": corpus_row.get("page_pdf"),
                    "page_printed": corpus_row.get("page_printed"),
                    "figure_refs": corpus_row.get("figure_refs") or [],
                    "table_refs": corpus_row.get("table_refs") or [],
                }
            )
        normalized.append(
            {
                "schema_version": 1,
                "benchmark_id": "MUTCD-150-v1.0",
                "question_id": question_id,
                "question": result.get("question"),
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "maximum_depth": 10,
                "native_context_ranking": native_context,
                "canonical_chunk_ranking_top10": ranking,
                "all_canonical_chunks_in_native_context": flattened_ids,
            }
        )
    return normalized


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = (
        "method",
        "display_name",
        "questions",
        "evaluable_questions",
        "unevaluable_questions",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "hit_rate_at_10",
        "micro_recall_at_10",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["methods"]:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize completed retrieval rows and compute Recall/MRR/nDCG at 10."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--graphrag-input", type=Path, default=DEFAULT_GRAPH_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.results, args.gold, args.corpus, args.graphrag_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    result_rows = read_jsonl(args.results)
    corpus, corpus_report = canonicalize_chunks(read_jsonl(args.corpus))
    corpus_by_id = {str(row["chunk_id"]): row for row in corpus}
    resolver = GraphChunkResolver(args.graphrag_input, set(corpus_by_id))
    normalized = normalize_results(result_rows, corpus_by_id, resolver)
    qrels, qrels_report = build_chunk_qrels(
        read_jsonl(args.gold),
        corpus,
        require_nonempty=False,
    )
    per_question, summary, native_sensitivity = score_canonical_rankings(
        normalized,
        qrels,
        depth=10,
    )
    qrels_report["canonical_corpus"] = corpus_report
    write_jsonl(args.output / "normalized_rankings.jsonl", normalized)
    write_jsonl(args.output / "qrels.jsonl", qrels)
    write_jsonl(args.output / "per_question_metrics.jsonl", per_question)
    write_jsonl(
        args.output / "native_context_sensitivity.jsonl",
        native_sensitivity,
    )
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "qrels_report.json", qrels_report)
    write_summary_csv(args.output / "summary.csv", summary)
    print(
        json.dumps(
            {
                "ok": True,
                "normalized_rows": len(normalized),
                "output": str(args.output),
                "methods": summary["methods"],
                "unevaluable_question_ids": qrels_report[
                    "text_retrieval_unevaluable_question_ids"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
