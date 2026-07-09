#!/usr/bin/env python3
"""Export MRAG QA/evidence rows into upstream Self-RAG and CRAG eval input files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gem_rags.config import DEFAULT_MRAG_DIR, RetrieverConfig
from gem_rags.data import load_qa_items
from gem_rags.qa_sets import load_qa_ids_file
from gem_rags.retrieval import build_retriever
from gem_rags.types import Evidence, QAItem


DEFAULT_QA_PATH = DEFAULT_MRAG_DIR / "eval" / "gold_qa.jsonl"
DEFAULT_OUT_DIR = Path("data/working/upstream_eval_inputs")


def main() -> int:
    args = _parse_args()
    qa_ids = _qa_ids(args)
    retriever_config = RetrieverConfig(
        name=args.retriever_name,
        kind=args.retriever_kind,
        top_k=args.top_k,
        options=_parse_options(args.retriever_option),
    )
    report = export_upstream_inputs(
        qa_path=args.qa_path,
        mrag_dir=args.mrag_dir,
        out_dir=args.out_dir,
        retriever_config=retriever_config,
        limit=args.limit,
        qa_ids=qa_ids,
        formats=set(args.format or ["selfrag", "crag"]),
        crag_ndocs=args.crag_ndocs,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qa-ids", help="Comma-separated QA IDs to export.")
    parser.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to export.")
    parser.add_argument("--retriever-name", default="upstream_export_bm25_graph")
    parser.add_argument("--retriever-kind", default="bm25_graph")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retriever-option", action="append", default=[], help="Retriever option as key=value. Repeatable.")
    parser.add_argument("--format", action="append", choices=["selfrag", "crag"], help="Output format. Repeatable; defaults to both.")
    parser.add_argument("--crag-ndocs", type=int, default=10, help="Number of repeated passages per question for CRAG input.")
    return parser.parse_args()


def export_upstream_inputs(
    *,
    qa_path: Path,
    mrag_dir: Path,
    out_dir: Path,
    retriever_config: RetrieverConfig,
    limit: int | None = None,
    qa_ids: list[str] | None = None,
    formats: set[str] | None = None,
    crag_ndocs: int = 10,
) -> dict[str, Any]:
    selected_formats = formats or {"selfrag", "crag"}
    items = load_qa_items(qa_path, limit=limit, qa_ids=qa_ids)
    retriever = build_retriever(retriever_config, mrag_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in items:
        retrieval = retriever.retrieve(item)
        rows.append(
            {
                "item": item,
                "evidence": retrieval.evidence,
                "retrieval_debug": retrieval.debug,
                "retrieval_error": retrieval.error,
            }
        )

    outputs: dict[str, str] = {}
    if "selfrag" in selected_formats:
        path = out_dir / "selfrag_input.jsonl"
        _write_selfrag(path, rows)
        outputs["selfrag_jsonl"] = str(path)
    if "crag" in selected_formats:
        paths = _write_crag(out_dir, rows, ndocs=crag_ndocs)
        outputs.update(paths)

    manifest = {
        "qa_path": str(qa_path),
        "mrag_dir": str(mrag_dir),
        "out_dir": str(out_dir),
        "retriever": {
            "name": retriever_config.name,
            "kind": retriever_config.kind,
            "top_k": retriever_config.top_k,
            "options": retriever_config.options,
        },
        "qa_count": len(rows),
        "formats": sorted(selected_formats),
        "crag_ndocs": crag_ndocs,
        "outputs": outputs,
        "rows_with_retrieval_errors": sum(1 for row in rows if row["retrieval_error"]),
        "evidence_counts": {
            "min": min((len(row["evidence"]) for row in rows), default=0),
            "max": max((len(row["evidence"]) for row in rows), default=0),
            "total": sum(len(row["evidence"]) for row in rows),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _write_selfrag(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item: QAItem = row["item"]
            ctxs = [_selfrag_context(ev) for ev in row["evidence"]]
            payload = {
                "id": item.qa_id,
                "qa_id": item.qa_id,
                "question": item.question,
                "answers": _answers(item),
                "ctxs": ctxs,
                "top_contexts": ctxs,
                "expected_refusal": item.expected_refusal,
                "question_type": item.question_type,
                "gold_answer": item.gold_answer,
                "references": item.references,
                "gold_figures": item.gold_figures,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_crag(out_dir: Path, rows: list[dict[str, Any]], *, ndocs: int) -> dict[str, str]:
    test_path = out_dir / "crag_test_mutcd.txt"
    sources_path = out_dir / "crag_sources"
    retrieved_path = out_dir / "crag_retrieved_psgs"
    answers_path = out_dir / "crag_answers.jsonl"
    with (
        test_path.open("w", encoding="utf-8") as test_handle,
        sources_path.open("w", encoding="utf-8") as sources_handle,
        retrieved_path.open("w", encoding="utf-8") as retrieved_handle,
        answers_path.open("w", encoding="utf-8") as answers_handle,
    ):
        for row in rows:
            item: QAItem = row["item"]
            passages = [_flat_passage(ev) for ev in row["evidence"][:ndocs]]
            while len(passages) < ndocs:
                passages.append("")
            question = _one_line(item.question)
            sources_handle.write(question + "\n")
            retrieved_handle.write(" [sep] ".join(passages) + "\n")
            answers_handle.write(json.dumps({"qa_id": item.qa_id, "question": item.question, "answers": _answers(item)}, ensure_ascii=False) + "\n")
            for passage in passages:
                test_handle.write(f"{question} [SEP] {passage}\n")
    return {
        "crag_test_txt": str(test_path),
        "crag_sources": str(sources_path),
        "crag_retrieved_psgs": str(retrieved_path),
        "crag_answers_jsonl": str(answers_path),
    }


def _selfrag_context(ev: Evidence) -> dict[str, Any]:
    return {
        "id": ev.evidence_id,
        "title": _title(ev),
        "text": ev.text,
        "score": ev.score,
        "metadata": ev.metadata,
    }


def _title(ev: Evidence) -> str:
    meta = ev.metadata
    title = meta.get("section_title") or meta.get("title") or meta.get("caption") or ev.evidence_id
    section = meta.get("section_id")
    if section and str(section) not in str(title):
        return f"Section {section} - {title}"
    return str(title)


def _flat_passage(ev: Evidence) -> str:
    return _one_line(f"{_title(ev)} // {ev.text}")


def _one_line(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").split())


def _answers(item: QAItem) -> list[str]:
    raw_answers = item.raw.get("answers")
    if isinstance(raw_answers, list):
        return [str(answer) for answer in raw_answers if str(answer).strip()]
    if isinstance(raw_answers, str) and raw_answers.strip():
        return [raw_answers.strip()]
    gold = item.gold_answer
    if not isinstance(gold, dict):
        return []
    candidates = [gold.get("direct_answer"), gold.get("answer"), gold.get("summary")]
    return [str(answer).strip() for answer in candidates if answer is not None and str(answer).strip()]


def _parse_options(raw_options: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for raw in raw_options:
        if "=" not in raw:
            raise SystemExit(f"retriever option must use key=value: {raw!r}")
        key, value = [part.strip() for part in raw.split("=", 1)]
        if not key:
            raise SystemExit(f"retriever option has empty key: {raw!r}")
        options[key] = _coerce_value(value)
    return options


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _qa_ids(args: argparse.Namespace) -> list[str] | None:
    inline = [item.strip() for item in str(args.qa_ids or "").split(",") if item.strip()]
    if inline and args.qa_ids_file:
        raise SystemExit("--qa-ids and --qa-ids-file are mutually exclusive")
    if args.qa_ids_file:
        return load_qa_ids_file(args.qa_ids_file)
    return inline or None


if __name__ == "__main__":
    raise SystemExit(main())
