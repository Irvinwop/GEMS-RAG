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
DEFAULT_SELFRAG_REPO = ROOT / "external" / "rag-implementations" / "self-rag"
DEFAULT_CRAG_REPO = ROOT / "external" / "rag-implementations" / "crag"
DEFAULT_SELFRAG_MODEL = "selfrag/selfrag_llama2_7b"


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
        selfrag_repo=args.selfrag_repo,
        crag_repo=args.crag_repo,
        selfrag_model=args.selfrag_model,
        selfrag_output=args.selfrag_output,
        selfrag_task=args.selfrag_task,
        selfrag_metric=args.selfrag_metric,
        selfrag_mode=args.selfrag_mode,
        selfrag_max_new_tokens=args.selfrag_max_new_tokens,
        selfrag_threshold=args.selfrag_threshold,
        selfrag_use_groundness=args.selfrag_use_groundness,
        selfrag_use_utility=args.selfrag_use_utility,
        selfrag_use_seqscore=args.selfrag_use_seqscore,
        crag_output=args.crag_output,
        crag_task=args.crag_task,
        crag_device=args.crag_device,
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
    parser.add_argument("--selfrag-repo", type=Path, default=DEFAULT_SELFRAG_REPO)
    parser.add_argument("--crag-repo", type=Path, default=DEFAULT_CRAG_REPO)
    parser.add_argument("--selfrag-model", default=DEFAULT_SELFRAG_MODEL)
    parser.add_argument("--selfrag-output", type=Path, help="Predictions path for the emitted Self-RAG run command.")
    parser.add_argument("--selfrag-task", help="Optional upstream Self-RAG task name. Omit for generic QA prompts.")
    parser.add_argument("--selfrag-metric", default="match")
    parser.add_argument("--selfrag-mode", default="adaptive_retrieval", choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"])
    parser.add_argument("--selfrag-max-new-tokens", type=int, default=100)
    parser.add_argument("--selfrag-threshold", type=float)
    parser.add_argument("--selfrag-use-groundness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selfrag-use-utility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selfrag-use-seqscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crag-output", type=Path, help="Predictions path for the emitted CRAG inference/eval command templates.")
    parser.add_argument("--crag-task", default="mutcd", help="Task label passed to CRAG_Inference.py; unknown labels use the PopQA-style generic prompt path.")
    parser.add_argument("--crag-device", default="cuda:0")
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
    selfrag_repo: Path = DEFAULT_SELFRAG_REPO,
    crag_repo: Path = DEFAULT_CRAG_REPO,
    selfrag_model: str = DEFAULT_SELFRAG_MODEL,
    selfrag_output: Path | None = None,
    selfrag_task: str | None = None,
    selfrag_metric: str = "match",
    selfrag_mode: str = "adaptive_retrieval",
    selfrag_max_new_tokens: int = 100,
    selfrag_threshold: float | None = None,
    selfrag_use_groundness: bool = True,
    selfrag_use_utility: bool = True,
    selfrag_use_seqscore: bool = True,
    crag_output: Path | None = None,
    crag_task: str = "mutcd",
    crag_device: str = "cuda:0",
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

    upstream_repos = _upstream_repo_status(selected_formats, selfrag_repo=selfrag_repo, crag_repo=crag_repo)
    upstream_commands = _upstream_commands(
        selected_formats,
        outputs=outputs,
        out_dir=out_dir,
        retriever_top_k=retriever_config.top_k,
        selfrag_repo=selfrag_repo,
        selfrag_model=selfrag_model,
        selfrag_output=selfrag_output,
        selfrag_task=selfrag_task,
        selfrag_metric=selfrag_metric,
        selfrag_mode=selfrag_mode,
        selfrag_max_new_tokens=selfrag_max_new_tokens,
        selfrag_threshold=selfrag_threshold,
        selfrag_use_groundness=selfrag_use_groundness,
        selfrag_use_utility=selfrag_use_utility,
        selfrag_use_seqscore=selfrag_use_seqscore,
        crag_repo=crag_repo,
        crag_output=crag_output,
        crag_task=crag_task,
        crag_ndocs=crag_ndocs,
        crag_device=crag_device,
    )
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
        "upstream_repos": upstream_repos,
        "upstream_commands": upstream_commands,
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


def _upstream_repo_status(selected_formats: set[str], *, selfrag_repo: Path, crag_repo: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    if "selfrag" in selected_formats:
        entrypoint = _selfrag_entrypoint(selfrag_repo)
        status["selfrag"] = {
            "repo": str(selfrag_repo),
            "repo_found": selfrag_repo.exists(),
            "entrypoint": str(entrypoint),
            "entrypoint_found": entrypoint.exists(),
        }
    if "crag" in selected_formats:
        inference = _crag_inference_entrypoint(crag_repo)
        eval_script = _crag_eval_entrypoint(crag_repo)
        status["crag"] = {
            "repo": str(crag_repo),
            "repo_found": crag_repo.exists(),
            "inference_entrypoint": str(inference),
            "inference_entrypoint_found": inference.exists(),
            "eval_entrypoint": str(eval_script),
            "eval_entrypoint_found": eval_script.exists(),
        }
    return status


def _upstream_commands(
    selected_formats: set[str],
    *,
    outputs: dict[str, str],
    out_dir: Path,
    retriever_top_k: int,
    selfrag_repo: Path,
    selfrag_model: str,
    selfrag_output: Path | None,
    selfrag_task: str | None,
    selfrag_metric: str,
    selfrag_mode: str,
    selfrag_max_new_tokens: int,
    selfrag_threshold: float | None,
    selfrag_use_groundness: bool,
    selfrag_use_utility: bool,
    selfrag_use_seqscore: bool,
    crag_repo: Path,
    crag_output: Path | None,
    crag_task: str,
    crag_ndocs: int,
    crag_device: str,
) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    if "selfrag" in selected_formats:
        output_file = selfrag_output or (out_dir / "selfrag_output.json")
        command = [
            "python",
            str(_selfrag_entrypoint(selfrag_repo)),
            "--model_name",
            selfrag_model,
            "--input_file",
            outputs["selfrag_jsonl"],
            "--output_file",
            str(output_file),
            "--metric",
            selfrag_metric,
            "--ndocs",
            str(retriever_top_k),
            "--mode",
            selfrag_mode,
            "--max_new_tokens",
            str(selfrag_max_new_tokens),
        ]
        if selfrag_task:
            command.extend(["--task", selfrag_task])
        if selfrag_threshold is not None:
            command.extend(["--threshold", str(selfrag_threshold)])
        if selfrag_use_groundness:
            command.append("--use_groundness")
        if selfrag_use_utility:
            command.append("--use_utility")
        if selfrag_use_seqscore:
            command.append("--use_seqscore")
        commands["selfrag_run_short_form"] = command
    if "crag" in selected_formats:
        output_file = crag_output or (out_dir / "crag_output.txt")
        ref_dir = out_dir / "crag_ref"
        commands["crag_inference_template"] = [
            "python",
            str(_crag_inference_entrypoint(crag_repo)),
            "--generator_path",
            "${CRAG_GENERATOR_PATH}",
            "--evaluator_path",
            "${CRAG_EVALUATOR_PATH}",
            "--input_file",
            outputs["crag_test_txt"],
            "--output_file",
            str(output_file),
            "--internal_knowledge_path",
            str(ref_dir / "correct"),
            "--external_knowledge_path",
            str(ref_dir / "incorrect"),
            "--combined_knowledge_path",
            str(ref_dir / "ambiguous"),
            "--task",
            crag_task,
            "--method",
            "crag",
            "--device",
            crag_device,
            "--ndocs",
            str(crag_ndocs),
        ]
        commands["crag_eval_match"] = [
            "python",
            str(_crag_eval_entrypoint(crag_repo)),
            "--input_file",
            outputs["crag_answers_jsonl"],
            "--eval_file",
            str(output_file),
            "--metric",
            "match",
        ]
        commands["crag_notes"] = (
            "CRAG inference also needs prepared correct/incorrect/ambiguous knowledge files. "
            "Use the exported crag_sources and crag_retrieved_psgs sidecars as the MUTCD input to the upstream knowledge-preparation scripts."
        )
    return commands


def _selfrag_entrypoint(repo: Path) -> Path:
    return repo / "retrieval_lm" / "run_short_form.py"


def _crag_inference_entrypoint(repo: Path) -> Path:
    return repo / "scripts" / "CRAG_Inference.py"


def _crag_eval_entrypoint(repo: Path) -> Path:
    return repo / "scripts" / "eval.py"


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
