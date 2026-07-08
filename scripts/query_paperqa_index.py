#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "paper-qa"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_INDEX = ROOT / "data" / "working" / "paperqa_index" / "docs.pkl"


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130


async def _main(args: argparse.Namespace) -> int:
    _add_repo(args.repo)
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "query" or (args.command == "index" and not args.defer_embedding):
        _ensure_api_key(args)

    try:
        from paperqa import Docs, Settings
        from paperqa.types import Doc, Text
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": "import_failed",
                    "detail": repr(exc),
                    "notes": "Install the cloned PaperQA2 package and its optional parser/vector dependencies before indexing.",
                }
            ),
            file=sys.stderr,
        )
        return 2

    if args.command == "index":
        args.index.parent.mkdir(parents=True, exist_ok=True)
        doc = Doc(docname="MUTCD 11th Edition Revision 1", dockey="mutcd11e", citation="MUTCD 11th Edition Revision 1")
        texts = []
        for row in _read_jsonl(args.chunks):
            metadata = row.get("metadata", {})
            texts.append(
                Text(
                    text=row["text"],
                    name=row["doc_id"],
                    doc=doc,
                    section_id=metadata.get("section_id"),
                    content_type=metadata.get("content_type"),
                    ordinal=metadata.get("ordinal"),
                    page_printed=metadata.get("page_printed"),
                    title=row.get("title"),
                )
            )
        docs = Docs()
        settings = Settings(parsing={"defer_embedding": args.defer_embedding})
        await docs.aadd_texts(texts=texts, doc=doc, settings=settings)
        with args.index.open("wb") as handle:
            pickle.dump(docs, handle)
        print(json.dumps({"indexed": True, "texts": len(texts), "index": str(args.index)}))
        return 0

    if args.command == "query":
        with args.index.open("rb") as handle:
            docs = pickle.load(handle)
        settings = Settings(embedding=args.embedding, llm=args.llm, summary_llm=args.summary_llm)
        session = await docs.aquery(args.question, settings=settings)
        contexts = [
            {
                "text": getattr(ctx, "text", ""),
                "score": getattr(ctx, "score", None),
                "name": getattr(ctx, "name", None),
            }
            for ctx in getattr(session, "contexts", [])
        ]
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": getattr(session, "answer", None) or getattr(session, "raw_answer", None),
                    "contexts": contexts,
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index or query PaperQA2 over exported MRAG chunks.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="Optional OpenAI-compatible base URL, exported as OPENAI_BASE_URL for PaperQA providers.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Report whether the local environment can run the PaperQA2 adapter.")
    check.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)

    index = sub.add_parser("index", help="Create an ignored PaperQA Docs pickle from exported chunks.")
    index.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index.add_argument("--defer-embedding", action="store_true", help="Do not embed at index time; embeddings are computed during query.")

    query = sub.add_parser("query", help="Query an existing PaperQA Docs pickle.")
    query.add_argument("--question", required=True)
    query.add_argument("--embedding", default="text-embedding-3-small")
    query.add_argument("--llm", default="gpt-4o-mini")
    query.add_argument("--summary-llm", default="gpt-4o-mini")
    return parser.parse_args()


def _add_repo(repo: Path) -> None:
    if not repo.exists():
        raise SystemExit(f"PaperQA repo not found: {repo}")
    sys.path.insert(0, str(repo / "src"))


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    modules = ["paperqa", "paperqa.types"]
    import_errors = _import_errors(modules)
    api_key_present = bool(os.getenv(args.api_key_env))
    api_key_usable = api_key_present or bool(args.allow_missing_api_key)
    chunks = getattr(args, "chunks", DEFAULT_CHUNKS)
    return {
        "runnable": args.repo.exists() and not import_errors and api_key_usable,
        "repo": str(args.repo),
        "repo_found": args.repo.exists(),
        "package_source_found": (args.repo / "src" / "paperqa").exists(),
        "index": str(args.index),
        "index_found": args.index.exists(),
        "chunks": str(chunks),
        "chunks_found": chunks.exists(),
        "api_key_env": args.api_key_env,
        "api_key_present": api_key_present,
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "api_key_usable": api_key_usable,
        "base_url": args.base_url,
        "missing_or_failed_imports": import_errors,
        "notes": "The default PaperQA2 query settings use OpenAI-compatible embedding and LLM model names; configure the API key or override settings before querying.",
    }


def _ensure_api_key(args: argparse.Namespace) -> None:
    base_url = getattr(args, "base_url", None)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if os.getenv(args.api_key_env):
        return
    if args.allow_missing_api_key:
        os.environ[args.api_key_env] = "local"
        return
    raise SystemExit(f"missing API key env var: {args.api_key_env}")


def _import_errors(module_names: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors[name] = repr(exc)
    return errors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
