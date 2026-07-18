#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.endpoint import probe_openai_endpoint
from gems_rag.index_completion import (
    completion_marker_matches,
    file_identity,
    publish_completion_marker,
    read_completion_marker,
    value_fingerprint,
)
from gems_rag.lightrag_compat import (
    cap_completion_tokens,
    lightrag_document_status_report,
)

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "lightrag"
DEFAULT_CORPUS = ROOT / "data" / "working" / "mrag_corpus" / "lightrag_corpus.txt"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "lightrag_index"
INDEX_SENTINEL = ".gems_rag_lightrag_index.json"


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
    if args.command == "query":
        report = _dependency_report(args)
        if not report["runnable"]:
            print(
                json.dumps({"error": "lightrag_not_ready", **report}, indent=2),
                file=sys.stderr,
            )
            return 2

    api_key = _api_key(args)
    try:
        rag = _make_rag(args, api_key)
    except Exception as exc:
        print(f"failed to initialize LightRAG adapter: {exc!r}", file=sys.stderr)
        return 2

    if args.command == "index":
        (args.working_dir / INDEX_SENTINEL).unlink(missing_ok=True)

    result: Any = None
    source_count = 0
    index_status: dict[str, Any] | None = None
    try:
        await rag.initialize_storages()
        if args.command == "index":
            corpus = args.corpus.read_text(encoding="utf-8")
            doc_id = _corpus_doc_id(corpus)
            source_count = 1
            index_status = await lightrag_document_status_report(
                rag,
                doc_ids=[doc_id],
            )
            if not index_status["complete"]:
                await rag.ainsert(corpus)
                index_status = await lightrag_document_status_report(
                    rag,
                    doc_ids=[doc_id],
                )
        elif args.command == "query":
            from lightrag import QueryParam

            result = await rag.aquery(
                args.question,
                param=QueryParam(
                    mode=args.mode,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                    only_need_context=args.only_need_context,
                    response_type=args.response_type,
                ),
            )
        else:
            raise AssertionError(args.command)
    finally:
        finalize = getattr(rag, "finalize_storages", None)
        if finalize:
            await finalize()

    if args.command == "index":
        if not index_status or not index_status["complete"]:
            print(
                json.dumps(
                    {
                        "error": "lightrag_index_incomplete",
                        "document_status": index_status,
                    }
                ),
                file=sys.stderr,
            )
            return 2
        index_files = _index_files(args.working_dir)
        if not index_files:
            print(json.dumps({"error": "lightrag_index_produced_no_artifacts"}), file=sys.stderr)
            return 2
        publish_completion_marker(
            args.working_dir / INDEX_SENTINEL,
            _index_identity(args),
            sources=source_count,
            index_files=index_files,
        )
        print(json.dumps({"indexed": True, "corpus": str(args.corpus), "working_dir": str(args.working_dir)}))
        return 0
    if args.json:
        print(json.dumps({"mode": args.mode, "question": args.question, "result": result}, ensure_ascii=False))
    else:
        print(result)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index or query the cloned LightRAG implementation over exported MRAG text.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Report whether the local environment can run the LightRAG adapter.")
    _add_common_args(check)
    check.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    index = sub.add_parser("index", help="Build or extend the ignored LightRAG index.")
    _add_common_args(index)
    index.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    index.add_argument("--force", action="store_true", help="Delete the existing ignored index directory before indexing.")

    query = sub.add_parser("query", help="Query an existing ignored LightRAG index.")
    _add_common_args(query)
    query.add_argument("--question", required=True)
    query.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix", "bypass"])
    query.add_argument("--top-k", type=int, default=12)
    query.add_argument("--chunk-top-k", type=int, default=12)
    query.add_argument("--only-need-context", action="store_true", help="Return retrieved context instead of a generated answer.")
    query.add_argument("--response-type", default="Multiple Paragraphs")
    query.add_argument("--json", action="store_true", help="Print a JSON wrapper instead of raw result text.")

    args = parser.parse_args()
    if args.llm_max_tokens is not None and args.llm_max_tokens <= 0:
        parser.error("--llm-max-tokens must be positive")
    if args.command == "index" and args.force and args.working_dir.exists():
        shutil.rmtree(args.working_dir)
    if args.command in {"index", "query"}:
        args.working_dir.mkdir(parents=True, exist_ok=True)
    return args


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="Path to cloned LightRAG repository.")
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR, help="Ignored LightRAG index directory.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--llm-model", default=os.getenv("LIGHTRAG_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("LIGHTRAG_EMBEDDING_DIM", "3072")))
    parser.add_argument("--embedding-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Hard ceiling for each internal LightRAG completion.",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument(
        "--entity-extraction-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LightRAG's structured JSON entity extraction mode.",
    )


def _add_repo(repo: Path) -> None:
    if not repo.exists():
        raise SystemExit(f"LightRAG repo not found: {repo}")
    sys.path.insert(0, str(repo))


def _corpus_doc_id(corpus: str) -> str:
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    return compute_mdhash_id(sanitize_text_for_encoding(corpus), prefix="doc-")


def _api_key(args: argparse.Namespace) -> str:
    api_key = os.getenv(args.api_key_env)
    if not api_key and args.allow_missing_api_key:
        return "local"
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")
    return api_key


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    modules = [
        "lightrag",
        "lightrag.llm.openai",
        "lightrag.utils",
    ]
    import_errors = _import_errors(modules)
    api_key = os.getenv(args.api_key_env)
    api_key_present = bool(api_key)
    credential_available = api_key_present or bool(args.allow_missing_api_key)
    endpoint = probe_openai_endpoint(
        getattr(args, "base_url", None),
        api_key=api_key or ("local" if args.allow_missing_api_key else None),
    )
    endpoint_usable = endpoint["usable"] if endpoint["checked"] else True
    api_key_usable = credential_available and endpoint_usable
    corpus = getattr(args, "corpus", DEFAULT_CORPUS)
    index_files = _index_files(args.working_dir)
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel = read_completion_marker(sentinel_path)
    sentinel_matches_input = completion_marker_matches(sentinel_path, _index_identity(args, corpus=corpus))
    sentinel_files_present = _sentinel_files_present(sentinel, index_files)
    environment_ready = args.repo.exists() and not import_errors
    index_ready = bool(index_files) and sentinel_matches_input and sentinel_files_present
    return {
        "runnable": environment_ready and api_key_usable and index_ready,
        "environment_ready": environment_ready,
        "repo": str(args.repo),
        "repo_found": args.repo.exists(),
        "working_dir": str(args.working_dir),
        "working_dir_exists": args.working_dir.exists(),
        "index_ready": index_ready,
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "sentinel": str(sentinel_path),
        "sentinel_found": sentinel_path.is_file(),
        "sentinel_matches_input": sentinel_matches_input,
        "sentinel_files_present": sentinel_files_present,
        "corpus": str(corpus),
        "corpus_found": corpus.exists(),
        "api_key_env": args.api_key_env,
        "api_key_present": api_key_present,
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "credential_available": credential_available,
        "api_key_usable": api_key_usable,
        "base_url": getattr(args, "base_url", None),
        "endpoint": endpoint,
        "endpoint_reachable": endpoint["reachable"],
        "endpoint_usable": endpoint["usable"],
        "model_service_ready": api_key_usable,
        "missing_or_failed_imports": import_errors,
        "notes": "The default LightRAG adapter uses OpenAI-compatible completion and embedding calls; set the API key/base URL before indexing or querying.",
    }


def _index_files(working_dir: Path) -> list[str]:
    if not working_dir.exists():
        return []
    return sorted(
        str(path.relative_to(working_dir))
        for path in working_dir.rglob("*")
        if path.is_file() and path.name != INDEX_SENTINEL
    )


def _index_identity(args: argparse.Namespace, *, corpus: Path | None = None) -> dict[str, Any]:
    source = corpus or getattr(args, "corpus", DEFAULT_CORPUS)
    return {
        "corpus": file_identity(source),
        "llm_model": getattr(args, "llm_model", os.getenv("LIGHTRAG_LLM_MODEL", "gpt-4o-mini")),
        "embedding_model": getattr(
            args,
            "embedding_model",
            os.getenv("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-large"),
        ),
        "embedding_dim": int(getattr(args, "embedding_dim", 3072)),
        "embedding_max_tokens": int(getattr(args, "embedding_max_tokens", 8192)),
        "llm_max_tokens": getattr(args, "llm_max_tokens", None),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "entity_extraction_json": bool(getattr(args, "entity_extraction_json", False)),
        "endpoint": value_fingerprint(getattr(args, "base_url", None)),
    }


def _sentinel_files_present(sentinel: dict[str, Any] | None, index_files: list[str]) -> bool:
    recorded = sentinel.get("index_files") if sentinel else None
    return bool(recorded and set(recorded).issubset(index_files))


def _import_errors(module_names: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in module_names:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as exc:
            errors[name] = repr(exc)
    return errors


def _make_rag(args: argparse.Namespace, api_key: str) -> Any:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    async def llm_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str, str]] | None = None, **kwargs: Any) -> Any:
        cap_completion_tokens(kwargs, getattr(args, "llm_max_tokens", None))
        reasoning_effort = getattr(args, "reasoning_effort", None)
        if reasoning_effort:
            kwargs.setdefault("reasoning_effort", reasoning_effort)
        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=args.base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.embedding_max_tokens,
        func=partial(openai_embed.func, model=args.embedding_model, api_key=api_key, base_url=args.base_url),
    )
    return LightRAG(
        working_dir=str(args.working_dir),
        llm_model_func=llm_model_func,
        llm_model_name=args.llm_model,
        embedding_func=embedding_func,
        entity_extraction_use_json=bool(getattr(args, "entity_extraction_json", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
