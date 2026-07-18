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
    SCHEMA_VERSION,
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

DEFAULT_RAGANYTHING_REPO = ROOT / "external" / "rag-implementations" / "rag-anything"
DEFAULT_LIGHTRAG_REPO = ROOT / "external" / "rag-implementations" / "lightrag"
DEFAULT_CONTENT_LIST = ROOT / "data" / "working" / "mrag_corpus" / "raganything_content_list.json"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "raganything_index"
DEFAULT_NATIVE_WORKING_DIR = ROOT / "data" / "working" / "raganything_native_pdf_index"
DEFAULT_PDF = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG" / "mutcd11theditionr1hl.pdf"
INDEX_SENTINEL = ".gems_rag_raganything_index.json"
INDEX_ATTEMPT = ".gems_rag_raganything_attempt.json"
DEFAULT_BATCH_PAGES = 25


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130


async def _main(args: argparse.Namespace) -> int:
    _add_repo(args.lightrag_repo, "LightRAG")
    _add_repo(args.repo, "RAG-Anything")
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "query":
        report = _dependency_report(args)
        if not report["runnable"]:
            print(
                json.dumps({"error": "raganything_not_ready", **report}, indent=2),
                file=sys.stderr,
            )
            return 2

    if args.command == "index":
        if getattr(args, "force", False) and args.working_dir.exists():
            shutil.rmtree(args.working_dir)
        args.working_dir.mkdir(parents=True, exist_ok=True)

    api_key = _api_key(args)
    try:
        rag = _make_rag(args, api_key)
    except Exception as exc:
        print(f"failed to initialize RAG-Anything adapter: {exc!r}", file=sys.stderr)
        return 2

    if args.command == "index":
        sentinel_path = args.working_dir / INDEX_SENTINEL
        attempt_path = args.working_dir / INDEX_ATTEMPT
        identity = _index_identity(args)
        index_files = _index_files(args.working_dir)
        sentinel = read_completion_marker(sentinel_path)
        if completion_marker_matches(sentinel_path, identity) and _sentinel_files_present(
            sentinel, index_files
        ):
            print(
                json.dumps(
                    {
                        "indexed": False,
                        "already_ready": True,
                        "ingestion_mode": args.ingestion_mode,
                        "working_dir": str(args.working_dir),
                    }
                )
            )
            return 0

        attempt = read_completion_marker(attempt_path)
        if index_files and not _attempt_marker_matches(attempt, identity):
            print(
                json.dumps(
                    {
                        "error": "raganything_index_input_changed_use_force",
                        "working_dir": str(args.working_dir),
                        "attempt_found": attempt is not None,
                        "attempt_matches_input": False,
                    }
                ),
                file=sys.stderr,
            )
            return 2

        sentinel_path.unlink(missing_ok=True)
        publish_completion_marker(
            attempt_path,
            identity,
            complete=False,
            ingestion_mode=args.ingestion_mode,
        )
        await _ensure_query_ready(rag)
        _install_lightrag_insert_guard(rag)
        skipped_documents = 0
        if args.ingestion_mode == "native_pdf":
            document_ids = [args.doc_id]
            if await _document_fully_processed(rag, args.doc_id):
                skipped_documents = 1
            else:
                await rag.process_document_complete(
                    file_path=str(args.pdf),
                    doc_id=args.doc_id,
                    display_stats=args.display_stats,
                )
            source_count = 1
        else:
            content_list = json.loads(args.content_list.read_text(encoding="utf-8"))
            limit = getattr(args, "limit", None)
            if limit is not None:
                content_list = content_list[:limit]
            batches = _shared_content_batches(
                content_list,
                pages_per_batch=getattr(args, "batch_pages", DEFAULT_BATCH_PAGES),
                base_doc_id=args.doc_id,
            )
            document_ids = [batch["doc_id"] for batch in batches]
            for batch_index, batch in enumerate(batches):
                doc_id = batch["doc_id"]
                if await _document_fully_processed(rag, doc_id):
                    skipped_documents += 1
                    continue
                await rag.insert_content_list(
                    content_list=batch["content_list"],
                    file_path=_batch_file_path(
                        args.file_path,
                        batch,
                        batch_index=batch_index,
                    ),
                    doc_id=doc_id,
                    display_stats=args.display_stats,
                )
                batch_status = await _raganything_document_status_report(
                    rag, [doc_id]
                )
                if not batch_status["complete"]:
                    print(
                        json.dumps(
                            {
                                "error": "raganything_batch_incomplete",
                                "doc_id": doc_id,
                                "page_start": batch["page_start"],
                                "page_end": batch["page_end"],
                                "document_status": batch_status,
                            }
                        ),
                        file=sys.stderr,
                    )
                    return 2
            source_count = len(content_list)
        index_status = await _raganything_document_status_report(
            rag,
            document_ids,
        )
        if not index_status["complete"]:
            print(
                json.dumps(
                    {
                        "error": "raganything_index_incomplete",
                        "document_status": index_status,
                    }
                ),
                file=sys.stderr,
            )
            return 2
        index_files = _index_files(args.working_dir)
        if not index_files:
            print(json.dumps({"error": "raganything_index_produced_no_artifacts"}), file=sys.stderr)
            return 2
        publish_completion_marker(
            sentinel_path,
            identity,
            sources=source_count,
            documents=len(document_ids),
            document_ids=document_ids,
            index_files=index_files,
        )
        attempt_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "indexed": True,
                    "ingestion_mode": args.ingestion_mode,
                    "sources": source_count,
                    "documents": len(document_ids),
                    "skipped_documents": skipped_documents,
                    "working_dir": str(args.working_dir),
                }
            )
        )
        return 0
    if args.command == "query":
        await _ensure_query_ready(rag)
        result = await rag.aquery(args.question, mode=args.mode, **_query_kwargs(args))
        if args.json:
            print(json.dumps(_query_payload(args, result), ensure_ascii=False))
        else:
            print(result)
        return 0
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index or query the cloned RAG-Anything implementation over exported MRAG content.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Report whether the local environment can run the RAG-Anything adapter.")
    _add_common_args(check)
    check.add_argument("--content-list", type=Path, default=DEFAULT_CONTENT_LIST)
    check.add_argument("--limit", type=int)

    index = sub.add_parser("index", help="Build or extend the ignored RAG-Anything index.")
    _add_common_args(index)
    index.add_argument("--content-list", type=Path, default=DEFAULT_CONTENT_LIST)
    index.add_argument("--limit", type=int)
    index.add_argument("--file-path", default="mutcd11theditionr1hl.pdf")
    index.add_argument("--doc-id", default="mutcd11e")
    index.add_argument("--display-stats", action="store_true")
    index.add_argument("--force", action="store_true", help="Delete the existing ignored index directory before indexing.")

    query = sub.add_parser("query", help="Query an existing ignored RAG-Anything index.")
    _add_common_args(query)
    query.add_argument("--content-list", type=Path, default=DEFAULT_CONTENT_LIST)
    query.add_argument("--limit", type=int)
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
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_pages <= 0:
        parser.error("--batch-pages must be positive")
    if args.working_dir is None:
        args.working_dir = DEFAULT_NATIVE_WORKING_DIR if args.ingestion_mode == "native_pdf" else DEFAULT_WORKING_DIR
    if args.command == "query":
        args.working_dir.mkdir(parents=True, exist_ok=True)
    return args


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_RAGANYTHING_REPO, help="Path to cloned RAG-Anything repository.")
    parser.add_argument("--lightrag-repo", type=Path, default=DEFAULT_LIGHTRAG_REPO, help="Path to cloned LightRAG repository.")
    parser.add_argument("--working-dir", type=Path, help="Ignored RAG-Anything index directory; defaults are isolated by ingestion mode.")
    parser.add_argument("--ingestion-mode", choices=["shared_corpus", "native_pdf"], default="shared_corpus")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--llm-model", default=os.getenv("RAGANYTHING_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--vision-model", default=os.getenv("RAGANYTHING_VISION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGANYTHING_EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("RAGANYTHING_EMBEDDING_DIM", "3072")))
    parser.add_argument("--embedding-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=DEFAULT_BATCH_PAGES,
        help="Group shared-corpus items into stable resumable page batches.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Hard ceiling for each internal LightRAG text completion.",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument(
        "--entity-extraction-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LightRAG's structured JSON entity extraction mode.",
    )


def _add_repo(repo: Path, label: str) -> None:
    if not repo.exists():
        raise SystemExit(f"{label} repo not found: {repo}")
    sys.path.insert(0, str(repo))


def _api_key(args: argparse.Namespace) -> str:
    api_key = os.getenv(args.api_key_env)
    if not api_key and args.allow_missing_api_key:
        return "local"
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")
    return api_key


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    modules = [
        "lightrag.llm.openai",
        "lightrag.utils",
        "raganything",
        "raganything.config",
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
    index_files = _index_files(args.working_dir)
    ingestion_mode = getattr(args, "ingestion_mode", "shared_corpus")
    pdf = getattr(args, "pdf", DEFAULT_PDF)
    content_list = getattr(args, "content_list", DEFAULT_CONTENT_LIST)
    source = pdf if ingestion_mode == "native_pdf" else content_list
    sentinel_path = args.working_dir / INDEX_SENTINEL
    attempt_path = args.working_dir / INDEX_ATTEMPT
    sentinel = read_completion_marker(sentinel_path)
    attempt = read_completion_marker(attempt_path)
    identity = _index_identity(args, source=source, ingestion_mode=ingestion_mode)
    sentinel_matches_input = completion_marker_matches(
        sentinel_path,
        identity,
    )
    sentinel_files_present = _sentinel_files_present(sentinel, index_files)
    environment_ready = args.repo.exists() and args.lightrag_repo.exists() and not import_errors
    index_ready = bool(index_files) and sentinel_matches_input and sentinel_files_present
    return {
        "runnable": environment_ready and api_key_usable and index_ready,
        "environment_ready": environment_ready,
        "repo": str(args.repo),
        "repo_found": args.repo.exists(),
        "lightrag_repo": str(args.lightrag_repo),
        "lightrag_repo_found": args.lightrag_repo.exists(),
        "working_dir": str(args.working_dir),
        "working_dir_exists": args.working_dir.exists(),
        "index_ready": index_ready,
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "sentinel": str(sentinel_path),
        "sentinel_found": sentinel_path.is_file(),
        "sentinel_matches_input": sentinel_matches_input,
        "sentinel_files_present": sentinel_files_present,
        "attempt": str(attempt_path),
        "attempt_found": attempt_path.is_file(),
        "attempt_matches_input": _attempt_marker_matches(attempt, identity),
        "index_resumable": bool(index_files)
        and _attempt_marker_matches(attempt, identity),
        "content_list": str(content_list),
        "content_list_found": content_list.exists(),
        "pdf": str(pdf),
        "pdf_found": pdf.exists(),
        "ingestion_mode": ingestion_mode,
        "limit": getattr(args, "limit", None),
        "source_found": source.exists(),
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
        "notes": "The shared-corpus adapter resumes stable page batches. Native PDF parsing additionally needs RAG-Anything's parser dependencies.",
    }


def _index_files(working_dir: Path) -> list[str]:
    if not working_dir.exists():
        return []
    return sorted(
        str(path.relative_to(working_dir))
        for path in working_dir.rglob("*")
        if path.is_file() and path.name not in {INDEX_SENTINEL, INDEX_ATTEMPT}
    )


def _index_identity(
    args: argparse.Namespace,
    *,
    source: Path | None = None,
    ingestion_mode: str | None = None,
) -> dict[str, Any]:
    mode = ingestion_mode or getattr(args, "ingestion_mode", "shared_corpus")
    source_path = source or (
        getattr(args, "pdf", DEFAULT_PDF)
        if mode == "native_pdf"
        else getattr(args, "content_list", DEFAULT_CONTENT_LIST)
    )
    return {
        "source": file_identity(source_path),
        "ingestion_mode": mode,
        "limit": getattr(args, "limit", None),
        "document_partitioning": "page_batches_v1" if mode == "shared_corpus" else "single_document",
        "batch_pages": (
            int(getattr(args, "batch_pages", DEFAULT_BATCH_PAGES))
            if mode == "shared_corpus"
            else None
        ),
        "doc_id": getattr(args, "doc_id", "mutcd11e"),
        "file_path": (
            getattr(args, "file_path", "mutcd11theditionr1hl.pdf")
            if mode == "shared_corpus"
            else None
        ),
        "parser": os.getenv("RAGANYTHING_PARSER", "mineru") if mode == "native_pdf" else None,
        "llm_model": getattr(args, "llm_model", os.getenv("RAGANYTHING_LLM_MODEL", "gpt-4o-mini")),
        "vision_model": getattr(
            args,
            "vision_model",
            os.getenv("RAGANYTHING_VISION_MODEL", "gpt-4o-mini"),
        ),
        "embedding_model": getattr(
            args,
            "embedding_model",
            os.getenv("RAGANYTHING_EMBEDDING_MODEL", "text-embedding-3-large"),
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


def _attempt_marker_matches(
    marker: dict[str, Any] | None,
    identity: dict[str, Any],
) -> bool:
    return bool(
        marker
        and marker.get("schema_version") == SCHEMA_VERSION
        and marker.get("complete") is False
        and marker.get("identity") == identity
    )


def _shared_content_batches(
    content_list: list[Any],
    *,
    pages_per_batch: int,
    base_doc_id: str,
) -> list[dict[str, Any]]:
    if pages_per_batch <= 0:
        raise ValueError("pages_per_batch must be positive")
    if not content_list:
        raise ValueError("RAG-Anything content list is empty")

    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(content_list):
        if not isinstance(item, dict):
            raise ValueError(f"RAG-Anything content item {index} is not an object")
        page_idx = item.get("page_idx")
        if isinstance(page_idx, bool) or not isinstance(page_idx, int):
            raise ValueError(
                f"RAG-Anything content item {index} has no integer page_idx"
            )
        items_by_page.setdefault(page_idx, []).append(item)

    pages = sorted(items_by_page)
    first_page = pages[0]
    pages_by_batch: dict[int, list[int]] = {}
    for page_idx in pages:
        page_start = first_page + (
            (page_idx - first_page) // pages_per_batch
        ) * pages_per_batch
        pages_by_batch.setdefault(page_start, []).append(page_idx)

    batches: list[dict[str, Any]] = []
    for page_start, batch_pages in sorted(pages_by_batch.items()):
        page_end = page_start + pages_per_batch - 1
        batch_items = [
            item
            for page_idx in batch_pages
            for item in items_by_page[page_idx]
        ]
        batches.append(
            {
                "doc_id": f"{base_doc_id}:pages:{page_start:04d}-{page_end:04d}",
                "page_start": page_start,
                "page_end": page_end,
                "content_list": batch_items,
            }
        )
    return batches


def _batch_file_path(
    file_path: str,
    batch: dict[str, Any],
    *,
    batch_index: int,
) -> str:
    if batch_index == 0:
        return file_path
    path = Path(file_path)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    batch_name = (
        f"{stem}_pages_{batch['page_start']:04d}-{batch['page_end']:04d}{suffix}"
    )
    return str(path.with_name(batch_name))


async def _document_fully_processed(rag: Any, doc_id: str) -> bool:
    check = getattr(rag, "is_document_fully_processed", None)
    if check is not None:
        return bool(await check(doc_id))
    report = await lightrag_document_status_report(
        getattr(rag, "lightrag", None),
        doc_ids=[doc_id],
    )
    return bool(report["complete"])


async def _raganything_document_status_report(
    rag: Any,
    doc_ids: list[str],
) -> dict[str, Any]:
    statuses = {
        doc_id: await _document_fully_processed(rag, doc_id)
        for doc_id in doc_ids
    }
    complete_count = sum(statuses.values())
    return {
        "complete": bool(statuses) and complete_count == len(statuses),
        "document_count": len(statuses),
        "complete_count": complete_count,
        "incomplete_document_ids": [
            doc_id for doc_id, complete in statuses.items() if not complete
        ],
    }


def _query_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "only_need_context": args.only_need_context,
        "response_type": args.response_type,
    }
    if args.only_need_context:
        kwargs["vlm_enhanced"] = False
    return kwargs


def _query_payload(args: argparse.Namespace, result: Any) -> dict[str, Any]:
    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    payload: dict[str, Any] = {
        "mode": args.mode,
        "question": args.question,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "only_need_context": args.only_need_context,
        "response_type": args.response_type,
        "result": result_text,
    }
    if args.only_need_context:
        payload["contexts"] = [
            {
                "name": f"raganything:{args.mode}:context",
                "kind": "tool_trace",
                "text": result_text,
                "metadata": {
                    "mode": args.mode,
                    "top_k": args.top_k,
                    "chunk_top_k": args.chunk_top_k,
                    "response_type": args.response_type,
                },
            }
        ]
    return payload


async def _ensure_query_ready(rag: Any) -> None:
    ensure = getattr(rag, "_ensure_lightrag_initialized", None)
    if ensure is None:
        return
    result = await ensure()
    if not result or not result.get("success"):
        error = (result or {}).get("error", "unknown error")
        raise RuntimeError(f"LightRAG initialization failed: {error}")


def _install_lightrag_insert_guard(rag: Any) -> None:
    """Raise when embedded LightRAG records failure without raising itself."""
    lightrag = getattr(rag, "lightrag", None)
    original = getattr(lightrag, "ainsert", None)
    if original is None or getattr(original, "_gems_rag_status_guard", False):
        return

    async def guarded_ainsert(*call_args: Any, **call_kwargs: Any) -> Any:
        result = await original(*call_args, **call_kwargs)
        ids = call_kwargs.get("ids")
        if ids is None and len(call_args) > 3:
            ids = call_args[3]
        doc_ids = None if ids is None else ([ids] if isinstance(ids, str) else ids)
        report = await lightrag_document_status_report(lightrag, doc_ids=doc_ids)
        if not report["complete"]:
            raise RuntimeError(
                f"embedded LightRAG document processing incomplete: {json.dumps(report)}"
            )
        return result

    guarded_ainsert._gems_rag_status_guard = True  # type: ignore[attr-defined]
    lightrag.ainsert = guarded_ainsert


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
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

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

    async def vision_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str, str]] | None = None, image_data: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        if messages is None and image_data:
            messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                },
            ]
            messages = [message for message in messages if message is not None]
        return await openai_complete_if_cache(
            args.vision_model,
            "" if messages else prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            messages=messages,
            api_key=api_key,
            base_url=args.base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.embedding_max_tokens,
        func=partial(openai_embed.func, model=args.embedding_model, api_key=api_key, base_url=args.base_url),
    )
    config = RAGAnythingConfig(
        working_dir=str(args.working_dir),
        parser=os.getenv("RAGANYTHING_PARSER", "mineru"),
        parse_method="auto",
        display_content_stats=False,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "entity_extraction_use_json": bool(
                getattr(args, "entity_extraction_json", False)
            )
        },
    )
    _skip_parser_check_for_preparsed_input(rag, args)
    return rag


def _skip_parser_check_for_preparsed_input(rag: Any, args: argparse.Namespace) -> None:
    ingestion_mode = getattr(args, "ingestion_mode", "shared_corpus")
    if ingestion_mode == "shared_corpus" or getattr(args, "command", None) == "query":
        rag._parser_installation_checked = True


if __name__ == "__main__":
    raise SystemExit(main())
