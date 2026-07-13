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

DEFAULT_RAGANYTHING_REPO = ROOT / "external" / "rag-implementations" / "rag-anything"
DEFAULT_LIGHTRAG_REPO = ROOT / "external" / "rag-implementations" / "lightrag"
DEFAULT_CONTENT_LIST = ROOT / "data" / "working" / "mrag_corpus" / "raganything_content_list.json"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "raganything_index"
DEFAULT_NATIVE_WORKING_DIR = ROOT / "data" / "working" / "raganything_native_pdf_index"
DEFAULT_PDF = ROOT / "data" / "extracted" / "MRAG-20260708T114057Z-3" / "MRAG" / "mutcd11theditionr1hl.pdf"


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

    api_key = _api_key(args)
    try:
        rag = _make_rag(args, api_key)
    except Exception as exc:
        print(f"failed to initialize RAG-Anything adapter: {exc!r}", file=sys.stderr)
        return 2

    if args.command == "index":
        if args.ingestion_mode == "native_pdf":
            await rag.process_document_complete(
                file_path=str(args.pdf),
                doc_id=args.doc_id,
                display_stats=args.display_stats,
            )
            source_count = 1
        else:
            content_list = json.loads(args.content_list.read_text(encoding="utf-8"))
            await rag.insert_content_list(
                content_list=content_list,
                file_path=args.file_path,
                doc_id=args.doc_id,
                display_stats=args.display_stats,
            )
            source_count = len(content_list)
        print(json.dumps({"indexed": True, "ingestion_mode": args.ingestion_mode, "sources": source_count, "working_dir": str(args.working_dir)}))
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

    index = sub.add_parser("index", help="Build or extend the ignored RAG-Anything index.")
    _add_common_args(index)
    index.add_argument("--content-list", type=Path, default=DEFAULT_CONTENT_LIST)
    index.add_argument("--file-path", default="mutcd11theditionr1hl.pdf")
    index.add_argument("--doc-id", default="mutcd11e")
    index.add_argument("--display-stats", action="store_true")
    index.add_argument("--force", action="store_true", help="Delete the existing ignored index directory before indexing.")

    query = sub.add_parser("query", help="Query an existing ignored RAG-Anything index.")
    _add_common_args(query)
    query.add_argument("--question", required=True)
    query.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix", "bypass"])
    query.add_argument("--top-k", type=int, default=12)
    query.add_argument("--chunk-top-k", type=int, default=12)
    query.add_argument("--only-need-context", action="store_true", help="Return retrieved context instead of a generated answer.")
    query.add_argument("--response-type", default="Multiple Paragraphs")
    query.add_argument("--json", action="store_true", help="Print a JSON wrapper instead of raw result text.")

    args = parser.parse_args()
    if args.working_dir is None:
        args.working_dir = DEFAULT_NATIVE_WORKING_DIR if args.ingestion_mode == "native_pdf" else DEFAULT_WORKING_DIR
    if args.command == "index" and args.force and args.working_dir.exists():
        shutil.rmtree(args.working_dir)
    if args.command in {"index", "query"}:
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
    environment_ready = args.repo.exists() and args.lightrag_repo.exists() and not import_errors
    index_ready = bool(index_files)
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
        "content_list": str(content_list),
        "content_list_found": content_list.exists(),
        "pdf": str(pdf),
        "pdf_found": pdf.exists(),
        "ingestion_mode": ingestion_mode,
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
        "notes": "The default RAG-Anything adapter uses LightRAG plus OpenAI-compatible LLM, vision, and embedding calls. Full multimodal indexing also needs RAG-Anything's parser dependencies.",
    }


def _index_files(working_dir: Path) -> list[str]:
    if not working_dir.exists():
        return []
    return sorted(str(path.relative_to(working_dir)) for path in working_dir.rglob("*") if path.is_file())


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
    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


if __name__ == "__main__":
    raise SystemExit(main())
