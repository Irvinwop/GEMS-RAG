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
DEFAULT_RAGANYTHING_REPO = ROOT / "external" / "rag-implementations" / "rag-anything"
DEFAULT_LIGHTRAG_REPO = ROOT / "external" / "rag-implementations" / "lightrag"
DEFAULT_CONTENT_LIST = ROOT / "data" / "working" / "mrag_corpus" / "raganything_content_list.json"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "raganything_index"


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
        content_list = json.loads(args.content_list.read_text(encoding="utf-8"))
        await rag.insert_content_list(
            content_list=content_list,
            file_path=args.file_path,
            doc_id=args.doc_id,
            display_stats=args.display_stats,
        )
        print(json.dumps({"indexed": True, "items": len(content_list), "working_dir": str(args.working_dir)}))
        return 0
    if args.command == "query":
        result = await rag.aquery(args.question, mode=args.mode)
        if args.json:
            print(json.dumps({"mode": args.mode, "question": args.question, "result": result}, ensure_ascii=False))
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
    query.add_argument("--json", action="store_true", help="Print a JSON wrapper instead of raw result text.")

    args = parser.parse_args()
    if args.command == "index" and args.force and args.working_dir.exists():
        shutil.rmtree(args.working_dir)
    if args.command in {"index", "query"}:
        args.working_dir.mkdir(parents=True, exist_ok=True)
    return args


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_RAGANYTHING_REPO, help="Path to cloned RAG-Anything repository.")
    parser.add_argument("--lightrag-repo", type=Path, default=DEFAULT_LIGHTRAG_REPO, help="Path to cloned LightRAG repository.")
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR, help="Ignored RAG-Anything index directory.")
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
    api_key_present = bool(os.getenv(args.api_key_env))
    api_key_usable = api_key_present or bool(args.allow_missing_api_key)
    index_files = sorted(
        str(path.relative_to(args.working_dir))
        for path in args.working_dir.glob("*")
        if args.working_dir.exists()
    )
    return {
        "runnable": args.repo.exists() and args.lightrag_repo.exists() and not import_errors and api_key_usable,
        "repo": str(args.repo),
        "repo_found": args.repo.exists(),
        "lightrag_repo": str(args.lightrag_repo),
        "lightrag_repo_found": args.lightrag_repo.exists(),
        "working_dir": str(args.working_dir),
        "working_dir_exists": args.working_dir.exists(),
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "content_list": str(args.content_list),
        "content_list_found": args.content_list.exists(),
        "api_key_env": args.api_key_env,
        "api_key_present": api_key_present,
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "api_key_usable": api_key_usable,
        "missing_or_failed_imports": import_errors,
        "notes": "The default RAG-Anything adapter uses LightRAG plus OpenAI-compatible LLM, vision, and embedding calls. Full multimodal indexing also needs RAG-Anything's parser dependencies.",
    }


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
