#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_chunks, load_figures
from gems_rag.endpoint import probe_openai_endpoint
from gems_rag.index_completion import value_fingerprint
from gems_rag.lightrag_compat import (
    cap_completion_tokens,
    lightrag_document_status_report,
)

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "megarag"
DEFAULT_LIGHTRAG_REPO = ROOT / "external" / "rag-implementations" / "megarag-lightrag-v1.4.3"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "megarag_index"
DEFAULT_PAGES_CONTENT = ROOT / "data" / "working" / "megarag_corpus" / "pages_content.json"
DEFAULT_ADDON_CONFIG = ROOT / "configs" / "megarag-addon-params.yaml"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "megarag" / "bin" / "python"
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
INDEX_SENTINEL = ".gems_rag_megarag_index.json"
INDEX_ATTEMPT = ".gems_rag_megarag_attempt.json"
CORE_INDEX_FILES = (
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "graph_chunk_entity_relation.graphml",
)
CHUNK_ID_RE = re.compile(r"\[Chunk ID:\s*([^\]]+?)\s*\]")
PAGE_IMAGE_RE = re.compile(r"filename:(page_(\d+)\.(?:png|jpe?g))", flags=re.IGNORECASE)


def main() -> int:
    args = _parse_args()
    if args.command in {"check", "index", "query"}:
        reexec_code = _maybe_reexec(args.python)
        if reexec_code is not None:
            return reexec_code
    if args.command == "prepare":
        report = prepare_pages_content(
            args.mrag_dir,
            args.pages_content,
            start_page=args.start_page,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["pages"] else 2

    _add_repo_paths(args)
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["runnable"] else 2
    try:
        if args.command == "index":
            return asyncio.run(_index(args))
        if args.command == "query":
            return asyncio.run(_query(args))
    except KeyboardInterrupt:
        return 130
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, index, or query the official MegaRAG MMKG over extracted MUTCD pages."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--lightrag-repo", type=Path, default=DEFAULT_LIGHTRAG_REPO)
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--pages-content", type=Path, default=DEFAULT_PAGES_CONTENT)
    parser.add_argument("--addon-config", type=Path, default=DEFAULT_ADDON_CONFIG)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.getenv("MEGARAG_PYTHON", str(DEFAULT_ENV_PYTHON))),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--allow-missing-api-key", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--llm-model", default=os.getenv("MEGARAG_LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument(
        "--vision-model",
        default=os.getenv("MEGARAG_VISION_MODEL", DEFAULT_LLM_MODEL),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Hard ceiling for each MegaRAG text or vision completion.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--start-page", type=int, help="Expected first PDF page for a smoke index.")
    check.add_argument("--limit", type=int, help="Expected smoke-index page limit.")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--start-page", type=int, help="Start at this 1-based PDF page.")
    prepare.add_argument("--limit", type=int, help="Prepare only the first N PDF pages for a smoke index.")
    index = subparsers.add_parser("index")
    index.add_argument("--force", action="store_true")
    index.add_argument("--start-page", type=int, help="First PDF page used by the prepared smoke index.")
    index.add_argument("--limit", type=int, help="Page limit used by the prepared smoke index.")
    query = subparsers.add_parser("query")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    query.add_argument("--start-page", type=int, help="Expected first PDF page for a smoke index.")
    query.add_argument("--limit", type=int, help="Expected smoke-index page limit.")
    args = parser.parse_args()
    if args.llm_max_tokens is not None and args.llm_max_tokens <= 0:
        parser.error("--llm-max-tokens must be positive")
    if getattr(args, "start_page", None) is not None and args.start_page <= 0:
        parser.error("--start-page must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def prepare_pages_content(
    mrag_dir: Path,
    out_path: Path,
    *,
    start_page: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    chunks_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in load_chunks(mrag_dir):
        page = _integer(chunk.get("page_pdf"))
        if page is not None:
            chunks_by_page[page].append(chunk)

    figure_paths_by_page: dict[int, list[str]] = defaultdict(list)
    missing_figure_images = 0
    for figure in load_figures(mrag_dir):
        page = _integer(figure.get("page_pdf"))
        if page is None:
            continue
        path = _local_figure_path(mrag_dir / "figures", figure)
        if path is None:
            missing_figure_images += 1
            continue
        resolved = str(path.resolve())
        if resolved not in figure_paths_by_page[page]:
            figure_paths_by_page[page].append(resolved)

    page_paths = sorted((mrag_dir / "page_images").glob("page_*.*"), key=_page_number)
    if start_page is not None:
        page_paths = [path for path in page_paths if _page_number(path) >= start_page]
    if limit is not None:
        page_paths = page_paths[:limit]
    payload: dict[str, dict[str, Any]] = {}
    indexed_chunks = 0
    indexed_figures = 0
    for index, page_path in enumerate(page_paths):
        page_pdf = _page_number(page_path)
        chunks = chunks_by_page.get(page_pdf, [])
        figures = figure_paths_by_page.get(page_pdf, [])
        indexed_chunks += len(chunks)
        indexed_figures += len(figures)
        chunk_text = "\n\n".join(_megarag_chunk_text(chunk) for chunk in chunks)
        payload[str(index)] = {
            "text": f"[PDF Page: {page_pdf}]" + (f"\n\n{chunk_text}" if chunk_text else ""),
            "page_image": str(page_path.resolve()),
            "figure_images": figures,
            "page_pdf": page_pdf,
            "chunk_ids": [str(chunk.get("chunk_id")) for chunk in chunks],
        }

    _write_json_atomic(out_path, payload)
    return {
        "status": "prepared",
        "mrag_dir": str(mrag_dir),
        "pages_content": str(out_path),
        "start_page": start_page,
        "limit": limit,
        "pages": len(payload),
        "chunks": indexed_chunks,
        "figure_images": indexed_figures,
        "missing_figure_images": missing_figure_images,
        "sha256": _file_digest(out_path),
    }


async def retrieve_dual_context(rag: Any, question: str, *, top_k: int) -> tuple[str, str]:
    QueryParam = _query_param_class()
    kg_param = QueryParam(
        mode="hybrid",
        top_k=top_k,
        chunk_top_k=top_k,
        only_need_context=True,
        enable_rerank=False,
    )
    page_param = QueryParam(
        mode="naive",
        top_k=top_k,
        chunk_top_k=top_k,
        only_need_context=True,
        enable_rerank=False,
    )
    kg_context, page_context = await asyncio.gather(
        rag.aquery(question, param=kg_param),
        rag.aquery(question, param=page_param),
    )
    return _context_text(kg_context), _context_text(page_context)


def query_payload(
    question: str,
    kg_context: str,
    page_context: str,
    mrag_dir: Path,
    *,
    top_k: int,
) -> dict[str, Any]:
    combined = (
        "===== MegaRAG MMKG retrieval branch =====\n"
        f"{kg_context}\n\n"
        "===== MegaRAG page-image retrieval branch =====\n"
        f"{page_context}"
    )
    chunk_ids = _ordered_unique(CHUNK_ID_RE.findall(combined))
    chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in load_chunks(mrag_dir)}
    chunks = []
    for rank, chunk_id in enumerate(chunk_ids[:top_k], 1):
        chunk = chunk_by_id.get(chunk_id)
        if chunk is not None:
            chunks.append({**chunk, "score": 1.0 / rank})

    pages = []
    seen_pages = set()
    for filename, page_text in PAGE_IMAGE_RE.findall(combined):
        page_pdf = int(page_text)
        if page_pdf in seen_pages:
            continue
        image_path = mrag_dir / "page_images" / filename
        if not image_path.exists():
            continue
        pages.append(
            {
                "page_id": f"page:{page_pdf:04d}",
                "page_pdf": page_pdf,
                "image_path": str(image_path.resolve()),
                "text": f"MegaRAG page-image retrieval selected MUTCD PDF page {page_pdf}.",
                "score": 1.0 / (len(pages) + 1),
                "metadata": {"retrieval_branch": "page_image", "source": "megarag"},
            }
        )
        seen_pages.add(page_pdf)
        if len(pages) >= top_k:
            break

    metadata = {
        "method": "megarag",
        "retrieval_branches": ["mmkg_hybrid", "page_image_naive"],
        "only_need_context": True,
        "upstream_second_generation_bypassed": True,
        "final_generation": "harness_model_matrix",
        "top_k": top_k,
    }
    return {
        "question": question,
        "chunks": chunks,
        "pages": pages,
        "contexts": [
            {
                "name": "megarag:dual_retrieval_context",
                "kind": "tool_trace",
                "text": combined,
                "metadata": metadata,
            }
        ],
        "debug": metadata,
    }


async def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"] or not report["api_key_usable"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["input_ready"]:
        print(json.dumps({"error": "input_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    if report["index_ready"] and not args.force:
        print(json.dumps({"status": "already_indexed", **report}, indent=2))
        return 0
    identity = _index_identity(args)
    attempt_path = args.working_dir / INDEX_ATTEMPT
    previous_attempt = _read_json(attempt_path)
    if (
        not args.force
        and _working_state_exists(args.working_dir)
        and previous_attempt != identity
    ):
        print(
            json.dumps(
                {
                    "error": "megarag_index_input_changed_use_force",
                    "working_dir": str(args.working_dir),
                    "previous_attempt": previous_attempt,
                    "requested_attempt": identity,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    if args.force and args.working_dir.exists():
        shutil.rmtree(args.working_dir)
    args.working_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(attempt_path, identity)
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel_path.unlink(missing_ok=True)

    api_key = _api_key(args)
    rag = None
    index_status: dict[str, Any] | None = None
    try:
        rag, token_tracker = await _initialize_rag(args, api_key)
        await rag.ainsert(
            input=args.pages_content.read_text(encoding="utf-8"),
            split_by_page=True,
            ids=identity["document_id"],
            file_paths=str(args.mrag_dir / "mutcd11theditionr1hl.pdf"),
        )
        index_status = await lightrag_document_status_report(
            rag,
            doc_ids=[identity["document_id"]],
        )
    except Exception as exc:
        print(json.dumps({"error": "megarag_index_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    finally:
        if rag is not None:
            await rag.finalize_storages()

    if not index_status or not index_status["complete"]:
        print(
            json.dumps(
                {
                    "error": "megarag_index_incomplete",
                    "document_status": index_status,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    sentinel = {
        **identity,
        "token_usage": str(token_tracker),
    }
    _write_json_atomic(sentinel_path, sentinel)
    final_report = _dependency_report(args)
    print(json.dumps({"status": "indexed", **final_report}, indent=2, ensure_ascii=False))
    return 0 if final_report["index_ready"] else 2


async def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    api_key = _api_key(args)
    rag = None
    try:
        rag, token_tracker = await _initialize_rag(args, api_key)
        kg_context, page_context = await retrieve_dual_context(
            rag,
            args.question,
            top_k=max(args.top_k, 1),
        )
    except Exception as exc:
        print(json.dumps({"error": "megarag_query_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    finally:
        if rag is not None:
            await rag.finalize_storages()
    payload = query_payload(
        args.question,
        kg_context,
        page_context,
        args.mrag_dir,
        top_k=max(args.top_k, 1),
    )
    payload["debug"]["retrieval_token_usage"] = str(token_tracker)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    import_errors = _import_errors(["lightrag", "megarag", "torch", "transformers", "yaml", "PIL"])
    repo_found = (args.repo / "megarag" / "megarag.py").exists()
    lightrag_repo_found = (args.lightrag_repo / "lightrag" / "base.py").exists()
    pages_content_found = args.pages_content.exists()
    addon_config_found = args.addon_config.exists()
    api_key = os.getenv(args.api_key_env)
    credential_available = bool(api_key) or bool(args.allow_missing_api_key)
    endpoint = probe_openai_endpoint(
        args.base_url,
        api_key=api_key or ("local" if args.allow_missing_api_key else None),
    )
    endpoint_usable = endpoint["usable"] if endpoint["checked"] else True
    api_key_usable = credential_available and endpoint_usable
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel = _read_json(sentinel_path)
    identity = _index_identity(args)
    current_digest = identity["pages_content_sha256"]
    sentinel_matches_input = bool(
        isinstance(sentinel, dict)
        and current_digest
        and all(
            sentinel.get(key) == identity[key]
            for key in (
                "pages_content",
                "pages_content_sha256",
                "start_page",
                "limit",
                "document_id",
            )
        )
    )
    sentinel_matches_backend = bool(
        isinstance(sentinel, dict)
        and all(
            sentinel.get(key) == identity[key]
            for key in (
                "embedding_model",
                "llm_model",
                "vision_model",
                "endpoint",
                "reasoning_effort",
                "llm_max_tokens",
            )
        )
    )
    attempt_path = args.working_dir / INDEX_ATTEMPT
    attempt_matches_input = _read_json(attempt_path) == identity
    core_files = {name: (args.working_dir / name).exists() for name in CORE_INDEX_FILES}
    index_ready = sentinel_matches_input and sentinel_matches_backend and all(core_files.values())
    environment_ready = repo_found and lightrag_repo_found and not import_errors
    input_ready = pages_content_found and addon_config_found
    return {
        "runnable": environment_ready and input_ready and api_key_usable and index_ready,
        "environment_ready": environment_ready,
        "input_ready": input_ready,
        "index_ready": index_ready,
        "repo": str(args.repo),
        "repo_found": repo_found,
        "lightrag_repo": str(args.lightrag_repo),
        "lightrag_repo_found": lightrag_repo_found,
        "lightrag_version": "v1.4.3",
        "working_dir": str(args.working_dir),
        "pages_content": str(args.pages_content.resolve()),
        "pages_content_found": pages_content_found,
        "pages_content_sha256": current_digest,
        "start_page": getattr(args, "start_page", None),
        "limit": getattr(args, "limit", None),
        "addon_config": str(args.addon_config),
        "addon_config_found": addon_config_found,
        "sentinel": str(sentinel_path),
        "sentinel_found": sentinel_path.exists(),
        "sentinel_matches_input": sentinel_matches_input,
        "sentinel_matches_backend": sentinel_matches_backend,
        "attempt": str(attempt_path),
        "attempt_matches_input": attempt_matches_input,
        "document_id": identity["document_id"],
        "core_index_files": core_files,
        "api_key_env": args.api_key_env,
        "api_key_present": bool(api_key),
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "credential_available": credential_available,
        "api_key_usable": api_key_usable,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "endpoint_reachable": endpoint["reachable"],
        "model_service_ready": api_key_usable,
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model,
        "vision_model": args.vision_model,
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "llm_max_tokens": getattr(args, "llm_max_tokens", None),
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_or_failed_imports": import_errors,
        "notes": (
            "MegaRAG uses the official MMKG and page-image retrievers. The adapter bypasses only the "
            "upstream second answer-generation stage so all methods share the selected harness generator."
        ),
    }


async def _initialize_rag(args: argparse.Namespace, api_key: str) -> tuple[Any, Any]:
    import torch
    import yaml
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.types import GPTKeywordExtractionFormat
    from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs
    from megarag import MegaRAG
    from megarag.llms.hf import hf_gme_embed
    from megarag.llms.openai import openai_complete_if_cache
    from transformers import AutoModel

    addon_payload = yaml.safe_load(args.addon_config.read_text(encoding="utf-8")) or {}
    addon_params = addon_payload.get("addon_params", addon_payload)
    device = _torch_device(torch, args.device)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
    }
    embed_model = AutoModel.from_pretrained(args.embedding_model, **model_kwargs).to(device).eval()

    @wrap_embedding_func_with_attrs(embedding_dim=args.embedding_dim, max_token_size=32768)
    async def embed_func(texts=None, images=None, is_query=False):
        return await hf_gme_embed(
            embed_model=embed_model,
            texts=list(texts or []),
            images=list(images or []),
            is_query=is_query,
        )

    token_tracker = TokenTracker()

    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        kwargs.pop("_priority", None)
        cap_completion_tokens(kwargs, getattr(args, "llm_max_tokens", None))
        model = _completion_model(args, input_images)
        reasoning_effort = getattr(args, "reasoning_effort", None)
        if reasoning_effort and model == args.llm_model:
            kwargs.setdefault("reasoning_effort", reasoning_effort)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        return await openai_complete_if_cache(
            model,
            prompt,
            input_images=input_images,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=args.base_url,
            api_key=api_key,
            token_tracker=token_tracker,
            **kwargs,
        )

    rag = MegaRAG(
        working_dir=str(args.working_dir),
        llm_model_func=llm_func,
        embedding_func=embed_func,
        addon_params=addon_params,
        auto_manage_storages_states=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, token_tracker


def _completion_model(args: argparse.Namespace, input_images: Any) -> str:
    return args.vision_model if input_images else args.llm_model


def _megarag_chunk_text(chunk: dict[str, Any]) -> str:
    header = (
        f"[Chunk ID: {chunk.get('chunk_id')}]\n"
        f"Section {chunk.get('section_id')} {chunk.get('content_type')} {chunk.get('ordinal')} - "
        f"{chunk.get('section_title')}"
    )
    return f"{header}\n{str(chunk.get('text') or '').strip()}"


def _local_figure_path(figures_dir: Path, record: dict[str, Any]) -> Path | None:
    raw = str(record.get("image_path") or "")
    candidates = [figures_dir / Path(raw).name]
    canonical = str(record.get("canonical_id") or "").strip()
    page = _integer(record.get("page_pdf"))
    kind = str(record.get("kind") or "figure").lower()
    if canonical and page is not None:
        candidates.append(figures_dir / f"{kind}_{canonical}_p{page:04d}.png")
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _add_repo_paths(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.lightrag_repo))
    sys.path.insert(0, str(args.repo))


def _import_errors(module_names: Iterable[str]) -> dict[str, str]:
    errors = {}
    for name in module_names:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as exc:
            errors[name] = repr(exc)
    return errors


def _api_key(args: argparse.Namespace) -> str:
    key = os.getenv(args.api_key_env)
    if key:
        return key
    if args.allow_missing_api_key:
        return "local"
    raise RuntimeError(f"missing API key env var: {args.api_key_env}")


def _query_param_class():
    from lightrag.base import QueryParam

    return QueryParam


def _torch_device(torch: Any, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_reexec(python: Path) -> int | None:
    if not python.exists():
        return None
    try:
        if python.resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        return None
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_id(args: argparse.Namespace) -> str:
    digest = _file_digest(args.pages_content) if args.pages_content.exists() else "missing"
    return f"mutcd11e-{digest[:24]}"


def _index_identity(args: argparse.Namespace) -> dict[str, Any]:
    pages_digest = _file_digest(args.pages_content) if args.pages_content.exists() else None
    return {
        "pages_content": str(args.pages_content.resolve()),
        "pages_content_sha256": pages_digest,
        "start_page": getattr(args, "start_page", None),
        "limit": getattr(args, "limit", None),
        "document_id": _document_id(args),
        "embedding_model": getattr(args, "embedding_model", DEFAULT_EMBEDDING_MODEL),
        "llm_model": getattr(args, "llm_model", DEFAULT_LLM_MODEL),
        "vision_model": getattr(args, "vision_model", DEFAULT_LLM_MODEL),
        "endpoint": value_fingerprint(getattr(args, "base_url", None)),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "llm_max_tokens": getattr(args, "llm_max_tokens", None),
    }


def _working_state_exists(working_dir: Path) -> bool:
    if not working_dir.exists():
        return False
    ignored = {INDEX_SENTINEL, INDEX_ATTEMPT}
    return any(path.name not in ignored for path in working_dir.iterdir())


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _page_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"page image has no numeric page: {path}")
    return int(match.group(1))


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _context_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
