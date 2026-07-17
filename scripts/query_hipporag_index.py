#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.endpoint import probe_openai_endpoint

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "hipporag"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_SAVE_DIR = ROOT / "data" / "working" / "hipporag_index"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "hipporag" / "bin" / "python"
INDEX_SENTINEL = ".gems_rag_hipporag_index.json"
REQUIRED_MODULES = ("torch", "transformers", "igraph", "openai", "networkx", "pydantic", "tiktoken", "hipporag")


def main() -> int:
    args = _parse_args()
    reexec_code = _maybe_reexec(args.python)
    if reexec_code is not None:
        return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "index":
        return _index(args)
    if args.command == "query":
        return _query(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index or query HippoRAG over exported MRAG chunks.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.getenv("HIPPORAG_PYTHON", str(DEFAULT_ENV_PYTHON))),
        help="Optional isolated Python with HippoRAG dependencies. Defaults to data/working/venvs/hipporag/bin/python when present.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--allow-missing-api-key", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--llm-model", default=os.getenv("HIPPORAG_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument("--embedding-model", default=os.getenv("HIPPORAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", default=os.getenv("HIPPORAG_EMBEDDING_BASE_URL"))
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the local environment can run HippoRAG.")

    index = sub.add_parser("index", help="Index exported MRAG chunks into HippoRAG's ignored save directory.")
    index.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index.add_argument("--limit", type=int)

    query = sub.add_parser("query", help="Retrieve from an existing HippoRAG index.")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    query.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Fallback exported MRAG chunks used to enrich returned docs with metadata.")
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    import_errors = _import_errors(args)
    package_ok = (args.repo / "src" / "hipporag").exists()
    chunks = getattr(args, "chunks", DEFAULT_CHUNKS)
    chunks_found = chunks.exists()
    index_files = _index_files(args.save_dir)
    sentinel_path = args.save_dir / INDEX_SENTINEL
    sentinel = _read_json(sentinel_path)
    expected_identity = _index_identity(args, chunks)
    sentinel_matches_input = _sentinel_matches(sentinel, expected_identity)
    environment_ready = not import_errors and package_ok
    index_ready = sentinel_matches_input and bool(index_files)
    api_key = os.getenv(getattr(args, "api_key_env", "OPENAI_API_KEY"))
    allow_missing_api_key = bool(getattr(args, "allow_missing_api_key", False))
    credential_available = bool(api_key) or allow_missing_api_key
    llm_base_url = _llm_base_url(args)
    embedding_base_url = _embedding_base_url(args)
    endpoints = _endpoint_reports(
        llm_base_url,
        embedding_base_url,
        api_key=api_key or ("local" if allow_missing_api_key else None),
    )
    checked_endpoints = [endpoint for endpoint in endpoints.values() if endpoint["checked"]]
    endpoint_reachable = (
        all(endpoint["reachable"] is True for endpoint in checked_endpoints) if checked_endpoints else None
    )
    endpoint_usable = all(endpoint["usable"] is True for endpoint in checked_endpoints)
    model_service_ready = credential_available and endpoint_usable
    return {
        "runnable": environment_ready and model_service_ready and index_ready,
        "environment_ready": environment_ready,
        "input_ready": chunks_found,
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_required_modules": sorted(import_errors),
        "missing_or_failed_imports": import_errors,
        "repo": str(args.repo),
        "package_source_found": package_ok,
        "repo_found": package_ok,
        "chunks": str(chunks),
        "chunks_found": chunks_found,
        "chunks_sha256": expected_identity.get("chunks_sha256"),
        "save_dir": str(args.save_dir),
        "save_dir_exists": args.save_dir.exists(),
        "index_ready": index_ready,
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "sentinel": str(sentinel_path),
        "sentinel_found": sentinel_path.exists(),
        "sentinel_matches_input": sentinel_matches_input,
        "indexed_docs": sentinel.get("indexed_docs") if isinstance(sentinel, dict) else None,
        "api_key_env": getattr(args, "api_key_env", "OPENAI_API_KEY"),
        "api_key_present": bool(api_key),
        "allow_missing_api_key": allow_missing_api_key,
        "credential_available": credential_available,
        "api_key_usable": model_service_ready,
        "base_url": getattr(args, "base_url", None),
        "llm_base_url": llm_base_url,
        "embedding_base_url": embedding_base_url,
        "endpoint": endpoints["llm"],
        "llm_endpoint": endpoints["llm"],
        "embedding_endpoint": endpoints["embedding"],
        "endpoint_reachable": endpoint_reachable,
        "endpoint_usable": endpoint_usable,
        "model_service_ready": model_service_ready,
        "llm_model": getattr(args, "llm_model", "gpt-4o-mini"),
        "embedding_model": getattr(args, "embedding_model", "text-embedding-3-small"),
        "notes": (
            "The retrieval-only environment patches HippoRAG's optional CUDA backends to load lazily. "
            "Indexing still requires OpenAI-compatible chat and embedding services."
        ),
    }


def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["input_ready"]:
        print(json.dumps({"error": "chunks_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["model_service_ready"]:
        error = "missing_api_key" if not report["credential_available"] else "model_service_unavailable"
        print(json.dumps({"error": error, **report}, indent=2), file=sys.stderr)
        return 2
    rows = _read_jsonl(args.chunks)
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    if not rows:
        print(json.dumps({"error": "empty_corpus", "chunks": str(args.chunks), "limit": args.limit}), file=sys.stderr)
        return 2
    docs = [row["text"] for row in rows]
    sentinel_path = args.save_dir / INDEX_SENTINEL
    sentinel_path.unlink(missing_ok=True)
    try:
        hipporag = _hipporag(args)
        hipporag.index(docs=docs)
    except Exception as exc:
        print(json.dumps({"error": "hipporag_index_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    sidecar = _write_metadata_sidecar(args.save_dir, rows)
    sentinel = {
        **_index_identity(args, args.chunks),
        "complete": True,
        "indexed_docs": len(docs),
        "limit": args.limit,
    }
    _write_json_atomic(sentinel_path, sentinel)
    print(
        json.dumps(
            {
                "indexed": True,
                "docs": len(docs),
                "save_dir": str(args.save_dir),
                "metadata_sidecar": str(sidecar),
                "sentinel": str(sentinel_path),
            }
        )
    )
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["model_service_ready"]:
        error = "missing_api_key" if not report["credential_available"] else "model_service_unavailable"
        print(json.dumps({"error": error, **report}, indent=2), file=sys.stderr)
        return 2
    if not report["index_ready"]:
        print(json.dumps({"error": "index_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    try:
        hipporag = _hipporag(args)
        results = hipporag.retrieve(queries=[args.question], num_to_retrieve=max(args.top_k, 1))
        first = results[0]
    except Exception as exc:
        print(json.dumps({"error": "hipporag_query_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    manifest = _load_metadata_by_text(args.save_dir, args.chunks)
    contexts = [
        _context_from_hit(doc, score, idx, manifest)
        for idx, (doc, score) in enumerate(zip(first.docs, first.doc_scores, strict=False), 1)
    ]
    print(json.dumps({"question": args.question, "contexts": contexts, "metadata_sidecar": str(_metadata_sidecar(args.save_dir))}, ensure_ascii=False))
    return 0


def _hipporag(args: argparse.Namespace):
    _ensure_api_key(args)
    sys.path.insert(0, str(args.repo / "src"))
    from hipporag import HippoRAG

    kwargs = {
        "save_dir": str(args.save_dir),
        "llm_model_name": args.llm_model,
        "embedding_model_name": args.embedding_model,
    }
    if _llm_base_url(args):
        kwargs["llm_base_url"] = _llm_base_url(args)
    if _embedding_base_url(args):
        kwargs["embedding_base_url"] = _embedding_base_url(args)
    rag = HippoRAG(**kwargs)
    reasoning_effort = getattr(args, "reasoning_effort", None)
    if reasoning_effort:
        generate_params = dict(rag.llm_model.llm_config.generate_params)
        generate_params["reasoning_effort"] = reasoning_effort
        rag.llm_model.batch_upsert_llm_config({"generate_params": generate_params})
        cache_path = Path(rag.llm_model.cache_file_name)
        rag.llm_model.cache_file_name = str(
            cache_path.with_name(
                f"{cache_path.stem}_reasoning_{reasoning_effort}{cache_path.suffix}"
            )
        )
    return rag


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metadata_sidecar(save_dir: Path) -> Path:
    return save_dir / "mrag_chunk_manifest.jsonl"


def _write_metadata_sidecar(save_dir: Path, rows: list[dict[str, Any]]) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar = _metadata_sidecar(save_dir)
    with sidecar.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "doc_id": row.get("doc_id"),
                "title": row.get("title"),
                "text": row.get("text", ""),
                "metadata": row.get("metadata") or {},
                "text_hash": _text_hash(str(row.get("text", ""))),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return sidecar


def _load_metadata_by_text(save_dir: Path, chunks: Path | None = None) -> dict[str, dict[str, Any]]:
    sources = [_metadata_sidecar(save_dir)]
    if chunks is not None:
        sources.append(chunks)
    manifest: dict[str, dict[str, Any]] = {}
    for path in sources:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            text = str(row.get("text", ""))
            if not text:
                continue
            manifest[_text_hash(text)] = row
    return manifest


def _context_from_hit(doc: str, score: Any, idx: int, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = manifest.get(_text_hash(doc), {})
    metadata = dict(row.get("metadata") or {})
    doc_id = str(row.get("doc_id") or f"hipporag:{idx}")
    if row.get("title"):
        metadata["title"] = row.get("title")
    metadata["doc_id"] = doc_id
    metadata["source"] = "hipporag"
    return {
        "name": doc_id,
        "kind": "chunk" if row else "tool_trace",
        "text": str(row.get("text") or doc),
        "score": float(score) if score is not None else None,
        "metadata": metadata,
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_files(save_dir: Path) -> list[str]:
    if not save_dir.exists():
        return []
    excluded = {_metadata_sidecar(save_dir).resolve(), (save_dir / INDEX_SENTINEL).resolve()}
    files = []
    for path in save_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.resolve() in excluded:
                continue
        except OSError:
            pass
        files.append(str(path.relative_to(save_dir)))
    return sorted(files)


def _import_errors(args: argparse.Namespace) -> dict[str, str]:
    source = args.repo / "src"
    if source.exists():
        sys.path.insert(0, str(source))
    errors: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as exc:
            errors[name] = repr(exc)
    return errors


def _endpoint_reports(
    llm_base_url: str | None,
    embedding_base_url: str | None,
    *,
    api_key: str | None,
) -> dict[str, dict[str, Any]]:
    cache: dict[str | None, dict[str, Any]] = {}

    def probe(url: str | None) -> dict[str, Any]:
        if url not in cache:
            cache[url] = probe_openai_endpoint(url, api_key=api_key)
        return cache[url]

    return {"llm": probe(llm_base_url), "embedding": probe(embedding_base_url)}


def _llm_base_url(args: argparse.Namespace) -> str | None:
    return getattr(args, "llm_base_url", None) or getattr(args, "base_url", None)


def _embedding_base_url(args: argparse.Namespace) -> str | None:
    return getattr(args, "embedding_base_url", None) or getattr(args, "base_url", None)


def _ensure_api_key(args: argparse.Namespace) -> str:
    api_key_env = getattr(args, "api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key and getattr(args, "allow_missing_api_key", False):
        api_key = "local"
    if not api_key:
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def _index_identity(args: argparse.Namespace, chunks: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "chunks": str(chunks.resolve()),
        "chunks_sha256": _file_digest(chunks) if chunks.exists() else None,
        "llm_model": getattr(args, "llm_model", "gpt-4o-mini"),
        "embedding_model": getattr(args, "embedding_model", "text-embedding-3-small"),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "llm_base_url": _llm_base_url(args),
        "embedding_base_url": _embedding_base_url(args),
    }


def _sentinel_matches(sentinel: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(sentinel, dict) or sentinel.get("complete") is not True:
        return False
    return all(sentinel.get(key) == value for key, value in expected.items())


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _maybe_reexec(python: Path) -> int | None:
    if not python.exists():
        return None
    try:
        if python.resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        return None
    completed = subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
