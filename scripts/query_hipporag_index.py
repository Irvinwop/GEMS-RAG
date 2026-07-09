#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "hipporag"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_SAVE_DIR = ROOT / "data" / "working" / "hipporag_index"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "hipporag" / "bin" / "python"
REQUIRED_MODULES = ["torch", "transformers", "igraph", "openai", "litellm"]


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
    parser.add_argument("--llm-model", default=os.getenv("HIPPORAG_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument("--embedding-model", default=os.getenv("HIPPORAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", default=os.getenv("HIPPORAG_EMBEDDING_BASE_URL"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the local environment can run HippoRAG.")

    index = sub.add_parser("index", help="Index exported MRAG chunks into HippoRAG's ignored save directory.")
    index.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index.add_argument("--limit", type=int)

    query = sub.add_parser("query", help="Retrieve from an existing HippoRAG index.")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    package_ok = (args.repo / "src" / "hipporag").exists()
    index_files = _index_files(args.save_dir)
    environment_ready = not missing and package_ok
    index_ready = bool(index_files)
    return {
        "runnable": environment_ready and index_ready,
        "environment_ready": environment_ready,
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_required_modules": missing,
        "repo": str(args.repo),
        "package_source_found": package_ok,
        "save_dir": str(args.save_dir),
        "save_dir_exists": args.save_dir.exists(),
        "index_ready": index_ready,
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "notes": "Install external/rag-implementations/hipporag/requirements.txt or the hipporag package in an isolated environment before indexing.",
    }


def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    hipporag = _hipporag(args)
    docs = [row["text"] for row in _read_jsonl(args.chunks)]
    if args.limit:
        docs = docs[: args.limit]
    hipporag.index(docs=docs)
    print(json.dumps({"indexed": True, "docs": len(docs), "save_dir": str(args.save_dir)}))
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["index_ready"]:
        print(json.dumps({"error": "index_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    hipporag = _hipporag(args)
    results = hipporag.retrieve(queries=[args.question], num_to_retrieve=args.top_k)
    first = results[0]
    contexts = [
        {"text": doc, "score": float(score) if score is not None else None, "name": f"hipporag:{idx}"}
        for idx, (doc, score) in enumerate(zip(first.docs, first.doc_scores, strict=False), 1)
    ]
    print(json.dumps({"question": args.question, "contexts": contexts}, ensure_ascii=False))
    return 0


def _hipporag(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo / "src"))
    from hipporag import HippoRAG

    kwargs = {
        "save_dir": str(args.save_dir),
        "llm_model_name": args.llm_model,
        "embedding_model_name": args.embedding_model,
    }
    if args.llm_base_url:
        kwargs["llm_base_url"] = args.llm_base_url
    if args.embedding_base_url:
        kwargs["embedding_base_url"] = args.embedding_base_url
    return HippoRAG(**kwargs)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _index_files(save_dir: Path) -> list[str]:
    if not save_dir.exists():
        return []
    return sorted(str(path.relative_to(save_dir)) for path in save_dir.rglob("*") if path.is_file())


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
