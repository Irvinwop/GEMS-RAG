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
DEFAULT_REPO = ROOT / "external" / "MRAG_stp2"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260708T114057Z-3" / "MRAG"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "mrag-reference" / "bin" / "python"
REQUIRED_MODULES = ["qdrant_client", "networkx", "numpy", "torch"]
TEXT_RETRIEVAL_MODULES = ["FlagEmbedding", "sentence_transformers"]
RERANK_MODULES = ["mxbai_rerank", "sentence_transformers"]


def main() -> int:
    args = _parse_args()
    reexec_code = _maybe_reexec(args.python)
    if reexec_code is not None:
        return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "retrieve":
        return _retrieve(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the cloned hannanazad/MRAG_stp2 retrieval stack.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.getenv("MRAG_REFERENCE_PYTHON", str(DEFAULT_ENV_PYTHON))),
        help="Optional isolated Python with MRAG reference dependencies. Defaults to data/working/venvs/mrag-reference/bin/python when present.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the local environment can run MRAG retrieval.")

    retrieve = sub.add_parser("retrieve", help="Run MRAG retrieval and print JSON evidence.")
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--top-k", type=int, default=6)
    retrieve.add_argument("--with-image", action="store_true", help="Load ColQwen/ColPali visual retrieval too.")
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    missing_required = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    text_ok = any(importlib.util.find_spec(name) is not None for name in TEXT_RETRIEVAL_MODULES)
    rerank_ok = any(importlib.util.find_spec(name) is not None for name in RERANK_MODULES)
    missing_groups = []
    if not text_ok:
        missing_groups.append({"group": "text_embedding", "one_of": TEXT_RETRIEVAL_MODULES})
    if not rerank_ok:
        missing_groups.append({"group": "reranking", "one_of": RERANK_MODULES})
    return {
        "runnable": not missing_required and text_ok and rerank_ok,
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_required_modules": missing_required,
        "missing_alternative_groups": missing_groups,
        "notes": "Install external/MRAG_stp2/requirements.txt into an isolated environment for the full dense+sparse MRAG baseline.",
    }


def _retrieve(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not args.repo.exists():
        print(json.dumps({"error": "repo_not_found", "repo": str(args.repo)}), file=sys.stderr)
        return 2
    if not args.mrag_dir.exists():
        print(json.dumps({"error": "mrag_dir_not_found", "mrag_dir": str(args.mrag_dir)}), file=sys.stderr)
        return 2

    os.environ["MRAG_BASE_DIR"] = str(args.mrag_dir)
    sys.path.insert(0, str(args.repo))
    try:
        from mrag.ask import init_pipeline
        from mrag.config import CFG
    except Exception as exc:
        print(json.dumps({"error": "import_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2

    CFG.top_k_after_rerank = args.top_k
    try:
        pipeline = init_pipeline(load_image_embedder=args.with_image, load_vlm=False)
        result = pipeline.retriever.retrieve(args.question)
    except Exception as exc:
        print(json.dumps({"error": "retrieve_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "question": args.question,
                "chunks": result.chunks,
                "figures": result.figures,
                "pages": result.pages,
                "debug": result.debug,
            },
            ensure_ascii=False,
        )
    )
    return 0


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
