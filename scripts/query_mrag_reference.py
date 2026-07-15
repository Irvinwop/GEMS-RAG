#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_chunks
from gems_rag.mrag_reference_modes import REFERENCE_MODES, retrieve_reference_mode

DEFAULT_REPO = ROOT / "external" / "MRAG_stp2"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "mrag-reference" / "bin" / "python"
REQUIRED_MODULES = ["qdrant_client", "numpy", "torch"]
GRAPH_MODULES = ["networkx"]
TEXT_RETRIEVAL_MODULES = ["FlagEmbedding", "sentence_transformers"]
RERANK_MODULES = ["mxbai_rerank", "sentence_transformers"]
VISUAL_MODULES = ["colpali_engine"]
RERANK_MODES = {"full", "no_graph", "no_visual", "no_rule", "no_hierarchy"}
VISUAL_MODES = {"multimodal", "full", "no_graph", "no_rule", "no_hierarchy"}
GRAPH_MODES = {"full", "no_visual", "no_rule", "no_hierarchy"}


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

    check = sub.add_parser("check", help="Report whether the local environment can run an MRAG retrieval mode.")
    check.add_argument("--mode", choices=REFERENCE_MODES, default="full")

    retrieve = sub.add_parser("retrieve", help="Run MRAG retrieval and print JSON evidence.")
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--top-k", type=int, default=6)
    retrieve.add_argument("--mode", choices=REFERENCE_MODES, default="full")
    retrieve.add_argument("--with-image", action="store_true", help="Deprecated compatibility flag; visual modes load the encoder automatically.")
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    mode = getattr(args, "mode", "full")
    missing_required = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    text_ok = any(importlib.util.find_spec(name) is not None for name in TEXT_RETRIEVAL_MODULES)
    rerank_ok = any(importlib.util.find_spec(name) is not None for name in RERANK_MODULES)
    visual_ok = all(importlib.util.find_spec(name) is not None for name in VISUAL_MODULES)
    graph_ok = all(importlib.util.find_spec(name) is not None for name in GRAPH_MODULES)
    missing_groups = []
    if not text_ok:
        missing_groups.append({"group": "text_embedding", "one_of": TEXT_RETRIEVAL_MODULES})
    if mode in RERANK_MODES and not rerank_ok:
        missing_groups.append({"group": "reranking", "one_of": RERANK_MODULES})
    if mode in VISUAL_MODES and not visual_ok:
        missing_groups.append({"group": "visual_embedding", "all_of": VISUAL_MODULES})
    if mode in GRAPH_MODES and not graph_ok:
        missing_groups.append({"group": "graph", "all_of": GRAPH_MODULES})
    return {
        "runnable": (
            not missing_required
            and text_ok
            and (mode not in RERANK_MODES or rerank_ok)
            and (mode not in VISUAL_MODES or visual_ok)
            and (mode not in GRAPH_MODES or graph_ok)
        ),
        "mode": mode,
        "components": {
            "reranker_required": mode in RERANK_MODES,
            "visual_required": mode in VISUAL_MODES,
            "graph_required": mode in GRAPH_MODES,
        },
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_required_modules": missing_required,
        "missing_alternative_groups": missing_groups,
        "notes": "Install external/MRAG_stp2/requirements.txt into an isolated environment for the selected reference mode.",
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
        from mrag.config import CFG
    except Exception as exc:
        print(json.dumps({"error": "import_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2

    try:
        pipeline = _init_mode_pipeline(CFG, args.mode)
        result = retrieve_reference_mode(
            pipeline,
            args.question,
            mode=args.mode,
            top_k=args.top_k,
            chunks=load_chunks(args.mrag_dir),
        )
    except Exception as exc:
        print(json.dumps({"error": "retrieve_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "question": args.question,
                "chunks": result["chunks"],
                "figures": result["figures"],
                "pages": result["pages"],
                "debug": result["debug"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _init_mode_pipeline(config: Any, mode: str) -> SimpleNamespace:
    from mrag.embeddings import ImageEmbedder, Reranker, TextEmbedder
    from mrag.vector_store import VectorStore

    pipeline = SimpleNamespace()
    pipeline.store = VectorStore(config.qdrant_dir)
    pipeline.text = TextEmbedder(config.bge_m3_model).load()
    pipeline.image = None
    pipeline.kg = None
    pipeline.rerank = None
    if mode in GRAPH_MODES:
        from mrag.kg import KG, read as kg_read

        pipeline.kg = KG(kg_read(config.graph_pickle))
    if mode in RERANK_MODES:
        pipeline.rerank = Reranker(config.reranker_model).load()
    if mode in VISUAL_MODES:
        pipeline.image = ImageEmbedder(config.colqwen_model).load()
    return pipeline


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
