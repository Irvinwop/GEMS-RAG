#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gem_rags.config import DEFAULT_MRAG_DIR
from gem_rags.data import load_chunks
from gem_rags.retrieval import QdrantHashVectorRetriever
from gem_rags.types import QAItem


def main() -> int:
    args = _parse_args()
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["runnable"] else 2
    if args.command == "search":
        retriever = QdrantHashVectorRetriever(
            name="qdrant_hash_vector_tool",
            chunks=load_chunks(args.mrag_dir),
            top_k=args.top_k,
            dims=args.dims,
            qdrant_path=args.path,
        )
        item = QAItem(
            qa_id="tool_query",
            question=args.question,
            question_type=None,
            expected_refusal=False,
            gold_answer={},
            references=[],
        )
        result = retriever.retrieve(item)
        print(
            json.dumps(
                {
                    "question": args.question,
                    "evidence": [
                        {
                            "evidence_id": ev.evidence_id,
                            "score": ev.score,
                            "kind": ev.kind,
                            "metadata": ev.metadata,
                            "text": ev.text,
                        }
                        for ev in result.evidence
                    ],
                    "debug": result.debug,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "open":
        chunks = {str(chunk.get("chunk_id")): chunk for chunk in load_chunks(args.mrag_dir)}
        chunk = chunks.get(args.chunk_id)
        if not chunk:
            print(json.dumps({"error": "chunk_not_found", "chunk_id": args.chunk_id}), file=sys.stderr)
            return 2
        print(json.dumps(chunk, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


def _dependency_report(args: argparse.Namespace) -> dict[str, object]:
    qdrant_installed = importlib.util.find_spec("qdrant_client") is not None
    chunks_path = args.mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    mrag_dir_found = args.mrag_dir.exists()
    chunks_found = chunks_path.exists()
    environment_ready = qdrant_installed and mrag_dir_found and chunks_found
    return {
        "runnable": environment_ready,
        "environment_ready": environment_ready,
        "qdrant_client_installed": qdrant_installed,
        "mrag_dir": str(args.mrag_dir),
        "mrag_dir_found": mrag_dir_found,
        "chunks": str(chunks_path),
        "chunks_found": chunks_found,
        "index": str(args.path),
        "index_ready": args.path.exists(),
        "notes": "Search builds the embedded Qdrant hash-vector index lazily when the index path is missing or stale.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search/open the local Qdrant vector DB baseline.")
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument("--path", type=Path, default=Path("data/working/qdrant_hash_vector"))
    parser.add_argument("--dims", type=int, default=512)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the local vector DB search can run.")

    search = sub.add_parser("search", help="Search the Qdrant-backed vector baseline.")
    search.add_argument("--question", required=True)
    search.add_argument("--top-k", type=int, default=6)

    open_hit = sub.add_parser("open", help="Open a chunk by chunk_id.")
    open_hit.add_argument("--chunk-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
