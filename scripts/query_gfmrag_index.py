#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gem_rags.data import canonicalize_chunks

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "gfm-rag"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260708T114057Z-3" / "MRAG"
DEFAULT_DATA_DIR = ROOT / "data" / "working" / "gfmrag_data"
DEFAULT_DATA_NAME = "mutcd"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "gfmrag" / "bin" / "python"
DEFAULT_MODEL = "rmanluo/GFM-RAG-8M"
REQUIRED_MODULES = ["gfmrag", "hydra", "omegaconf", "pandas", "torch"]

sys.path.insert(0, str(DEFAULT_REPO))


def main() -> int:
    args = _parse_args()
    reexec_code = _maybe_reexec(args.python)
    if reexec_code is not None:
        return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "index":
        return _index(args)
    if args.command == "query":
        return _query(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the official GFM-RAG graph foundation retriever on MUTCD.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-name", default=DEFAULT_DATA_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=Path(os.getenv("GFMRAG_PYTHON", str(DEFAULT_ENV_PYTHON))))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")
    index = sub.add_parser("index")
    index.add_argument("--force", action="store_true")
    query = sub.add_parser("query")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = getattr(args, "repo", DEFAULT_REPO)
    mrag_dir = getattr(args, "mrag_dir", DEFAULT_MRAG_DIR)
    data_dir = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    data_name = getattr(args, "data_name", DEFAULT_DATA_NAME)
    stage1 = data_dir / data_name / "processed" / "stage1"
    graph_path = mrag_dir / "mmrag_cache_v3" / "graph.gpickle"
    chunks_path = mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    source_ready = repo.exists() and graph_path.exists() and chunks_path.exists()
    environment_ready = repo.exists() and not missing
    index_ready = all((stage1 / name).exists() for name in ["nodes.csv", "relations.csv", "edges.csv"])
    return {
        "runnable": environment_ready and index_ready,
        "environment_ready": environment_ready,
        "source_ready": source_ready,
        "index_ready": index_ready,
        "repo": str(repo),
        "repo_found": repo.exists(),
        "mrag_dir": str(mrag_dir),
        "graph_found": graph_path.exists(),
        "chunks_found": chunks_path.exists(),
        "stage1_dir": str(stage1),
        "model": getattr(args, "model", DEFAULT_MODEL),
        "missing_or_failed_imports": {name: "not installed" for name in missing},
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
    }


def _prepare(args: argparse.Namespace) -> int:
    stage1 = args.data_dir / args.data_name / "processed" / "stage1"
    if stage1.exists() and not args.force and all((stage1 / name).exists() for name in ["nodes.csv", "relations.csv", "edges.csv"]):
        print(json.dumps({"status": "already_prepared", "stage1_dir": str(stage1)}, indent=2))
        return 0
    graph_path = args.mrag_dir / "mmrag_cache_v3" / "graph.gpickle"
    chunks_path = args.mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    if not graph_path.exists() or not chunks_path.exists():
        print(
            json.dumps({"error": "missing_source", "graph": str(graph_path), "chunks": str(chunks_path)}),
            file=sys.stderr,
        )
        return 2
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    chunks, _chunk_report = canonicalize_chunks(_read_jsonl(chunks_path))
    counts = _export_stage1(graph, chunks, stage1)
    print(json.dumps({"status": "prepared", "stage1_dir": str(stage1), **counts}, indent=2))
    return 0


def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    marker = args.data_dir / args.data_name / "processed" / ".gfmrag-ready.json"
    if marker.exists() and not args.force:
        print(marker.read_text(encoding="utf-8"))
        return 0
    try:
        _load_retriever(args)
    except Exception as exc:
        print(json.dumps({"error": "index_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1
    payload = {"status": "indexed", "model": args.model, "data_name": args.data_name}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    try:
        retriever = _load_retriever(args)
        result = retriever.retrieve(args.question, top_k=args.top_k, target_types=["document"])
    except Exception as exc:
        print(json.dumps({"error": "query_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1
    chunks = []
    for item in result.get("document", []):
        raw_id = str(item.get("id") or "")
        attributes = item.get("attributes")
        if isinstance(attributes, str):
            try:
                attributes = ast.literal_eval(attributes)
            except (SyntaxError, ValueError):
                attributes = {"text": attributes}
        attributes = dict(attributes or {})
        chunk_id = raw_id.split(":", 1)[1] if raw_id.startswith("chunk:") else raw_id
        chunks.append({**attributes, "chunk_id": chunk_id, "score": float(item.get("score", 0.0))})
    print(
        json.dumps(
            {
                "question": args.question,
                "chunks": chunks,
                "debug": {
                    "method": "gfm_rag",
                    "model": args.model,
                    "upstream_repo": "RManLuo/gfm-rag",
                    "entity_linker": "deterministic_lexical_adapter",
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_retriever(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo))
    from gfmrag import GFMRetriever

    return GFMRetriever.from_index(
        data_dir=str(args.data_dir),
        data_name=args.data_name,
        model_path=args.model,
        ner_model=_LexicalNER(),
        el_model=_LexicalEL(),
        force_reindex=bool(getattr(args, "force", False)),
    )


def _export_stage1(graph: Any, chunks: Iterable[dict[str, Any]], stage1: Path) -> dict[str, int]:
    stage1.mkdir(parents=True, exist_ok=True)
    chunk_by_node = {f"chunk:{chunk.get('chunk_id')}": dict(chunk) for chunk in chunks}
    node_names = {str(node) for node in graph.nodes}
    node_names.update(chunk_by_node)
    nodes = []
    for name in sorted(node_names):
        graph_attributes = dict(graph.nodes[name]) if graph.has_node(name) else {}
        attributes = {**graph_attributes, **chunk_by_node.get(name, {})}
        nodes.append(
            {
                "name": name,
                "type": "document" if name.startswith("chunk:") else "entity",
                "attributes": repr(_literal_safe(attributes)),
            }
        )

    edge_rows = []
    relations = set()
    for source, target, data in graph.edges(data=True):
        attributes = dict(data or {})
        relation = str(attributes.pop("label", None) or attributes.pop("relation", None) or "related_to")
        relations.add(relation)
        edge_rows.append(
            {
                "source": str(source),
                "relation": relation,
                "target": str(target),
                "attributes": repr(_literal_safe(attributes)),
            }
        )

    _write_csv(stage1 / "nodes.csv", ["name", "type", "attributes"], nodes)
    _write_csv(
        stage1 / "relations.csv",
        ["name", "attributes"],
        [{"name": relation, "attributes": "{}"} for relation in sorted(relations)],
    )
    _write_csv(stage1 / "edges.csv", ["source", "relation", "target", "attributes"], edge_rows)
    return {
        "nodes": len(nodes),
        "relations": len(relations),
        "edges": len(edge_rows),
        "documents": sum(row["type"] == "document" for row in nodes),
    }


class _LexicalNER:
    def __call__(self, text: str) -> list[str]:
        explicit = re.findall(r"\b(?:Section\s+)?([0-9]+[A-Z]\.[0-9]+)\b", text, flags=re.IGNORECASE)
        explicit.extend(re.findall(r"\b(?:Figure|Table)\s+[0-9A-Z]+-[0-9A-Za-z-]+\b", text, flags=re.IGNORECASE))
        words = [word.lower() for word in re.findall(r"[A-Za-z0-9.-]+", text) if len(word) > 2]
        phrases = explicit + words + [text]
        return list(dict.fromkeys(phrases))


class _LexicalEL:
    def __init__(self) -> None:
        self.entities: list[str] = []

    def index(self, entity_list: list[str]) -> None:
        self.entities = list(entity_list)

    def __call__(self, ner_entity_list: list[str], topk: int = 1) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for phrase in ner_entity_list:
            phrase_lower = str(phrase).lower()
            phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase_lower))
            ranked = []
            for entity in self.entities:
                entity_lower = entity.lower()
                entity_tokens = set(re.findall(r"[a-z0-9]+", entity_lower))
                overlap = len(phrase_tokens & entity_tokens) / max(len(phrase_tokens | entity_tokens), 1)
                sequence = difflib.SequenceMatcher(None, phrase_lower, entity_lower).ratio()
                ranked.append((max(overlap, sequence), entity))
            ranked.sort(reverse=True)
            hits = ranked[:topk]
            max_score = hits[0][0] if hits else 1.0
            result[str(phrase)] = [
                {"entity": entity, "score": score, "norm_score": score / max_score if max_score else 0.0}
                for score, entity in hits
            ]
        return result


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _literal_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _literal_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_literal_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
