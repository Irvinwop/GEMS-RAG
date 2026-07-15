#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import canonicalize_chunks

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "gfm-rag"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"
DEFAULT_DATA_DIR = ROOT / "data" / "working" / "gfmrag_data"
DEFAULT_DATA_NAME = "mutcd"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "gfmrag" / "bin" / "python"
DEFAULT_MODEL = "rmanluo/GFM-RAG-8M"
DEFAULT_MODEL_REVISION = "4da9e4655d126a783ae2b795ab73b7c7a7c3f4ac"
REQUIRED_MODULES = ["gfmrag", "hydra", "omegaconf", "pandas", "torch"]
STAGE1_FILENAMES = ["nodes.csv", "relations.csv", "edges.csv"]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
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
    marker = data_dir / data_name / "processed" / ".gfmrag-ready.json"
    graph_path = mrag_dir / "mmrag_cache_v3" / "graph.gpickle"
    chunks_path = mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    failures = _module_failures(REQUIRED_MODULES)
    source_ready = repo.exists() and graph_path.exists() and chunks_path.exists()
    environment_ready = repo.exists() and not failures
    stage1_ready = all((stage1 / name).exists() for name in STAGE1_FILENAMES)
    stage1_fingerprint = _stage1_fingerprint(stage1) if stage1_ready else None
    marker_payload = _read_json_object(marker)
    index_ready = _marker_matches(
        marker_payload,
        model=getattr(args, "model", DEFAULT_MODEL),
        model_revision=getattr(args, "model_revision", DEFAULT_MODEL_REVISION),
        stage1_fingerprint=stage1_fingerprint,
        dataset_root=data_dir / data_name,
    )
    return {
        "runnable": environment_ready and stage1_ready and index_ready,
        "environment_ready": environment_ready,
        "source_ready": source_ready,
        "stage1_ready": stage1_ready,
        "index_ready": index_ready,
        "repo": str(repo),
        "repo_found": repo.exists(),
        "mrag_dir": str(mrag_dir),
        "graph_found": graph_path.exists(),
        "chunks_found": chunks_path.exists(),
        "stage1_dir": str(stage1),
        "stage1_fingerprint": stage1_fingerprint,
        "marker": str(marker),
        "model": getattr(args, "model", DEFAULT_MODEL),
        "model_revision": getattr(args, "model_revision", DEFAULT_MODEL_REVISION),
        "missing_or_failed_imports": failures,
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
    }


def _prepare(args: argparse.Namespace) -> int:
    stage1 = args.data_dir / args.data_name / "processed" / "stage1"
    if stage1.exists() and not args.force and all((stage1 / name).exists() for name in STAGE1_FILENAMES):
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
    if not report["environment_ready"] or not report["stage1_ready"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    marker = args.data_dir / args.data_name / "processed" / ".gfmrag-ready.json"
    if report["index_ready"] and not args.force:
        print(marker.read_text(encoding="utf-8"))
        return 0
    try:
        _load_retriever(args, force_reindex=True)
    except Exception as exc:
        print(json.dumps({"error": "index_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1
    dataset_root = args.data_dir / args.data_name
    stage2_graphs = sorted((dataset_root / "processed" / "stage2").glob("*/graph.pt"))
    if not stage2_graphs:
        print(json.dumps({"error": "index_failed", "detail": "stage2 graph.pt was not created"}), file=sys.stderr)
        return 1
    payload = {
        "status": "indexed",
        "model": args.model,
        "model_revision": args.model_revision,
        "data_name": args.data_name,
        "stage1_fingerprint": _stage1_fingerprint(dataset_root / "processed" / "stage1"),
        "stage2_graphs": [str(path.relative_to(dataset_root)) for path in stage2_graphs],
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(marker, payload)
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
                    "model_revision": args.model_revision,
                    "upstream_repo": "RManLuo/gfm-rag",
                    "entity_linker": "deterministic_bm25_section_aliases",
                    "linked_entities": [
                        hit["entity"]
                        for hits in getattr(retriever.el_model, "last_result", {}).values()
                        for hit in hits
                    ],
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_retriever(args: argparse.Namespace, *, force_reindex: bool | None = None):
    sys.path.insert(0, str(args.repo))
    from gfmrag import GFMRetriever

    entity_linker = _LexicalEL()
    model_path = _resolve_model_snapshot(args.model, args.model_revision)
    retriever = GFMRetriever.from_index(
        data_dir=str(args.data_dir),
        data_name=args.data_name,
        model_path=model_path,
        ner_model=_LexicalNER(),
        el_model=entity_linker,
        force_reindex=bool(getattr(args, "force", False)) if force_reindex is None else force_reindex,
    )
    _configure_entity_aliases(retriever, entity_linker)
    return retriever


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
    direct_document_links = set()
    reverse_document_links = set()
    for source, target, data in graph.edges(data=True):
        source = str(source)
        target = str(target)
        attributes = dict(data or {})
        relation = str(attributes.pop("label", None) or attributes.pop("relation", None) or "related_to")
        relations.add(relation)
        edge_rows.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "attributes": repr(_literal_safe(attributes)),
            }
        )
        source_is_document = source.startswith("chunk:")
        target_is_document = target.startswith("chunk:")
        if not source_is_document and target_is_document:
            direct_document_links.add((source, target))
        elif source_is_document and not target_is_document:
            reverse_document_links.add((target, source))

    added_document_links = sorted(reverse_document_links - direct_document_links)
    if added_document_links:
        relation = "gfm_document_link"
        relations.add(relation)
        edge_rows.extend(
            {
                "source": entity,
                "relation": relation,
                "target": document,
                "attributes": "{}",
            }
            for entity, document in added_document_links
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
        "document_links_added": len(added_document_links),
    }


class _LexicalNER:
    def __call__(self, text: str) -> list[str]:
        sections = [
            f"section:{section.upper()}"
            for section in re.findall(r"\b(?:Section\s+)?([0-9]+[A-Z]\.[0-9]+)\b", text, flags=re.IGNORECASE)
        ]
        figures = [
            f"figure:{kind.title()} {identifier}"
            for kind, identifier in re.findall(
                r"\b(Figure|Table)\s+([0-9A-Z]+-[0-9A-Za-z-]+)\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        sign_codes = [
            f"signcode:{code}"
            for code in re.findall(
                r"\b(?:[A-Z]{1,3}\d{1,2}(?:-\d+[A-Z]?)?[A-Z]?P?|OM\d(?:-[LCR])?)\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        explicit = sections + figures + sign_codes
        return list(dict.fromkeys(explicit or [text]))


class _LexicalEL:
    def __init__(self) -> None:
        self.entities: list[str] = []
        self._entity_terms: dict[str, Counter[str]] = {}
        self._idf: dict[str, float] = {}
        self._average_length = 1.0
        self._exact: dict[str, str] = {}
        self.last_result: dict[str, list[dict[str, Any]]] = {}

    def index(self, entity_list: list[str]) -> None:
        self.entities = [entity for entity in entity_list if not entity.startswith("chunk:")]
        self.set_aliases({})

    def set_aliases(self, aliases: dict[str, list[str]]) -> None:
        self._exact = {entity.casefold(): entity for entity in self.entities}
        self._entity_terms = {}
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for entity in self.entities:
            text = " ".join([entity, *aliases.get(entity, [])])
            terms = Counter(_lexical_tokens(text))
            self._entity_terms[entity] = terms
            total_length += sum(terms.values())
            document_frequency.update(terms)
        count = max(len(self.entities), 1)
        self._average_length = max(total_length / count, 1.0)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def __call__(self, ner_entity_list: list[str], topk: int = 1) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for phrase in ner_entity_list:
            phrase_text = str(phrase)
            exact = self._exact.get(phrase_text.casefold())
            if exact is not None:
                hits = [(1.0, exact)]
            else:
                query_terms = set(_lexical_tokens(phrase_text))
                ranked = [
                    (self._bm25_score(query_terms, entity), entity)
                    for entity in self.entities
                ]
                ranked = [item for item in ranked if item[0] > 0]
                if not ranked and self.entities:
                    fallback = max(
                        (
                            difflib.SequenceMatcher(None, phrase_text.casefold(), entity.casefold()).ratio(),
                            entity,
                        )
                        for entity in self.entities
                    )
                    ranked = [fallback] if fallback[0] >= 0.5 else []
                ranked.sort(reverse=True)
                hits = ranked[: max(topk, 0)]
            if not hits:
                continue
            max_score = hits[0][0] if hits else 1.0
            result[str(phrase)] = [
                {"entity": entity, "score": score, "norm_score": score / max_score if max_score else 0.0}
                for score, entity in hits
            ]
        self.last_result = result
        return result

    def _bm25_score(self, query_terms: set[str], entity: str) -> float:
        terms = self._entity_terms.get(entity, Counter())
        length = max(sum(terms.values()), 1)
        score = 0.0
        for term in query_terms:
            frequency = terms.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + 1.5 * (0.25 + 0.75 * length / self._average_length)
            score += self._idf.get(term, 0.0) * frequency * 2.5 / denominator
        return score


_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "may",
    "must",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "who",
    "with",
}


def _lexical_tokens(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", text.casefold()):
        if raw in _LEXICAL_STOPWORDS or len(raw) < 2:
            continue
        if len(raw) > 4 and raw.endswith("ies"):
            raw = raw[:-3] + "y"
        elif len(raw) > 4 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


def _configure_entity_aliases(retriever: Any, entity_linker: _LexicalEL) -> None:
    aliases: dict[str, list[str]] = {}
    entity_names = set(entity_linker.entities)
    for name, row in retriever.node_info.iterrows():
        if row.get("type") != "document":
            attributes = row.get("attributes") or {}
            if isinstance(attributes, dict):
                aliases.setdefault(str(name), []).extend(str(value) for value in attributes.values() if value)
            continue
        attributes = row.get("attributes") or {}
        if not isinstance(attributes, dict):
            continue
        section_id = str(attributes.get("section_id") or attributes.get("section") or "").strip()
        section_entity = f"section:{section_id}"
        if section_id and section_entity in entity_names:
            aliases.setdefault(section_entity, []).extend(
                str(value)
                for value in [attributes.get("section_title"), attributes.get("text")]
                if value
            )
    entity_linker.set_aliases(aliases)


def _resolve_model_snapshot(model: str, revision: str) -> str:
    local_path = Path(model).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model,
        revision=revision or None,
        allow_patterns=["config.json", "model.pth", "README.md"],
    )


def _module_failures(module_names: Iterable[str]) -> dict[str, str]:
    failures = {}
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures[name] = repr(exc)
    return failures


def _stage1_fingerprint(stage1: Path) -> str:
    digest = hashlib.sha256()
    for filename in STAGE1_FILENAMES:
        path = stage1 / filename
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _marker_matches(
    payload: dict[str, Any] | None,
    *,
    model: str,
    model_revision: str,
    stage1_fingerprint: str | None,
    dataset_root: Path,
) -> bool:
    if not payload or stage1_fingerprint is None:
        return False
    stage2_graphs = payload.get("stage2_graphs")
    return (
        payload.get("status") == "indexed"
        and payload.get("model") == model
        and payload.get("model_revision") == model_revision
        and payload.get("stage1_fingerprint") == stage1_fingerprint
        and isinstance(stage2_graphs, list)
        and bool(stage2_graphs)
        and all((dataset_root / str(path)).is_file() for path in stage2_graphs)
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
