from __future__ import annotations

import collections
import hashlib
import json
import math
import pickle
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .config import RetrieverConfig
from .data import load_chunks, load_figures
from .types import Evidence, QAItem, RetrievalResult

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")
SECTION_RE = re.compile(r"\b([1-9][A-Z]\.[0-9]{2})\b", re.IGNORECASE)
FIGURE_RE = re.compile(r"\b(?:Figure|Table)\s+([1-9][A-Z]-[0-9]+[A-Z]?)\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def evidence_text_from_chunk(chunk: dict, max_chars: int = 1600) -> str:
    header = (
        f"Section {chunk.get('section_id')} {chunk.get('content_type')} "
        f"{chunk.get('ordinal')} - {chunk.get('section_title')} "
        f"(p.{chunk.get('page_printed')}, {chunk.get('part')})"
    )
    body = (chunk.get("text") or "").strip()
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    return f"{header}\n{body}"


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, item: QAItem) -> RetrievalResult:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(self, name: str, chunks: list[dict], top_k: int = 6, *, graph_boost: bool = False) -> None:
        self.name = name
        self.chunks = chunks
        self.top_k = top_k
        self.graph_boost = graph_boost
        self.doc_tokens = [tokenize(_chunk_search_text(chunk)) for chunk in chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: collections.Counter[str] = collections.Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = max(len(self.doc_tokens), 1)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def retrieve(self, item: QAItem) -> RetrievalResult:
        query_tokens = tokenize(item.question)
        explicit_sections = {m.upper() for m in SECTION_RE.findall(item.question)}
        explicit_figures = {m.upper() for m in FIGURE_RE.findall(item.question)}
        scores: list[tuple[float, int]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            score = self._score(query_tokens, tokens, self.doc_len[idx])
            chunk = self.chunks[idx]
            section_id = str(chunk.get("section_id", "")).upper()
            if self.graph_boost and section_id in explicit_sections:
                score += 5.0
            if self.graph_boost and explicit_figures:
                refs = {str(ref).upper() for ref in chunk.get("figure_refs", []) + chunk.get("table_refs", [])}
                if refs & explicit_figures:
                    score += 2.0
            if score > 0:
                scores.append((score, idx))
        scores.sort(reverse=True)
        evidence = [
            _chunk_to_evidence(self.chunks[idx], score)
            for score, idx in scores[: self.top_k]
        ]
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                "query_tokens": query_tokens,
                "explicit_sections": sorted(explicit_sections),
                "explicit_figures": sorted(explicit_figures),
            },
        )

    def _score(self, query_tokens: list[str], doc_tokens: list[str], doc_len: int) -> float:
        counts = collections.Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (k1 + 1)) / denom
        return score


class HashVectorRetriever(Retriever):
    """Dependency-free vector-search baseline using hashed token counts."""

    def __init__(self, name: str, chunks: list[dict], top_k: int = 6, dims: int = 2048) -> None:
        self.name = name
        self.chunks = chunks
        self.top_k = top_k
        self.dims = dims
        self.doc_vectors = [_hash_vector(tokenize(_chunk_search_text(chunk)), dims) for chunk in chunks]
        self.doc_norms = [_vector_norm(vec) for vec in self.doc_vectors]

    def retrieve(self, item: QAItem) -> RetrievalResult:
        query_tokens = tokenize(item.question)
        query_vec = _hash_vector(query_tokens, self.dims)
        query_norm = _vector_norm(query_vec)
        scores: list[tuple[float, int]] = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            denom = query_norm * self.doc_norms[idx]
            if denom <= 0:
                continue
            score = _dot(query_vec, doc_vec) / denom
            if score > 0:
                scores.append((score, idx))
        scores.sort(reverse=True)
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=[_chunk_to_evidence(self.chunks[idx], score) for score, idx in scores[: self.top_k]],
            debug={"vector_dims": self.dims, "query_tokens": query_tokens, "backend": "local_hashed_counts"},
        )


class QdrantHashVectorRetriever(Retriever):
    """Embedded-Qdrant vector DB baseline with deterministic local embeddings."""

    def __init__(
        self,
        name: str,
        chunks: list[dict],
        top_k: int = 6,
        dims: int = 512,
        qdrant_path: Path = Path("data/working/qdrant_hash_vector"),
        collection: str | None = None,
    ) -> None:
        self.name = name
        self.chunks = chunks
        self.top_k = top_k
        self.dims = dims
        self.qdrant_path = qdrant_path
        self.collection = collection or f"mutcd_hash_vectors_d{dims}"
        self._client = None

    def retrieve(self, item: QAItem) -> RetrievalResult:
        client = self._ensure_index()
        query_tokens = tokenize(item.question)
        vector = _dense_hash_vector(query_tokens, self.dims)
        try:
            response = client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=self.top_k,
                with_payload=True,
            )
        except AttributeError:
            hits = client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=self.top_k,
                with_payload=True,
            )
        else:
            hits = response.points
        evidence = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            score = float(getattr(hit, "score", 0.0))
            evidence.append(_chunk_to_evidence(payload, score))
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                "backend": "embedded_qdrant_hash_vector",
                "qdrant_path": str(self.qdrant_path),
                "collection": self.collection,
                "vector_dims": self.dims,
                "query_tokens": query_tokens,
            },
        )

    def _ensure_index(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as exc:
            raise RuntimeError(f"qdrant-client is required for {self.name}: {exc}") from exc
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(self.qdrant_path))
        needs_build = True
        try:
            if client.collection_exists(self.collection):
                count = client.count(collection_name=self.collection, exact=True).count
                needs_build = count != len(self.chunks)
        except Exception:
            needs_build = True
        if needs_build:
            if client.collection_exists(self.collection):
                client.delete_collection(self.collection)
            client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dims, distance=qm.Distance.COSINE),
            )
            self._upsert_chunks(client)
        self._client = client
        return client

    def _upsert_chunks(self, client, batch_size: int = 128) -> None:
        from qdrant_client.http import models as qm

        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start : start + batch_size]
            points = []
            for chunk in batch:
                chunk_id = str(chunk.get("chunk_id"))
                points.append(
                    qm.PointStruct(
                        id=_stable_point_id(chunk_id),
                        vector=_dense_hash_vector(tokenize(_chunk_search_text(chunk)), self.dims),
                        payload=chunk,
                    )
                )
            client.upsert(collection_name=self.collection, points=points, wait=True)


class OracleRetriever(Retriever):
    def __init__(self, name: str, chunks: list[dict], figures: list[dict], top_k: int = 12) -> None:
        self.name = name
        self.top_k = top_k
        self.chunk_index = {
            (str(c.get("section_id")), str(c.get("content_type")), str(c.get("ordinal")).zfill(2)): c
            for c in chunks
        }
        self.section_index: dict[str, list[dict]] = collections.defaultdict(list)
        for chunk in chunks:
            self.section_index[str(chunk.get("section_id"))].append(chunk)
        self.figure_index = {str(f.get("figure_id")): f for f in figures}

    def retrieve(self, item: QAItem) -> RetrievalResult:
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for ref in item.references:
            section_id = str(ref.get("section_id"))
            content_type = str(ref.get("content_type", ""))
            ordinal = str(ref.get("ordinal", "")).zfill(2)
            candidates = []
            if (section_id, content_type, ordinal) in self.chunk_index:
                candidates = [self.chunk_index[(section_id, content_type, ordinal)]]
            else:
                candidates = self.section_index.get(section_id, [])
            for chunk in candidates:
                ev = _chunk_to_evidence(chunk, 100.0)
                if ev.evidence_id not in seen:
                    evidence.append(ev)
                    seen.add(ev.evidence_id)
                if len(evidence) >= self.top_k:
                    break
        for fig in item.gold_figures:
            fig_id = fig if isinstance(fig, str) else fig.get("figure_id")
            record = self.figure_index.get(str(fig_id))
            if record:
                evidence.append(_figure_to_evidence(record, 100.0))
        return RetrievalResult(adapter=self.name, query=item.question, evidence=evidence[: self.top_k], debug={"oracle": True})


class MGraphRetriever(Retriever):
    """Use the repaired NetworkX graph for section-neighborhood expansion."""

    def __init__(self, name: str, mrag_dir: Path, chunks: list[dict], top_k: int = 8) -> None:
        self.name = name
        self.top_k = top_k
        self.base = BM25Retriever(name + ":bm25_seed", chunks, top_k=max(top_k, 6), graph_boost=True)
        self.chunk_by_id = {str(c.get("chunk_id")): c for c in chunks}
        with (mrag_dir / "mmrag_cache_v3" / "graph.gpickle").open("rb") as handle:
            self.graph = pickle.load(handle)

    def retrieve(self, item: QAItem) -> RetrievalResult:
        seed = self.base.retrieve(item)
        evidence = list(seed.evidence)
        seen = {ev.evidence_id for ev in evidence}
        undirected = self.graph.to_undirected(as_view=True)
        for ev in seed.evidence[:3]:
            chunk_id = ev.metadata.get("chunk_id")
            node = f"chunk:{chunk_id}"
            if not self.graph.has_node(node):
                continue
            frontier = {node}
            reached = {node}
            for _ in range(2):
                nxt = set()
                for current in frontier:
                    nxt.update(undirected.neighbors(current))
                nxt -= reached
                reached |= nxt
                frontier = nxt
            for neighbor in sorted(reached):
                if not str(neighbor).startswith("chunk:"):
                    continue
                neighbor_id = str(neighbor).split(":", 1)[1]
                if neighbor_id == chunk_id:
                    continue
                chunk = self.chunk_by_id.get(neighbor_id)
                if not chunk:
                    continue
                expanded = _chunk_to_evidence(chunk, ev.score * 0.8)
                if expanded.evidence_id not in seen:
                    evidence.append(expanded)
                    seen.add(expanded.evidence_id)
                if len(evidence) >= self.top_k:
                    break
            if len(evidence) >= self.top_k:
                break
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence[: self.top_k],
            debug={**seed.debug, "graph_expanded": True},
        )


class ExternalRagPlaceholder(Retriever):
    """Explicit adapter slot for cloned external implementations.

    This keeps experiment configs honest: an external RAG baseline is visible in
    the matrix, but running it requires a concrete indexing/query adapter.
    """

    def __init__(self, name: str, implementation_path: str) -> None:
        self.name = name
        self.implementation_path = implementation_path

    def retrieve(self, item: QAItem) -> RetrievalResult:
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=[],
            debug={
                "status": "not_indexed",
                "implementation_path": self.implementation_path,
                "message": "External RAG implementation is cloned but no indexed adapter has been built yet.",
            },
        )


class ExternalCommandRetriever(Retriever):
    """Run a preexisting RAG implementation through a command template."""

    def __init__(self, name: str, command: list[str], mrag_dir: Path, timeout_s: int = 300, top_k: int = 6) -> None:
        self.name = name
        self.command = command
        self.mrag_dir = mrag_dir
        self.timeout_s = timeout_s
        self.top_k = top_k

    def retrieve(self, item: QAItem) -> RetrievalResult:
        values = {
            "question": item.question,
            "qa_id": item.qa_id,
            "mrag_dir": str(self.mrag_dir),
        }
        cmd = [part.format(**values) for part in self.command]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except Exception as exc:
            error = f"external command failed before completion: {exc!r}"
            return RetrievalResult(
                adapter=self.name,
                query=item.question,
                evidence=[],
                debug={"command": cmd, "error": error},
                error=error,
            )
        text = completed.stdout.strip()
        evidence = _external_stdout_to_evidence(
            self.name,
            item.qa_id,
            text,
            cmd,
            completed.returncode,
            completed.stderr,
            self.top_k,
        )
        error = None if completed.returncode == 0 else f"external command exited with return code {completed.returncode}"
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={"command": cmd, "returncode": completed.returncode, "stderr": completed.stderr[-4000:], "error": error},
            error=error,
        )


class SelfRagPolicyRetriever(Retriever):
    """Self-RAG-style retrieval-control policy over an existing retriever.

    This mirrors the ablation modes exposed by the Self-RAG implementation:
    always retrieve, never retrieve, or adaptively retrieve based on a cheap
    retrieval-necessity score. The generator model is still supplied by the
    main harness.
    """

    def __init__(self, name: str, base: Retriever, mode: str = "adaptive_retrieval", threshold: float = 0.2) -> None:
        self.name = name
        self.base = base
        self.mode = mode
        self.threshold = threshold

    def retrieve(self, item: QAItem) -> RetrievalResult:
        necessity = _retrieval_necessity(item.question)
        do_retrieve = self.mode == "always_retrieve" or (
            self.mode == "adaptive_retrieval" and necessity >= self.threshold
        )
        if self.mode == "no_retrieval":
            do_retrieve = False
        if not do_retrieve:
            return RetrievalResult(
                adapter=self.name,
                query=item.question,
                evidence=[],
                debug={
                    "policy": "self_rag",
                    "mode": self.mode,
                    "retrieval_necessity": necessity,
                    "threshold": self.threshold,
                    "decision": "no_retrieval",
                    "base_adapter": self.base.name,
                },
            )
        result = self.base.retrieve(item)
        return RetrievalResult(
            adapter=self.name,
            query=result.query,
            evidence=result.evidence,
            debug={
                **result.debug,
                "policy": "self_rag",
                "mode": self.mode,
                "retrieval_necessity": necessity,
                "threshold": self.threshold,
                "decision": "retrieve",
                "base_adapter": result.adapter,
            },
        )


class CragPolicyRetriever(Retriever):
    """CRAG-style corrective retrieval policy over local retrievers."""

    def __init__(
        self,
        name: str,
        primary: Retriever,
        fallback: Retriever,
        accept_threshold: float = 0.45,
        reject_threshold: float = 0.18,
    ) -> None:
        self.name = name
        self.primary = primary
        self.fallback = fallback
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

    def retrieve(self, item: QAItem) -> RetrievalResult:
        primary_result = self.primary.retrieve(item)
        quality = _retrieval_quality(item.question, primary_result.evidence)
        if quality >= self.accept_threshold:
            action = "accept"
            evidence = primary_result.evidence
            debug_extra = {}
        elif quality <= self.reject_threshold:
            fallback_result = self.fallback.retrieve(item)
            action = "fallback"
            evidence = fallback_result.evidence
            debug_extra = {"fallback_adapter": fallback_result.adapter, "fallback_debug": fallback_result.debug}
        else:
            fallback_result = self.fallback.retrieve(item)
            action = "refine_merge"
            evidence = _dedupe_evidence(primary_result.evidence + fallback_result.evidence)
            debug_extra = {"fallback_adapter": fallback_result.adapter, "fallback_debug": fallback_result.debug}
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                **primary_result.debug,
                **debug_extra,
                "policy": "crag",
                "primary_adapter": primary_result.adapter,
                "retrieval_quality": quality,
                "accept_threshold": self.accept_threshold,
                "reject_threshold": self.reject_threshold,
                "action": action,
            },
        )


def build_retriever(config: RetrieverConfig, mrag_dir: Path) -> Retriever:
    chunks = load_chunks(mrag_dir)
    if config.kind == "bm25":
        return BM25Retriever(config.name, chunks, config.top_k, graph_boost=bool(config.options.get("graph_boost", False)))
    if config.kind == "hash_vector":
        return HashVectorRetriever(config.name, chunks, config.top_k, dims=int(config.options.get("dims", 2048)))
    if config.kind == "qdrant_hash_vector":
        return QdrantHashVectorRetriever(
            config.name,
            chunks,
            config.top_k,
            dims=int(config.options.get("dims", 512)),
            qdrant_path=Path(str(config.options.get("path", "data/working/qdrant_hash_vector"))),
            collection=config.options.get("collection"),
        )
    if config.kind == "bm25_graph":
        return MGraphRetriever(config.name, mrag_dir, chunks, config.top_k)
    if config.kind == "oracle":
        return OracleRetriever(config.name, chunks, load_figures(mrag_dir), config.top_k)
    if config.kind == "external_placeholder":
        return ExternalRagPlaceholder(config.name, str(config.options.get("path", "")))
    if config.kind == "external_command":
        command = config.options.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not command:
            raise ValueError(f"external_command retriever {config.name!r} requires options.command")
        return ExternalCommandRetriever(
            config.name,
            list(command),
            mrag_dir,
            timeout_s=int(config.options.get("timeout_s", 300)),
            top_k=config.top_k,
        )
    if config.kind == "self_rag_policy":
        base = _build_policy_base(config, mrag_dir, chunks, "base", default_kind="bm25_graph", default_top_k=config.top_k)
        return SelfRagPolicyRetriever(
            config.name,
            base,
            mode=str(config.options.get("mode", "adaptive_retrieval")),
            threshold=float(config.options.get("threshold", 0.2)),
        )
    if config.kind == "crag_policy":
        primary = _build_policy_base(config, mrag_dir, chunks, "primary", default_kind="bm25", default_top_k=max(config.top_k, 8))
        fallback = _build_policy_base(config, mrag_dir, chunks, "fallback", default_kind="bm25_graph", default_top_k=config.top_k)
        return CragPolicyRetriever(
            config.name,
            primary,
            fallback,
            accept_threshold=float(config.options.get("accept_threshold", 0.45)),
            reject_threshold=float(config.options.get("reject_threshold", 0.18)),
        )
    raise ValueError(f"unknown retriever kind: {config.kind}")


def _chunk_search_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(key) or "")
        for key in ["part", "chapter", "section_id", "section_title", "content_type", "text"]
    )


def _hash_vector(tokens: Iterable[str], dims: int) -> dict[int, float]:
    counts: collections.Counter[int] = collections.Counter()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        counts[int.from_bytes(digest, "big") % dims] += 1.0
    return dict(counts)


def _dense_hash_vector(tokens: Iterable[str], dims: int) -> list[float]:
    vector = [0.0] * dims
    for idx, value in _hash_vector(tokens, dims).items():
        vector[idx] = value
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _vector_norm(vector: dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _dot(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(idx, 0.0) for idx, value in left.items())


def _build_policy_base(
    config: RetrieverConfig,
    mrag_dir: Path,
    chunks: list[dict],
    prefix: str,
    *,
    default_kind: str,
    default_top_k: int,
) -> Retriever:
    kind = str(config.options.get(f"{prefix}_kind", default_kind))
    top_k = int(config.options.get(f"{prefix}_top_k", default_top_k))
    name = f"{config.name}:{prefix}_{kind}"
    options = dict(config.options.get(f"{prefix}_options", {}))
    if kind == "bm25":
        return BM25Retriever(name, chunks, top_k, graph_boost=bool(options.get("graph_boost", False)))
    if kind == "hash_vector":
        return HashVectorRetriever(name, chunks, top_k, dims=int(options.get("dims", 2048)))
    if kind == "bm25_graph":
        return MGraphRetriever(name, mrag_dir, chunks, top_k)
    raise ValueError(f"{config.kind} {config.name!r} does not support {prefix}_kind={kind!r}")


def _retrieval_necessity(question: str) -> float:
    tokens = tokenize(question)
    if not tokens:
        return 0.0
    score = 0.15
    if SECTION_RE.search(question) or FIGURE_RE.search(question):
        score += 0.45
    if any(term in tokens for term in ["mutcd", "section", "standard", "guidance", "option", "support", "shall", "should", "may"]):
        score += 0.3
    if any(term in tokens for term in ["what", "when", "where", "how", "required", "minimum", "maximum", "prohibit", "allow"]):
        score += 0.2
    if len(tokens) >= 10:
        score += 0.1
    return round(min(score, 1.0), 4)


def _retrieval_quality(question: str, evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    query_terms = set(tokenize(question))
    evidence_terms = set(tokenize(" ".join(ev.text[:1200] for ev in evidence[:5])))
    lexical = len(query_terms & evidence_terms) / max(len(query_terms), 1)
    top_score = max((ev.score for ev in evidence), default=0.0)
    score_component = min(top_score / 10.0, 1.0)
    explicit_sections = {m.upper() for m in SECTION_RE.findall(question)}
    retrieved_sections = {str(ev.metadata.get("section_id", "")).upper() for ev in evidence}
    explicit_bonus = 0.2 if explicit_sections and explicit_sections & retrieved_sections else 0.0
    return round(min(0.55 * lexical + 0.35 * score_component + explicit_bonus, 1.0), 4)


def _stable_point_id(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) >> 1


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    out = []
    seen = set()
    for ev in evidence:
        if ev.evidence_id in seen:
            continue
        out.append(ev)
        seen.add(ev.evidence_id)
    return out


def _external_stdout_to_evidence(
    adapter: str,
    qa_id: str,
    stdout: str,
    command: list[str],
    returncode: int,
    stderr: str,
    top_k: int,
) -> list[Evidence]:
    if not stdout:
        return []
    metadata = {"command": command, "stderr": stderr[-4000:], "returncode": returncode}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return [
            Evidence(
                evidence_id=f"{adapter}:{qa_id}",
                kind="tool_trace",
                text=stdout,
                metadata=metadata,
                score=1.0 if returncode == 0 else 0.0,
            )
        ]

    evidence: list[Evidence] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("chunks"), list):
        for idx, chunk in enumerate(parsed["chunks"][:top_k], 1):
            if isinstance(chunk, dict):
                evidence.append(_external_chunk_to_evidence(adapter, qa_id, idx, chunk, metadata))
    if isinstance(parsed, dict) and isinstance(parsed.get("figures"), list):
        for idx, figure in enumerate(parsed["figures"][:top_k], 1):
            if isinstance(figure, dict):
                evidence.append(_external_figure_to_evidence(adapter, qa_id, idx, figure, metadata))
    if isinstance(parsed, dict) and isinstance(parsed.get("pages"), list):
        for idx, page in enumerate(parsed["pages"][:top_k], 1):
            evidence.append(_external_page_to_evidence(adapter, qa_id, idx, page, metadata))
    if isinstance(parsed, dict) and isinstance(parsed.get("contexts"), list):
        for idx, context in enumerate(parsed["contexts"][:top_k], 1):
            if isinstance(context, dict):
                evidence.append(_external_context_to_evidence(adapter, qa_id, idx, context, metadata))
    if evidence:
        return evidence
    text = parsed.get("result") or parsed.get("answer") if isinstance(parsed, dict) else None
    return [
        Evidence(
            evidence_id=f"{adapter}:{qa_id}",
            kind="tool_trace",
            text=str(text if text is not None else stdout),
            metadata={**metadata, "parsed_json": isinstance(parsed, dict)},
            score=1.0 if returncode == 0 else 0.0,
        )
    ]


def _external_chunk_to_evidence(adapter: str, qa_id: str, idx: int, chunk: dict[str, Any], base_metadata: dict[str, Any]) -> Evidence:
    chunk_id = str(chunk.get("chunk_id") or chunk.get("doc_id") or chunk.get("id") or f"{adapter}:{qa_id}:chunk:{idx}")
    metadata = {
        **base_metadata,
        "chunk_id": chunk_id,
        "section_id": chunk.get("section_id"),
        "section_title": chunk.get("section_title"),
        "content_type": chunk.get("content_type"),
        "ordinal": chunk.get("ordinal"),
        "page_printed": chunk.get("page_printed"),
        "part": chunk.get("part"),
        "source_adapter": adapter,
    }
    text = evidence_text_from_chunk({**chunk, "chunk_id": chunk_id}) if chunk.get("section_id") else str(chunk.get("text") or chunk)
    return Evidence(evidence_id=chunk_id, kind="chunk", text=text, metadata=metadata, score=float(chunk.get("score") or 1.0))


def _external_figure_to_evidence(
    adapter: str,
    qa_id: str,
    idx: int,
    figure: dict[str, Any],
    base_metadata: dict[str, Any],
) -> Evidence:
    figure_id = str(figure.get("figure_id") or figure.get("id") or figure.get("name") or f"{adapter}:{qa_id}:figure:{idx}")
    metadata = {
        **base_metadata,
        **dict(figure.get("metadata") or {}),
        **figure,
        "figure_id": figure_id,
        "source_adapter": adapter,
    }
    text = str(
        figure.get("text")
        or figure.get("caption")
        or figure.get("title")
        or f"{figure_id} visual evidence"
    )
    return Evidence(
        evidence_id=figure_id,
        kind="figure",
        text=text,
        metadata=metadata,
        score=float(figure.get("score") or 1.0),
    )


def _external_page_to_evidence(
    adapter: str,
    qa_id: str,
    idx: int,
    page: Any,
    base_metadata: dict[str, Any],
) -> Evidence:
    if isinstance(page, dict):
        page_id = str(
            page.get("page_id")
            or page.get("id")
            or page.get("name")
            or page.get("page_pdf")
            or page.get("page_printed")
            or f"{adapter}:{qa_id}:page:{idx}"
        )
        metadata = {
            **base_metadata,
            **dict(page.get("metadata") or {}),
            **page,
            "source_adapter": adapter,
        }
        text = str(page.get("text") or page.get("caption") or f"MUTCD page {page_id} visual evidence")
        score = float(page.get("score") or 1.0)
    else:
        page_id = str(page)
        metadata = {**base_metadata, "page": page, "source_adapter": adapter}
        text = f"MUTCD page {page_id} visual evidence"
        score = 1.0
    evidence_id = page_id if page_id.startswith("page:") else f"page:{page_id}"
    return Evidence(evidence_id=evidence_id, kind="page", text=text, metadata=metadata, score=score)


def _external_context_to_evidence(adapter: str, qa_id: str, idx: int, context: dict[str, Any], base_metadata: dict[str, Any]) -> Evidence:
    name = str(context.get("name") or f"{adapter}:{qa_id}:context:{idx}")
    kind = str(context.get("kind") or "tool_trace")
    if kind not in {"chunk", "figure", "page", "tool_trace"}:
        kind = "tool_trace"
    extra_metadata = dict(context.get("metadata") or {})
    for key in ["image_path", "doc_id", "page_pdf", "page_printed", "figure_id"]:
        if key in context:
            extra_metadata[key] = context.get(key)
    return Evidence(
        evidence_id=name,
        kind=kind,
        text=str(context.get("text") or context),
        metadata={**base_metadata, **extra_metadata, "source_adapter": adapter},
        score=float(context.get("score") or 1.0),
    )


def _chunk_to_evidence(chunk: dict, score: float) -> Evidence:
    chunk_id = str(chunk.get("chunk_id"))
    return Evidence(
        evidence_id=chunk_id,
        kind="chunk",
        text=evidence_text_from_chunk(chunk),
        metadata={
            "chunk_id": chunk_id,
            "section_id": chunk.get("section_id"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "ordinal": chunk.get("ordinal"),
            "page_printed": chunk.get("page_printed"),
            "part": chunk.get("part"),
            "figure_refs": chunk.get("figure_refs", []),
            "table_refs": chunk.get("table_refs", []),
        },
        score=float(score),
    )


def _figure_to_evidence(figure: dict, score: float) -> Evidence:
    figure_id = str(figure.get("figure_id"))
    return Evidence(
        evidence_id=figure_id,
        kind="figure",
        text=f"{figure_id}: {figure.get('caption') or figure.get('title') or ''}",
        metadata=figure,
        score=float(score),
    )
