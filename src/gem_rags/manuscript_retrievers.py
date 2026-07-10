from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from .types import Evidence, QAItem, RetrievalResult


class _Retriever(Protocol):
    name: str

    def retrieve(self, item: QAItem) -> RetrievalResult: ...


class MultimodalCandidateRetriever:
    """Rank text and figure metadata into one bounded candidate stream."""

    def __init__(
        self,
        name: str,
        text_retriever: _Retriever,
        figures: Sequence[dict[str, Any]],
        *,
        top_k: int = 18,
    ) -> None:
        self.name = name
        self.text_retriever = text_retriever
        self.figures = [dict(figure) for figure in figures]
        self.top_k = top_k

    def retrieve(self, item: QAItem) -> RetrievalResult:
        text_result = self.text_retriever.retrieve(item)
        ranked: list[tuple[float, Evidence]] = []
        for rank, evidence in enumerate(text_result.evidence):
            ranked.append((1.0 / (rank + 1), evidence))

        query_terms = _content_terms(item.question)
        for figure in self.figures:
            relevance = _term_relevance(query_terms, _content_terms(_figure_search_text(figure)))
            if relevance <= 0:
                continue
            ranked.append(
                (
                    relevance,
                    _figure_evidence(
                        figure,
                        relevance,
                        {"modality": "visual", "retrieval_stage": "shared_embedding_proxy"},
                    ),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1].evidence_id), reverse=True)
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=[evidence for _score, evidence in ranked[: self.top_k]],
            debug={
                **text_result.debug,
                "method": "multimodal_candidates",
                "text_adapter": text_result.adapter,
                "modalities": ["text", "visual"],
                "candidate_limit": self.top_k,
            },
            error=text_result.error,
        )


class SAMRAGRetriever:
    """Retrieval-side SAM-RAG batching and relevance verification policy."""

    def __init__(
        self,
        name: str,
        candidate_retriever: _Retriever,
        *,
        top_k: int = 6,
        batch_size: int = 6,
        relevance_threshold: float = 0.2,
    ) -> None:
        self.name = name
        self.candidate_retriever = candidate_retriever
        self.top_k = top_k
        self.batch_size = batch_size
        self.relevance_threshold = relevance_threshold

    def retrieve(self, item: QAItem) -> RetrievalResult:
        candidates = self.candidate_retriever.retrieve(item)
        query_terms = _content_terms(item.question)
        selected: list[Evidence] = []
        verification: dict[str, dict[str, Any]] = {}
        batches_scanned = 0
        for start in range(0, len(candidates.evidence), self.batch_size):
            batches_scanned += 1
            batch = candidates.evidence[start : start + self.batch_size]
            relevant = []
            for evidence in batch:
                score = _term_relevance(query_terms, _content_terms(evidence.text))
                is_relevant = score >= self.relevance_threshold
                verification[evidence.evidence_id] = {
                    "isRel": is_relevant,
                    "score": round(score, 4),
                    "kind": evidence.kind,
                }
                if is_relevant:
                    relevant.append(evidence)
            if relevant:
                selected.extend(relevant)
                break

        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=selected[: self.top_k],
            debug={
                **candidates.debug,
                "method": "sam_rag",
                "implementation": "retrieval_policy_adapted_from_official_flow",
                "candidate_adapter": candidates.adapter,
                "batch_size": self.batch_size,
                "batches_scanned": batches_scanned,
                "relevance_threshold": self.relevance_threshold,
                "stopped_after_relevant_batch": bool(selected),
                "verification": verification,
                "answer_verification": "reported_separately_by_harness_grading",
            },
            error=candidates.error,
        )


class M3KGRAGRetriever:
    """Paper-spec M3KG-RAG adaptation for image/text MUTCD evidence."""

    def __init__(
        self,
        name: str,
        text_seed_retriever: _Retriever,
        chunks: Sequence[dict[str, Any]],
        figures: Sequence[dict[str, Any]],
        graph: Any,
        *,
        top_k: int = 6,
        graph_hops: int = 2,
        presence_threshold: float = 0.2,
    ) -> None:
        self.name = name
        self.text_seed_retriever = text_seed_retriever
        self.top_k = top_k
        self.graph_hops = graph_hops
        self.presence_threshold = presence_threshold
        self.undirected = graph.to_undirected(as_view=True)
        self.chunk_by_id = {str(chunk.get("chunk_id")): dict(chunk) for chunk in chunks}
        self.figure_by_id = {str(figure.get("figure_id")): dict(figure) for figure in figures}

    def retrieve(self, item: QAItem) -> RetrievalResult:
        query_terms = _content_terms(item.question)
        text_result = self.text_seed_retriever.retrieve(item)
        text_seed_ids = [
            evidence.evidence_id
            for evidence in text_result.evidence
            if evidence.kind == "chunk" and evidence.evidence_id in self.chunk_by_id
        ]
        figure_scores = {
            figure_id: _term_relevance(query_terms, _content_terms(_figure_search_text(figure)))
            for figure_id, figure in self.figure_by_id.items()
        }
        figure_seed_ids = [
            figure_id
            for figure_id, score in sorted(figure_scores.items(), key=lambda item: item[1], reverse=True)
            if score >= self.presence_threshold
        ][: self.top_k]

        candidate_chunk_ids = set(text_seed_ids)
        candidate_figure_ids = set(figure_seed_ids)
        starts = [f"chunk:{chunk_id}" for chunk_id in text_seed_ids]
        starts.extend(f"figure:{figure_id}" for figure_id in figure_seed_ids)
        for start in starts:
            for node in _bfs_distances(self.undirected, start, self.graph_hops):
                node_text = str(node)
                if node_text.startswith("chunk:"):
                    chunk_id = node_text.split(":", 1)[1]
                    if chunk_id in self.chunk_by_id:
                        candidate_chunk_ids.add(chunk_id)
                elif node_text.startswith("figure:"):
                    figure_id = node_text.split(":", 1)[1]
                    if figure_id in self.figure_by_id:
                        candidate_figure_ids.add(figure_id)

        kept: list[tuple[float, Evidence]] = []
        pruned = []
        for chunk_id in candidate_chunk_ids:
            chunk = self.chunk_by_id[chunk_id]
            presence = _term_relevance(query_terms, _content_terms(_chunk_search_text(chunk)))
            if chunk_id not in text_seed_ids and presence < self.presence_threshold:
                pruned.append(chunk_id)
                continue
            score = presence + (1.0 if chunk_id in text_seed_ids else 0.0)
            kept.append(
                (
                    score,
                    _chunk_evidence(
                        chunk,
                        score,
                        {
                            "modality": "text",
                            "retrieval_stage": "modality_seed" if chunk_id in text_seed_ids else "graph_lift",
                            "grasp_presence_score": presence,
                        },
                    ),
                )
            )
        for figure_id in candidate_figure_ids:
            figure = self.figure_by_id[figure_id]
            presence = figure_scores.get(
                figure_id,
                _term_relevance(query_terms, _content_terms(_figure_search_text(figure))),
            )
            if figure_id not in figure_seed_ids and presence < self.presence_threshold:
                pruned.append(figure_id)
                continue
            score = presence + (1.0 if figure_id in figure_seed_ids else 0.0)
            kept.append(
                (
                    score,
                    _figure_evidence(
                        figure,
                        score,
                        {
                            "modality": "visual",
                            "retrieval_stage": "modality_seed" if figure_id in figure_seed_ids else "graph_lift",
                            "grasp_presence_score": presence,
                        },
                    ),
                )
            )
        kept.sort(key=lambda item: (item[0], item[1].evidence_id), reverse=True)
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=[evidence for _score, evidence in kept[: self.top_k]],
            debug={
                "method": "m3kg_rag",
                "implementation": "paper_spec_no_public_code",
                "modalities": ["text", "visual"],
                "text_seed_ids": text_seed_ids,
                "visual_seed_ids": figure_seed_ids,
                "graph_hops": self.graph_hops,
                "presence_threshold": self.presence_threshold,
                "grasp_pruned_ids": sorted(pruned),
                "grasp_answer_usefulness": "lexical_query_alignment_proxy",
                "audio_supported": False,
            },
            error=text_result.error,
        )


class OKHRAGRetriever:
    """Paper-spec ordered-hypergraph retrieval using MUTCD sections as hyperedges."""

    def __init__(
        self,
        name: str,
        seed_retriever: _Retriever,
        chunks: Sequence[dict[str, Any]],
        graph: Any,
        *,
        top_k: int = 6,
        graph_hops: int = 2,
    ) -> None:
        self.name = name
        self.seed_retriever = seed_retriever
        self.top_k = top_k
        self.graph_hops = graph_hops
        self.undirected = graph.to_undirected(as_view=True)
        self.chunk_by_id = {str(chunk.get("chunk_id")): dict(chunk) for chunk in chunks}

    def retrieve(self, item: QAItem) -> RetrievalResult:
        seed_result = self.seed_retriever.retrieve(item)
        seed_scores = {
            evidence.evidence_id: evidence.score
            for evidence in seed_result.evidence
            if evidence.kind == "chunk" and evidence.evidence_id in self.chunk_by_id
        }
        candidate_ids = set(seed_scores)
        for chunk_id in seed_scores:
            for node in _bfs_distances(self.undirected, f"chunk:{chunk_id}", self.graph_hops):
                node_text = str(node)
                if not node_text.startswith("chunk:"):
                    continue
                candidate_id = node_text.split(":", 1)[1]
                if candidate_id in self.chunk_by_id:
                    candidate_ids.add(candidate_id)

        query_terms = _content_terms(item.question)
        hyperedges: dict[str, list[str]] = {}
        for chunk_id in candidate_ids:
            section_id = str(self.chunk_by_id[chunk_id].get("section_id") or "unscoped")
            hyperedges.setdefault(section_id, []).append(chunk_id)
        hyperedge_scores = {
            section_id: max(
                seed_scores.get(chunk_id, 0.0)
                + _term_relevance(query_terms, _content_terms(_chunk_search_text(self.chunk_by_id[chunk_id])))
                for chunk_id in chunk_ids
            )
            for section_id, chunk_ids in hyperedges.items()
        }
        ordered_hyperedges = sorted(hyperedges, key=lambda section_id: (-hyperedge_scores[section_id], section_id))

        trajectory: list[tuple[str, str]] = []
        for section_id in ordered_hyperedges:
            remaining = self.top_k - len(trajectory)
            if remaining <= 0:
                break
            ordered_chunks = sorted(hyperedges[section_id], key=lambda chunk_id: _document_order(self.chunk_by_id[chunk_id]))
            if len(ordered_chunks) > remaining:
                anchor_id = max(
                    ordered_chunks,
                    key=lambda chunk_id: (
                        chunk_id in seed_scores,
                        seed_scores.get(chunk_id, 0.0),
                        _term_relevance(
                            query_terms,
                            _content_terms(_chunk_search_text(self.chunk_by_id[chunk_id])),
                        ),
                    ),
                )
                anchor_index = ordered_chunks.index(anchor_id)
                start = max(0, anchor_index - remaining // 2)
                start = min(start, len(ordered_chunks) - remaining)
                ordered_chunks = ordered_chunks[start : start + remaining]
            trajectory.extend((section_id, chunk_id) for chunk_id in ordered_chunks)

        evidence = []
        previous_id = None
        for index, (section_id, chunk_id) in enumerate(trajectory):
            score = hyperedge_scores[section_id]
            evidence.append(
                _chunk_evidence(
                    self.chunk_by_id[chunk_id],
                    score,
                    {
                        "hyperedge_id": f"section:{section_id}",
                        "trajectory_index": index,
                        "preceded_by": previous_id,
                        "precedence_source": "document_order",
                    },
                )
            )
            previous_id = chunk_id

        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                "method": "okh_rag",
                "implementation": "paper_spec_no_public_code",
                "seed_adapter": seed_result.adapter,
                "seed_chunk_ids": list(seed_scores),
                "hyperedge_model": "mutcd_section_membership",
                "ordered_hyperedges": ordered_hyperedges,
                "trajectory": [chunk_id for _section_id, chunk_id in trajectory],
                "precedence_source": "document_order",
                "learned_transition_model": False,
            },
            error=seed_result.error,
        )


class KG2RAGRetriever:
    """KG2RAG seed-expand-organize retrieval adapted to the MUTCD graph."""

    def __init__(
        self,
        name: str,
        seed_retriever: _Retriever,
        chunks: Sequence[dict[str, Any]],
        graph: Any,
        *,
        top_k: int = 6,
        graph_hops: int = 2,
    ) -> None:
        self.name = name
        self.seed_retriever = seed_retriever
        self.top_k = top_k
        self.graph_hops = graph_hops
        self.graph = graph
        self.undirected = graph.to_undirected(as_view=True)
        self.chunk_by_id = {str(chunk.get("chunk_id")): dict(chunk) for chunk in chunks}

    def retrieve(self, item: QAItem) -> RetrievalResult:
        seed_result = self.seed_retriever.retrieve(item)
        seed_scores = {
            evidence.evidence_id: evidence.score
            for evidence in seed_result.evidence
            if evidence.kind == "chunk"
        }
        candidate_scores = dict(seed_scores)
        graph_distances: dict[str, int] = {chunk_id: 0 for chunk_id in seed_scores}
        for seed_id, seed_score in seed_scores.items():
            start = f"chunk:{seed_id}"
            for node, distance in _bfs_distances(self.undirected, start, self.graph_hops).items():
                if not str(node).startswith("chunk:"):
                    continue
                chunk_id = str(node).split(":", 1)[1]
                if chunk_id not in self.chunk_by_id:
                    continue
                propagated = float(seed_score) / (distance + 1)
                candidate_scores[chunk_id] = max(candidate_scores.get(chunk_id, 0.0), propagated)
                graph_distances[chunk_id] = min(graph_distances.get(chunk_id, distance), distance)

        ordered_ids = sorted(
            candidate_scores,
            key=lambda chunk_id: (
                chunk_id not in seed_scores,
                _document_order(self.chunk_by_id[chunk_id]),
            ),
        )[: self.top_k]
        evidence = [
            _chunk_evidence(
                self.chunk_by_id[chunk_id],
                candidate_scores[chunk_id],
                {
                    "retrieval_stage": "seed" if chunk_id in seed_scores else "graph_expansion",
                    "graph_distance": graph_distances.get(chunk_id),
                    "context_order": index,
                },
            )
            for index, chunk_id in enumerate(ordered_ids)
        ]
        expanded = [chunk_id for chunk_id in ordered_ids if chunk_id not in seed_scores]
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                "method": "kg2rag",
                "implementation": "mutcd_adaptation_of_official_algorithm",
                "seed_adapter": seed_result.adapter,
                "seed_chunk_ids": list(seed_scores),
                "expanded_chunk_ids": expanded,
                "graph_hops": self.graph_hops,
                "organization": "seed_first_then_document_order",
            },
            error=seed_result.error,
        )


def _bfs_distances(graph: Any, start: str, max_hops: int) -> dict[str, int]:
    if not graph.has_node(start):
        return {}
    distances = {start: 0}
    frontier = [start]
    for distance in range(1, max_hops + 1):
        next_frontier = []
        for node in frontier:
            for neighbor in graph.neighbors(node):
                if neighbor in distances:
                    continue
                distances[neighbor] = distance
                next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return distances


def _document_order(record: dict[str, Any]) -> tuple[int, str, int]:
    try:
        page = int(record.get("page_pdf") or 0)
    except (TypeError, ValueError):
        page = 0
    try:
        ordinal = int(record.get("ordinal") or 0)
    except (TypeError, ValueError):
        ordinal = 0
    return page, str(record.get("section_id") or ""), ordinal


def _chunk_evidence(chunk: dict[str, Any], score: float, extra: dict[str, Any]) -> Evidence:
    chunk_id = str(chunk.get("chunk_id"))
    header = (
        f"Section {chunk.get('section_id')} {chunk.get('content_type')} "
        f"{chunk.get('ordinal')} - {chunk.get('section_title')} "
        f"(p.{chunk.get('page_printed')}, {chunk.get('part')})"
    )
    return Evidence(
        evidence_id=chunk_id,
        kind="chunk",
        text=f"{header}\n{str(chunk.get('text') or '').strip()}",
        metadata={
            **dict(chunk),
            **extra,
            "chunk_id": chunk_id,
        },
        score=float(score),
    )


def _figure_evidence(figure: dict[str, Any], score: float, extra: dict[str, Any]) -> Evidence:
    figure_id = str(figure.get("figure_id"))
    return Evidence(
        evidence_id=figure_id,
        kind="figure",
        text=f"{figure_id}: {figure.get('caption') or figure.get('title') or ''}",
        metadata={**dict(figure), **extra, "figure_id": figure_id},
        score=float(score),
    )


def _chunk_search_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(key) or "")
        for key in ["part", "chapter", "section_id", "section_title", "content_type", "text", "sign_codes"]
    )


def _figure_search_text(figure: dict[str, Any]) -> str:
    return " ".join(
        str(figure.get(key) or "")
        for key in ["figure_id", "caption", "title", "sign_codes_depicted"]
    )


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "does",
    "for",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def _content_terms(text: str) -> set[str]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*", text)}
    normalized = {"require" if term in {"required", "requires", "requiring"} else term for term in terms}
    return normalized - _STOPWORDS


def _term_relevance(query_terms: set[str], evidence_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & evidence_terms) / len(query_terms)
