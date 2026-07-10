from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


REFERENCE_MODES = (
    "dense",
    "hybrid",
    "multimodal",
    "full",
    "no_graph",
    "no_visual",
    "no_rule",
    "no_hierarchy",
)


@dataclass(frozen=True)
class _ModeFeatures:
    sparse: bool = False
    visual: bool = False
    graph: bool = False
    hierarchy: bool = False
    rule_ranking: bool = False
    rerank: bool = False


_MODE_FEATURES = {
    "dense": _ModeFeatures(),
    "hybrid": _ModeFeatures(sparse=True),
    "multimodal": _ModeFeatures(sparse=True, visual=True),
    "full": _ModeFeatures(True, True, True, True, True, True),
    "no_graph": _ModeFeatures(True, True, False, True, True, True),
    "no_visual": _ModeFeatures(True, False, True, True, True, True),
    "no_rule": _ModeFeatures(True, True, True, True, False, True),
    "no_hierarchy": _ModeFeatures(True, True, True, False, True, True),
}


def retrieve_reference_mode(
    pipeline: Any,
    query: str,
    *,
    mode: str,
    top_k: int,
    chunks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    try:
        features = _MODE_FEATURES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown MRAG reference mode: {mode!r}") from exc
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not features.sparse:
        dense = pipeline.text.encode_dense([query])[0]
        hits = _dense_hits(pipeline.store, dense, top_k=top_k)
    else:
        dense_batch, sparse_batch = pipeline.text.encode_both([query])
        dense = dense_batch[0]
        candidate_k = max(top_k * 5, 30) if features.rerank else top_k
        hits = _hybrid_hits(
            pipeline.store,
            dense,
            sparse_batch[0],
            top_k=candidate_k,
        )

    hits = _rehydrate_canonical_payloads(hits, chunks)

    debug: dict[str, Any] = {
        "mode": mode,
        "components": ["dense"] + (["sparse"] if features.sparse else []),
        "initial_candidates": len(hits),
        "graph_expanded_chunks": 0,
    }
    if features.rerank:
        retrieved, rank_debug = _rank_candidates(
            pipeline,
            query,
            hits,
            chunks,
            top_k=top_k,
            features=features,
        )
        debug.update(rank_debug)
        if features.graph:
            debug["components"].append("graph")
        if features.hierarchy:
            debug["components"].append("hierarchy")
        if features.rule_ranking:
            debug["components"].append("rule_ranking")
        debug["components"].append("reranker")
    else:
        retrieved = [
            {**dict(hit["payload"]), "score": float(hit["score"])}
            for hit in hits[:top_k]
        ]

    figures: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    if features.visual:
        figures, pages = _retrieve_visual_evidence(
            pipeline,
            query,
            dense,
            retrieved,
            top_k=top_k,
            include_graph_links=features.graph,
        )
        debug["components"].append("visual")

    return {
        "question": query,
        "chunks": retrieved,
        "figures": figures,
        "pages": pages,
        "debug": debug,
    }


def _rehydrate_canonical_payloads(
    hits: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunk_by_id = {str(chunk.get("chunk_id")): dict(chunk) for chunk in chunks}
    canonical_hits: list[dict[str, Any]] = []
    for hit in hits:
        canonical_hit = dict(hit)
        payload = dict(hit.get("payload") or {})
        chunk_id = str(payload.get("chunk_id") or hit.get("id"))
        canonical_hit["payload"] = chunk_by_id.get(chunk_id, payload)
        canonical_hits.append(canonical_hit)
    return canonical_hits


def _dense_hits(store: Any, dense: Any, *, top_k: int) -> list[dict[str, Any]]:
    response = store._client.query_points(
        collection_name="mutcd_chunks",
        query=_tolist(dense),
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": point.id,
            "score": float(point.score),
            "payload": dict(point.payload or {}),
        }
        for point in response.points
    ]


def _hybrid_hits(store: Any, dense: Any, sparse: dict[int, float], *, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "id": hit.get("id"),
            "score": float(hit.get("score", 0.0)),
            "payload": dict(hit.get("payload") or {}),
        }
        for hit in store.search_chunks_hybrid(
            "mutcd_chunks",
            dense,
            sparse,
            top_k=top_k,
        )
    ]


def _rank_candidates(
    pipeline: Any,
    query: str,
    hits: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    *,
    top_k: int,
    features: _ModeFeatures,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pipeline.rerank is None:
        raise RuntimeError("enhanced MRAG modes require the cross-encoder reranker")
    if features.graph and pipeline.kg is None:
        raise RuntimeError("graph-enabled MRAG modes require the MUTCD knowledge graph")

    chunk_by_id = {str(chunk.get("chunk_id")): dict(chunk) for chunk in chunks}
    candidates: dict[str, dict[str, Any]] = {}
    for hit in hits:
        payload = dict(hit["payload"])
        chunk_id = str(payload.get("chunk_id") or hit.get("id"))
        candidates[chunk_id] = {
            "payload": payload,
            "base_score": float(hit["score"]),
            "graph_seed_score": 0.0,
        }

    explicit_entities: set[str] = set()
    initial_ids = set(candidates)
    if features.graph:
        explicit_entities = set(pipeline.kg.query_entities(query))
        max_base = max((record["base_score"] for record in candidates.values()), default=1.0) or 1.0
        seed_records = list(candidates.items())[:10]
        for chunk_id, record in seed_records:
            propagated = 0.5 * record["base_score"] / max_base
            _add_graph_neighbors(
                pipeline.kg,
                f"chunk:{chunk_id}",
                candidates,
                chunk_by_id,
                propagated,
            )
        for entity in explicit_entities:
            _add_graph_neighbors(
                pipeline.kg,
                entity,
                candidates,
                chunk_by_id,
                1.0,
            )

    if not candidates:
        return [], {
            "ranked_candidates": 0,
            "graph_expanded_chunks": 0,
            "explicit_graph_entities": sorted(explicit_entities),
            "scoring_weights": _scoring_weights(features),
        }

    max_base = max((record["base_score"] for record in candidates.values()), default=1.0) or 1.0
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for chunk_id, record in candidates.items():
        payload = record["payload"]
        base_score = record["base_score"] / max_base
        graph_score = record["graph_seed_score"]
        if features.graph:
            graph_score = max(
                graph_score,
                float(pipeline.kg.proximity_score(explicit_entities, chunk_id)),
            )
        hierarchy_score = _hierarchy_prior(query, payload) if features.hierarchy else 0.0
        rule_score = _rule_prior(payload) if features.rule_ranking else 0.0
        final_score = base_score + 0.4 * graph_score + 0.2 * hierarchy_score + 0.3 * rule_score
        scored.append((final_score, chunk_id, payload))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    precursor = scored[: max(top_k * 5, top_k)]
    documents = [payload.get("text", "")[:1500] for _score, _chunk_id, payload in precursor]
    reranked = pipeline.rerank.rank(query, documents, top_k=top_k)
    retrieved = []
    for index, rerank_score in reranked:
        if index < 0 or index >= len(precursor):
            continue
        _score, _chunk_id, payload = precursor[index]
        retrieved.append({**dict(payload), "score": float(rerank_score)})

    return retrieved, {
        "ranked_candidates": len(candidates),
        "graph_expanded_chunks": len(set(candidates) - initial_ids),
        "explicit_graph_entities": sorted(explicit_entities),
        "scoring_weights": _scoring_weights(features),
    }


def _scoring_weights(features: _ModeFeatures) -> dict[str, float]:
    return {
        "hybrid": 1.0,
        "graph": 0.4 if features.graph else 0.0,
        "hierarchy": 0.2 if features.hierarchy else 0.0,
        "rule": 0.3 if features.rule_ranking else 0.0,
    }


def _add_graph_neighbors(
    kg: Any,
    start_node: str,
    candidates: dict[str, dict[str, Any]],
    chunk_by_id: dict[str, dict[str, Any]],
    propagated_score: float,
) -> None:
    for node in kg.neighbors(start_node, n_hops=2):
        if not str(node).startswith("chunk:"):
            continue
        chunk_id = str(node).split(":", 1)[1]
        if chunk_id not in candidates:
            payload = chunk_by_id.get(chunk_id)
            if payload is None:
                continue
            candidates[chunk_id] = {
                "payload": payload,
                "base_score": 0.0,
                "graph_seed_score": propagated_score,
            }
        else:
            candidates[chunk_id]["graph_seed_score"] = max(
                candidates[chunk_id]["graph_seed_score"],
                propagated_score,
            )


def _retrieve_visual_evidence(
    pipeline: Any,
    query: str,
    dense: Any,
    retrieved_chunks: Sequence[dict[str, Any]],
    *,
    top_k: int,
    include_graph_links: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pipeline.image is None:
        raise RuntimeError("visual MRAG modes require the ColQwen/ColPali query encoder")

    linked_figures = (
        _linked_figure_payloads(pipeline.kg, retrieved_chunks)
        if include_graph_links
        else []
    )
    visual_query = pipeline.image.encode_queries([query])[0]
    visual_figures = _points_to_payloads(
        pipeline.store.search_figures_visual(
            "mutcd_figures_visual",
            visual_query,
            top_k=top_k,
        ),
        source="visual",
    )
    caption_figures = _points_to_payloads(
        pipeline.store.search_figures(
            "mutcd_figures",
            dense,
            top_k=top_k,
        ),
        source="caption",
    )
    figures = _dedupe_figures(linked_figures + visual_figures + caption_figures)[:top_k]
    pages = _points_to_payloads(
        pipeline.store.search_pages(
            "mutcd_pages",
            visual_query,
            top_k=top_k,
        ),
        source="visual",
    )[:top_k]
    return figures, pages


def _linked_figure_payloads(kg: Any, chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    figures = []
    for chunk in chunks:
        for figure_id in kg.figures_for_chunk(str(chunk.get("chunk_id") or "")):
            node = kg.figure(figure_id)
            if not node:
                continue
            data = dict(kg.g.nodes[node])
            figures.append(
                {
                    "figure_id": data.get("id", figure_id),
                    "page_pdf": data.get("page_pdf"),
                    "page_printed": data.get("page_printed"),
                    "caption": data.get("caption", ""),
                    "image_path": data.get("image_path", ""),
                    "sign_codes": list(data.get("sign_codes", [])),
                    "score": 1.0,
                    "source": "kg_link",
                }
            )
    return _dedupe_figures(figures)


def _hierarchy_prior(query: str, payload: dict[str, Any]) -> float:
    score = 0.0
    lowered = query.lower()
    part = str(payload.get("part") or "").lower()
    chapter = str(payload.get("chapter") or "").lower()
    part_match = re.search(r"\bpart\s+(\d+)\b", lowered)
    if part_match and f"part {part_match.group(1)}" in part:
        score += 0.5
    chapter_match = re.search(r"\bchapter\s+([0-9a-z]+)\b", lowered)
    if chapter_match and chapter_match.group(1) in chapter:
        score += 0.5
    return min(score, 1.0)


def _rule_prior(payload: dict[str, Any]) -> float:
    return {
        "standard": 1.0,
        "guidance": 2.0 / 3.0,
        "option": 1.0 / 3.0,
        "support": 0.0,
    }.get(str(payload.get("content_type") or "").lower(), 0.0)


def _tolist(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _points_to_payloads(points: Sequence[Any], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            **dict(getattr(point, "payload", None) or {}),
            "score": float(getattr(point, "score", 0.0)),
            "source": source,
        }
        for point in points
    ]


def _dedupe_figures(figures: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for figure in figures:
        figure_id = str(figure.get("figure_id") or figure.get("image_path") or "")
        if figure_id in seen:
            continue
        seen.add(figure_id)
        result.append(figure)
    return result
