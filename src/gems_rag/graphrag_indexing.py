"""Small, reversible adaptations around GraphRAG's official index workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from functools import wraps
from typing import Any, get_args


Workflow = Callable[[Any, Any], Awaitable[Any]]
_BOUNDED_REPORT_FORMATS: dict[type, type] = {}


def install_community_report_token_floor(max_tokens: int) -> int:
    """Raise only the provider-side report budget without changing cache keys."""
    if max_tokens <= 0:
        raise ValueError("community report token floor must be positive")

    import litellm

    if not hasattr(litellm, "_gems_rag_original_completion"):
        litellm._gems_rag_original_completion = litellm.completion
        litellm._gems_rag_original_acompletion = litellm.acompletion

        def completion(**kwargs: Any):
            adjusted = _community_report_provider_args(
                kwargs,
                litellm._gems_rag_community_report_token_floor,
            )
            return litellm._gems_rag_original_completion(**adjusted)

        async def acompletion(**kwargs: Any):
            adjusted = _community_report_provider_args(
                kwargs,
                litellm._gems_rag_community_report_token_floor,
            )
            return await litellm._gems_rag_original_acompletion(**adjusted)

        litellm.completion = completion
        litellm.acompletion = acompletion

    current = getattr(litellm, "_gems_rag_community_report_token_floor", 0)
    litellm._gems_rag_community_report_token_floor = max(current, max_tokens)
    return litellm._gems_rag_community_report_token_floor


def _community_report_provider_args(
    kwargs: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    configured = kwargs.get("max_tokens")
    response_format = kwargs.get("response_format")
    if (
        isinstance(configured, int)
        and configured < max_tokens
        and getattr(response_format, "__name__", None) == "CommunityReportResponse"
    ):
        return {
            **kwargs,
            "max_tokens": max_tokens,
            "response_format": _bounded_community_report_format(response_format),
        }
    return kwargs


def _bounded_community_report_format(response_format: type) -> type:
    cached = _BOUNDED_REPORT_FORMATS.get(response_format)
    if cached is not None:
        return cached

    from pydantic import Field, create_model

    fields = getattr(response_format, "model_fields", {})
    findings_field = fields.get("findings")
    finding_types = get_args(getattr(findings_field, "annotation", None))
    if not finding_types:
        return response_format
    finding_format = finding_types[0]

    bounded_finding = create_model(
        "FindingModel",
        __base__=finding_format,
        summary=(str, Field(min_length=1, max_length=240)),
        explanation=(str, Field(min_length=1, max_length=1000)),
    )
    bounded_report = create_model(
        "CommunityReportResponse",
        __base__=response_format,
        title=(str, Field(min_length=1, max_length=200)),
        summary=(str, Field(min_length=1, max_length=1200)),
        findings=(list[bounded_finding], Field(min_length=1, max_length=4)),
        rating_explanation=(str, Field(min_length=1, max_length=600)),
    )
    _BOUNDED_REPORT_FORMATS[response_format] = bounded_report
    return bounded_report


def install_community_report_level_filter(levels: Iterable[int]) -> tuple[int, ...]:
    """Limit report generation while preserving GraphRAG's complete community table."""
    selected = tuple(sorted({int(level) for level in levels}))
    if not selected or selected[0] < 0:
        raise ValueError("community report levels must contain non-negative integers")

    from graphrag.data_model.data_reader import DataReader
    from graphrag.index.workflows.factory import PipelineFactory

    for name in ("create_community_reports", "create_community_reports_text"):
        workflow = PipelineFactory.workflows.get(name)
        if workflow is None:
            raise RuntimeError(f"GraphRAG workflow is not registered: {name}")
        PipelineFactory.workflows[name] = _filtered_community_workflow(
            workflow,
            DataReader,
            frozenset(selected),
        )
    return selected


def _filtered_community_workflow(
    workflow: Workflow,
    data_reader_type: type,
    levels: frozenset[int],
) -> Workflow:
    @wraps(workflow)
    async def run(config: Any, context: Any) -> Any:
        read_all_communities = data_reader_type.communities

        async def read_selected_communities(reader: Any):
            communities = await read_all_communities(reader)
            selected = communities.loc[communities["level"].isin(levels)].copy()
            if selected.empty:
                available = sorted(
                    int(level) for level in communities["level"].dropna().unique()
                )
                raise ValueError(
                    "no GraphRAG communities matched report levels "
                    f"{sorted(levels)}; available levels are {available}"
                )
            return selected

        data_reader_type.communities = read_selected_communities
        try:
            return await workflow(config, context)
        finally:
            data_reader_type.communities = read_all_communities

    return run
