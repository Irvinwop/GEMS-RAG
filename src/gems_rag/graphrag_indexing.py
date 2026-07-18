"""Small, reversible adaptations around GraphRAG's official index workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from functools import wraps
from typing import Any


Workflow = Callable[[Any, Any], Awaitable[Any]]


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
