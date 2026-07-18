from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ALL_CONTEXT_MODES, ExperimentConfig, write_experiment_config


def write_context_segments(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    context_modes: list[str] | None = None,
) -> dict[str, Any]:
    modes = list(context_modes or ALL_CONTEXT_MODES)
    unknown = sorted(set(modes) - set(ALL_CONTEXT_MODES))
    if not modes or unknown:
        raise ValueError(f"invalid context modes: {unknown or 'empty'}")
    if len(set(modes)) != len(modes):
        raise ValueError("context modes must be unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = _safe_stem(config.name)
    segments = []
    for mode in modes:
        included = [
            retriever for retriever in config.retrievers if mode in retriever.context_modes
        ]
        if not included:
            raise ValueError(f"no retrievers support context mode {mode}")
        included_names = {retriever.name for retriever in included}
        excluded = [
            retriever.name
            for retriever in config.retrievers
            if retriever.name not in included_names
        ]
        experiment_name = f"{file_stem}-{mode.replace('_', '-')}"
        segmented = replace(
            config,
            name=experiment_name,
            retrievers=included,
            context_modes=[mode],
        )
        path = output_dir / f"{file_stem}-{mode}.json"
        write_experiment_config(segmented, path)
        segments.append(
            {
                "context_mode": mode,
                "experiment": experiment_name,
                "config": str(path),
                "retriever_count": len(included),
                "retrievers": [retriever.name for retriever in included],
                "excluded_retrievers": excluded,
                "run_command": [".venv/bin/gems-rag", "run", str(path), "--resume"],
            }
        )

    return {
        "schema_version": 1,
        "source_experiment": config.name,
        "output_dir": str(output_dir),
        "segments": segments,
        "segment_count": len(segments),
        "total_rows_per_question_model": sum(
            segment["retriever_count"] for segment in segments
        ),
    }


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return stem or "experiment"
