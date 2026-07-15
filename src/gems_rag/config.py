from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ContextMode


DEFAULT_MRAG_DIR = Path("data/extracted/MRAG-20260708T114057Z-3/MRAG")
ALL_CONTEXT_MODES: tuple[ContextMode, ...] = ("injected", "tool_explore", "tool_search", "tool_native")


@dataclass(frozen=True)
class DatasetConfig:
    qa_path: Path = DEFAULT_MRAG_DIR / "eval" / "gold_qa.jsonl"
    mrag_dir: Path = DEFAULT_MRAG_DIR
    limit: int | None = None
    qa_ids: list[str] | None = None


@dataclass(frozen=True)
class RetrieverConfig:
    name: str
    kind: str
    top_k: int = 6
    options: dict[str, Any] = field(default_factory=dict)
    context_modes: tuple[ContextMode, ...] = ALL_CONTEXT_MODES
    interaction: str = "query_driven"


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "dry_run"
    model: str = "dry-run"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderConfig:
    provider: str = "heuristic"
    model: str = "heuristic"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrievers: list[RetrieverConfig] = field(default_factory=list)
    context_modes: list[ContextMode] = field(default_factory=lambda: ["injected"])
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    grader: GraderConfig = field(default_factory=GraderConfig)
    output_dir: Path = Path("runs")
    max_evidence_chars: int = 1600
    dry_run: bool = False


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def incompatible_context_modes(retriever: RetrieverConfig, requested: list[ContextMode]) -> list[ContextMode]:
    supported = set(retriever.context_modes)
    return [mode for mode in requested if mode not in supported]


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset_raw = raw.get("dataset", {})
    dataset = DatasetConfig(
        qa_path=_path(dataset_raw.get("qa_path", DatasetConfig.qa_path)),
        mrag_dir=_path(dataset_raw.get("mrag_dir", DatasetConfig.mrag_dir)),
        limit=dataset_raw.get("limit"),
        qa_ids=dataset_raw.get("qa_ids"),
    )
    retrievers = [
        RetrieverConfig(
            name=item["name"],
            kind=item["kind"],
            top_k=int(item.get("top_k", 6)),
            options=dict(item.get("options", {})),
            context_modes=tuple(item.get("context_modes", ALL_CONTEXT_MODES)),
            interaction=str(item.get("interaction") or "query_driven"),
        )
        for item in raw.get("retrievers", [])
    ]
    models = [
        ModelConfig(
            provider=item.get("provider", "dry_run"),
            model=item.get("model", "dry-run"),
            options=dict(item.get("options", {})),
        )
        for item in raw.get("models", [{"provider": "dry_run", "model": "dry-run"}])
    ]
    grader_raw = raw.get("grader", {})
    grader = GraderConfig(
        provider=grader_raw.get("provider", "heuristic"),
        model=grader_raw.get("model", "heuristic"),
        options=dict(grader_raw.get("options", {})),
    )
    return ExperimentConfig(
        name=raw["name"],
        dataset=dataset,
        retrievers=retrievers,
        context_modes=list(raw.get("context_modes", ["injected"])),
        models=models,
        grader=grader,
        output_dir=_path(raw.get("output_dir", "runs")),
        max_evidence_chars=int(raw.get("max_evidence_chars", 1600)),
        dry_run=bool(raw.get("dry_run", False)),
    )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "dataset": {
            "qa_path": str(config.dataset.qa_path),
            "mrag_dir": str(config.dataset.mrag_dir),
            "limit": config.dataset.limit,
            "qa_ids": config.dataset.qa_ids,
        },
        "retrievers": [
            {
                "name": retriever.name,
                "kind": retriever.kind,
                "top_k": retriever.top_k,
                "options": retriever.options,
                "context_modes": list(retriever.context_modes),
                "interaction": retriever.interaction,
            }
            for retriever in config.retrievers
        ],
        "context_modes": list(config.context_modes),
        "models": [
            {
                "provider": model.provider,
                "model": model.model,
                "options": model.options,
            }
            for model in config.models
        ],
        "grader": {
            "provider": config.grader.provider,
            "model": config.grader.model,
            "options": config.grader.options,
        },
        "output_dir": str(config.output_dir),
        "max_evidence_chars": config.max_evidence_chars,
        "dry_run": config.dry_run,
    }


def write_experiment_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(experiment_config_to_dict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
