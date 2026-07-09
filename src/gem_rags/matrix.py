from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DatasetConfig, ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from .preflight import preflight_config


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def parse_model_spec(spec: str) -> ModelConfig:
    provider, model, options = _parse_provider_model_spec(spec, "model")
    return ModelConfig(provider=provider, model=model, options=options)


def load_model_specs_file(path: Path) -> list[ModelConfig]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [
            parse_model_spec(line)
            for line in _plain_spec_lines(text)
        ]
    if isinstance(payload, dict):
        payload = payload.get("models")
    if not isinstance(payload, list):
        raise ValueError("models file must be a JSON list, a JSON object with a models list, or plain spec lines")
    models = []
    for item in payload:
        if isinstance(item, str):
            models.append(parse_model_spec(item))
        elif isinstance(item, dict):
            if "provider" not in item or "model" not in item:
                raise ValueError(f"model entry must include provider and model: {item!r}")
            models.append(
                ModelConfig(
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    options=dict(item.get("options", {})),
                )
            )
        else:
            raise ValueError(f"unsupported model entry in models file: {item!r}")
    return models


def parse_grader_spec(spec: str) -> GraderConfig:
    provider, model, options = _parse_provider_model_spec(spec, "grader")
    return GraderConfig(provider=provider, model=model, options=options)


def materialize_config(
    base: ExperimentConfig,
    *,
    name: str | None = None,
    limit: int | None = None,
    qa_ids: list[str] | None = None,
    retrievers: list[RetrieverConfig] | None = None,
    retriever_names: list[str] | None = None,
    drop_retriever_names: list[str] | None = None,
    context_modes: list[str] | None = None,
    models: list[ModelConfig] | None = None,
    grader: GraderConfig | None = None,
    max_evidence_chars: int | None = None,
    dry_run: bool | None = None,
) -> ExperimentConfig:
    retrievers = list(retrievers) if retrievers is not None else list(base.retrievers)
    if retriever_names is not None:
        wanted = set(retriever_names)
        retrievers = [retriever for retriever in retrievers if retriever.name in wanted]
        missing = wanted - {retriever.name for retriever in retrievers}
        if missing:
            raise ValueError(f"unknown retriever names: {sorted(missing)}")
    if drop_retriever_names:
        drop = set(drop_retriever_names)
        retrievers = [retriever for retriever in retrievers if retriever.name not in drop]

    dataset = base.dataset
    if limit is not None or qa_ids is not None:
        dataset = DatasetConfig(
            qa_path=dataset.qa_path,
            mrag_dir=dataset.mrag_dir,
            limit=limit if limit is not None else dataset.limit,
            qa_ids=qa_ids if qa_ids is not None else dataset.qa_ids,
        )

    return ExperimentConfig(
        name=name or base.name,
        dataset=dataset,
        retrievers=retrievers,
        context_modes=list(context_modes) if context_modes is not None else list(base.context_modes),
        models=list(models) if models is not None else list(base.models),
        grader=grader or base.grader,
        output_dir=base.output_dir,
        max_evidence_chars=max_evidence_chars if max_evidence_chars is not None else base.max_evidence_chars,
        dry_run=base.dry_run if dry_run is None else dry_run,
    )


def filter_ready_config(
    config: ExperimentConfig,
    *,
    check_external: bool = True,
    timeout_s: int = 30,
    allow_not_checked: bool = False,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    report = preflight_config(config, check_external=check_external, timeout_s=timeout_s)
    allowed_statuses = {"ready"}
    if allow_not_checked:
        allowed_statuses.add("not_checked")
    dataset_status = report["sections"]["dataset"].get("status")
    if dataset_status not in allowed_statuses:
        raise ValueError(f"ready-only cannot fix dataset status {dataset_status}: {report['sections']['dataset'].get('problems', [])}")
    bad_modes = [
        mode
        for mode in report["sections"]["context_modes"]
        if mode.get("status") not in allowed_statuses
    ]
    if bad_modes:
        raise ValueError(f"ready-only cannot fix context modes: {bad_modes}")
    grader_status = report["sections"]["grader"].get("status")
    if grader_status not in allowed_statuses:
        raise ValueError(f"ready-only cannot fix grader status {grader_status}: {report['sections']['grader'].get('problems', [])}")
    retrievers = [
        retriever
        for retriever, status in zip(
            config.retrievers,
            (section.get("status") for section in report["sections"]["retrievers"]),
            strict=True,
        )
        if status in allowed_statuses
    ]
    models = [
        model
        for model, status in zip(
            config.models,
            (section.get("status") for section in report["sections"]["models"]),
            strict=True,
        )
        if status in allowed_statuses
    ]
    filtered = replace(config, retrievers=retrievers, models=models)
    if not filtered.retrievers:
        raise ValueError("ready-only filtering removed every retriever")
    if not filtered.models:
        raise ValueError("ready-only filtering removed every model")
    return filtered, report


def _parse_provider_model_spec(spec: str, label: str) -> tuple[str, str, dict[str, Any]]:
    head, *option_parts = [part.strip() for part in spec.split(",")]
    if ":" not in head:
        raise ValueError(f"{label} spec must use provider:model syntax: {spec!r}")
    provider, model = [part.strip() for part in head.split(":", 1)]
    if not provider or not model:
        raise ValueError(f"{label} spec must include both provider and model: {spec!r}")
    options = {}
    for part in option_parts:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{label} option must use key=value syntax: {part!r}")
        key, value = [item.strip() for item in part.split("=", 1)]
        if not key:
            raise ValueError(f"{label} option has an empty key: {part!r}")
        options[key] = _coerce_option_value(value)
    return provider, model, options


def _plain_spec_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _coerce_option_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
