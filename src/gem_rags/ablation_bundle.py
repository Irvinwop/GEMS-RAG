from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .analysis import write_csv
from .config import DatasetConfig, ExperimentConfig, GraderConfig, load_experiment_config, write_experiment_config
from .data import load_qa_items
from .matrix import materialize_config
from .model_catalog import load_model_catalog, render_model_specs, select_model_catalog
from .planning import plan_experiment
from .preflight import preflight_config
from .qa_sets import make_qa_split, write_qa_split
from .retriever_catalog import catalog_entries_to_retrievers_payload, load_retriever_catalog, select_retriever_catalog


def prepare_ablation_bundle(
    *,
    base_config_path: Path,
    name: str | None = None,
    output_dir: Path | None = None,
    qa_size: int | None = None,
    qa_seed: int = 0,
    qa_strategy: str = "balanced",
    qa_ids: list[str] | None = None,
    limit: int | None = None,
    model_catalog_path: Path = Path("configs/model-catalog.example.json"),
    model_providers: list[str] | None = None,
    model_sizes: list[str] | None = None,
    model_roles: list[str] | None = None,
    model_tags: list[str] | None = None,
    include_disabled_models: bool = False,
    retriever_catalog_path: Path = Path("configs/retriever-catalog.example.json"),
    retriever_families: list[str] | None = None,
    retriever_modes: list[str] | None = None,
    retriever_tags: list[str] | None = None,
    include_disabled_retrievers: bool = False,
    context_modes: list[str] | None = None,
    grader: GraderConfig | None = None,
    max_evidence_chars: int | None = None,
    dry_run: bool | None = None,
    attach_preflight: bool = False,
    check_external: bool = True,
    timeout_s: int = 30,
) -> dict[str, Any]:
    if qa_size is not None and qa_ids:
        raise ValueError("--qa-size cannot be combined with explicit QA IDs")
    base = load_experiment_config(base_config_path)
    experiment_name = name or base.name
    bundle_dir = output_dir or Path("data/working/ablation-bundles") / experiment_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    qa_ids, qa_artifact = _prepare_qa_ids(
        base,
        bundle_dir=bundle_dir,
        qa_size=qa_size,
        qa_seed=qa_seed,
        qa_strategy=qa_strategy,
        qa_ids=qa_ids,
    )
    model_entries = select_model_catalog(
        load_model_catalog(model_catalog_path),
        providers=model_providers,
        sizes=model_sizes,
        roles=model_roles or ["answer"],
        tags=model_tags,
        include_disabled=include_disabled_models,
    )
    if not model_entries:
        raise ValueError("model catalog filters selected no models")
    retriever_entries = select_retriever_catalog(
        load_retriever_catalog(retriever_catalog_path),
        families=retriever_families,
        modes=retriever_modes,
        tags=retriever_tags,
        include_disabled=include_disabled_retrievers,
    )
    if not retriever_entries:
        raise ValueError("retriever catalog filters selected no retrievers")

    model_matrix_path = bundle_dir / "models.txt"
    model_matrix_path.write_text(render_model_specs(model_entries), encoding="utf-8")
    retriever_matrix_path = bundle_dir / "retrievers.json"
    _write_json(retriever_matrix_path, catalog_entries_to_retrievers_payload(retriever_entries))

    config = materialize_config(
        base,
        name=experiment_name,
        limit=limit,
        qa_ids=qa_ids,
        retrievers=[entry.config for entry in retriever_entries],
        context_modes=context_modes,
        models=[entry.config for entry in model_entries],
        grader=grader,
        max_evidence_chars=max_evidence_chars,
        dry_run=dry_run,
    )
    if qa_ids is not None and limit is None:
        config = replace(config, dataset=_dataset_without_limit(config))
    config_path = bundle_dir / "materialized_config.json"
    write_experiment_config(config, config_path)

    preflight_report = None
    preflight_path = None
    if attach_preflight:
        preflight_report = preflight_config(config, check_external=check_external, timeout_s=timeout_s)
        preflight_path = bundle_dir / "preflight.json"
        _write_json(preflight_path, preflight_report)

    plan = plan_experiment(config, preflight_report=preflight_report)
    plan_path = bundle_dir / "plan.json"
    plan_csv_path = bundle_dir / "plan.csv"
    _write_json(plan_path, plan)
    write_csv(plan_csv_path, plan["conditions"])

    report = {
        "status": "ready" if preflight_report is None or preflight_report.get("ok") else "blocked",
        "experiment": experiment_name,
        "bundle_dir": str(bundle_dir),
        "base_config": str(base_config_path),
        "qa_ids": len(qa_ids) if qa_ids is not None else None,
        "models": len(model_entries),
        "retrievers": len(retriever_entries),
        "context_modes": len(config.context_modes),
        "dry_run": config.dry_run,
        "row_estimate": plan["estimates"]["rows"],
        "total_model_calls": plan["estimates"]["total_model_calls"],
        "paid_model_calls": plan["estimates"]["paid_model_calls"],
        "artifacts": {
            "qa_split": str(qa_artifact) if qa_artifact else None,
            "models": str(model_matrix_path),
            "retrievers": str(retriever_matrix_path),
            "config": str(config_path),
            "plan_json": str(plan_path),
            "plan_csv": str(plan_csv_path),
            "preflight": str(preflight_path) if preflight_path else None,
        },
        "next_commands": {
            "preflight": f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight {config_path} --strict",
            "sweep": f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli sweep {config_path} --overwrite",
            "resume": f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli sweep {config_path} --resume",
            "retry_errors": f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli sweep {config_path} --retry-errors",
            "analyze_context": (
                f"PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze "
                f"{config.output_dir / config.name / 'runs.jsonl'} "
                f"--output-dir {config.output_dir / config.name / 'analysis'} "
                f"--qa-path {config.dataset.qa_path} --axis context_mode --baseline injected"
            ),
        },
    }
    if preflight_report is not None:
        report["preflight_ok"] = preflight_report["ok"]
        report["preflight_blocking"] = preflight_report.get("blocking", [])
    return report


def _prepare_qa_ids(
    base: ExperimentConfig,
    *,
    bundle_dir: Path,
    qa_size: int | None,
    qa_seed: int,
    qa_strategy: str,
    qa_ids: list[str] | None,
) -> tuple[list[str] | None, Path | None]:
    if qa_size is not None:
        split = {
            "qa_path": str(base.dataset.qa_path),
            **make_qa_split(load_qa_items(base.dataset.qa_path), size=qa_size, seed=qa_seed, strategy=qa_strategy),
        }
        path = bundle_dir / "qa_split.json"
        write_qa_split(path, split)
        return list(split["qa_ids"]), path
    if qa_ids is not None:
        path = bundle_dir / "qa_ids.json"
        payload = {"qa_path": str(base.dataset.qa_path), "qa_ids": qa_ids}
        write_qa_split(path, payload)
        return qa_ids, path
    return base.dataset.qa_ids, None


def _dataset_without_limit(config: ExperimentConfig) -> DatasetConfig:
    return DatasetConfig(
        qa_path=config.dataset.qa_path,
        mrag_dir=config.dataset.mrag_dir,
        limit=None,
        qa_ids=config.dataset.qa_ids,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
