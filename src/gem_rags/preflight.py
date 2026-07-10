from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, GraderConfig, ModelConfig, RetrieverConfig
from .data import load_qa_items
from .models import (
    KNOWN_MODEL_PROVIDERS,
    LLM_MODEL_PROVIDERS,
    is_placeholder_model_name,
    model_api,
    model_api_key_envs,
    model_backend,
    model_required_package,
)

KNOWN_CONTEXT_MODES = {"injected", "tool_explore", "tool_search", "tool_native"}
KNOWN_RETRIEVER_KINDS = {
    "bm25",
    "hash_vector",
    "qdrant_hash_vector",
    "bm25_graph",
    "oracle",
    "external_placeholder",
    "external_command",
    "self_rag_policy",
    "crag_policy",
    "kg2rag",
    "m3kg_rag",
    "okh_rag",
    "sam_rag",
}
KNOWN_GRADER_PROVIDERS = {"heuristic", *LLM_MODEL_PROVIDERS}
EXTERNAL_CHECK_SCRIPTS = {
    "scripts/query_dpr_index.py",
    "scripts/query_mrag_reference.py",
    "scripts/query_graphrag_index.py",
    "scripts/query_lightrag_index.py",
    "scripts/query_raganything_index.py",
    "scripts/query_hipporag_index.py",
    "scripts/query_visrag_index.py",
    "scripts/query_paperqa_index.py",
    "scripts/query_vector_db.py",
}
ROOT = Path(__file__).resolve().parents[2]


def preflight_config(config: ExperimentConfig, *, check_external: bool = True, timeout_s: int = 30) -> dict[str, Any]:
    dataset = _check_dataset(config)
    retrievers = [_check_retriever(ret, check_external=check_external, timeout_s=timeout_s) for ret in config.retrievers]
    models = [_check_model(model, force_dry_run=config.dry_run) for model in config.models]
    grader = _check_grader(config.grader, force_dry_run=config.dry_run)
    context_modes = [
        {"name": mode, "ok": mode in KNOWN_CONTEXT_MODES, "status": "ready" if mode in KNOWN_CONTEXT_MODES else "unknown"}
        for mode in config.context_modes
    ]
    qa_count = dataset.get("qa_count") or 0
    row_estimate = qa_count * len(config.retrievers) * len(config.context_modes) * len(config.models)
    sections = {
        "dataset": dataset,
        "context_modes": context_modes,
        "retrievers": retrievers,
        "models": models,
        "grader": grader,
    }
    blocking = _collect_blocking(sections)
    return {
        "experiment": config.name,
        "ok": not blocking,
        "status": "ready" if not blocking else "blocked",
        "row_estimate": row_estimate,
        "sections": sections,
        "blocking": blocking,
    }


def _check_dataset(config: ExperimentConfig) -> dict[str, Any]:
    qa_path = config.dataset.qa_path
    mrag_dir = config.dataset.mrag_dir
    chunk_path = mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    figure_path = mrag_dir / "mmrag_cache_v3" / "figures.jsonl"
    graph_path = mrag_dir / "mmrag_cache_v3" / "graph.gpickle"
    report = {
        "qa_path": str(qa_path),
        "qa_found": qa_path.exists(),
        "mrag_dir": str(mrag_dir),
        "mrag_dir_found": mrag_dir.exists(),
        "chunks_found": chunk_path.exists(),
        "figures_found": figure_path.exists(),
        "graph_found": graph_path.exists(),
        "limit": config.dataset.limit,
        "qa_ids": config.dataset.qa_ids,
        "qa_count": 0,
        "status": "ready",
        "problems": [],
    }
    if not qa_path.exists():
        report["problems"].append(f"missing QA file: {qa_path}")
    if not mrag_dir.exists():
        report["problems"].append(f"missing MRAG dir: {mrag_dir}")
    if not chunk_path.exists():
        report["problems"].append(f"missing chunks file: {chunk_path}")
    if not figure_path.exists():
        report["problems"].append(f"missing figures file: {figure_path}")
    if not graph_path.exists():
        report["problems"].append(f"missing graph file: {graph_path}")
    if not report["problems"]:
        try:
            report["qa_count"] = len(load_qa_items(qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids))
        except Exception as exc:
            report["problems"].append(f"failed to load QA file: {exc!r}")
    if report["problems"]:
        report["status"] = "blocked"
    return report


def _check_retriever(config: RetrieverConfig, *, check_external: bool, timeout_s: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": config.name,
        "kind": config.kind,
        "top_k": config.top_k,
        "status": "ready",
        "problems": [],
    }
    if config.kind not in KNOWN_RETRIEVER_KINDS:
        report["status"] = "blocked"
        report["problems"].append(f"unknown retriever kind: {config.kind}")
        return report
    if config.kind == "external_placeholder":
        path = Path(str(config.options.get("path", "")))
        report["path"] = str(path)
        report["path_found"] = path.exists()
        if not path.exists():
            report["status"] = "blocked"
            report["problems"].append(f"missing external placeholder path: {path}")
    if config.kind == "external_command":
        command = config.options.get("command")
        command = _command_parts(command)
        if not command:
            report["status"] = "blocked"
            report["problems"].append("external_command requires options.command")
            return report
        report["command"] = command
        check_command = config.options.get("check_command")
        check_command = _command_parts(check_command)
        check = _external_command_check(
            command,
            check_external=check_external,
            timeout_s=timeout_s,
            check_command=check_command or None,
        )
        report["external_check"] = check
        if check["status"] != "ready":
            report["status"] = check["status"]
            report["problems"].extend(check.get("problems", []))
    if config.kind == "qdrant_hash_vector":
        report["qdrant_path"] = str(config.options.get("path", "data/working/qdrant_hash_vector"))
        report["qdrant_client_installed"] = importlib.util.find_spec("qdrant_client") is not None
        if not report["qdrant_client_installed"]:
            report["status"] = "blocked"
            report["problems"].append("qdrant_client is not installed")
    return report


def _external_command_check(command: list[str], *, check_external: bool, timeout_s: int, check_command: list[str] | None = None) -> dict[str, Any]:
    if check_command:
        if not check_external:
            return {"status": "not_checked", "check_command": check_command, "problems": []}
        return _run_external_check_command(check_command, timeout_s=timeout_s)
    script_idx = next((idx for idx, part in enumerate(command) if _is_known_external_script(str(part))), None)
    if script_idx is None:
        return {"status": "unknown", "problems": ["no known adapter check for external command"]}
    script = str(command[script_idx])
    python = command[script_idx - 1] if script_idx > 0 and _looks_like_python(command[script_idx - 1]) else ".venv/bin/python"
    check_command = [str(python), script, "check"]
    if not check_external:
        return {"status": "not_checked", "check_command": check_command, "problems": []}
    return _run_external_check_command(check_command, timeout_s=timeout_s)


def _command_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list | tuple):
        return [str(part) for part in value]
    return []


def _is_known_external_script(value: str) -> bool:
    return any(value == script or value.endswith(f"/{script}") for script in EXTERNAL_CHECK_SCRIPTS)


def _looks_like_python(value: Any) -> bool:
    return Path(str(value)).name.startswith("python")


def _run_external_check_command(check_command: list[str], *, timeout_s: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            check_command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {"status": "blocked", "check_command": check_command, "problems": [repr(exc)]}
    parsed = _parse_json(completed.stdout)
    status = "ready" if completed.returncode == 0 else "blocked"
    problems = []
    if isinstance(parsed, dict):
        api_key_usable = parsed.get("api_key_usable", parsed.get("api_key_present"))
        if parsed.get("cli_runnable") is True and api_key_usable is False:
            status = "blocked_by_credentials"
        elif parsed.get("missing_or_failed_imports"):
            problems.append(f"import failures: {parsed['missing_or_failed_imports']}")
        elif parsed.get("missing_required_modules"):
            problems.append(f"missing modules: {parsed['missing_required_modules']}")
        elif api_key_usable is False:
            status = "blocked_by_credentials"
        if status == "blocked_by_credentials" and parsed.get("api_key_env"):
            problems.append(f"missing API key env var: {parsed['api_key_env']}")
        if parsed.get("index_ready") is False:
            index_location = parsed.get("index") or parsed.get("working_dir") or parsed.get("save_dir") or parsed.get("embeddings")
            problems.append(f"index not ready: {index_location}" if index_location else "index not ready")
        if not parsed.get("runnable", completed.returncode == 0) and not problems and status == "blocked":
            problems.append(parsed.get("notes") or parsed.get("stderr") or "adapter check failed")
    else:
        problems.append(completed.stderr[-1000:] or completed.stdout[-1000:] or "adapter check failed")
    return {
        "status": status,
        "check_command": check_command,
        "returncode": completed.returncode,
        "stdout_json": parsed,
        "problems": problems,
    }


def _check_model(config: ModelConfig, *, force_dry_run: bool = False) -> dict[str, Any]:
    report = {
        "provider": config.provider,
        "model": config.model,
        "status": "ready",
        "problems": [],
        "api_key_envs": [],
        "missing_api_key_envs": [],
    }
    if config.provider not in KNOWN_MODEL_PROVIDERS:
        report["status"] = "blocked"
        report["problems"].append(f"unknown model provider: {config.provider}")
        return report
    if is_placeholder_model_name(config.model):
        report["status"] = "blocked"
        report["problems"].append(f"unresolved model placeholder: {config.model}")
        return report
    if config.provider == "dry_run":
        return report
    backend = model_backend(config)
    report["backend"] = backend
    report["api"] = model_api(config)
    if force_dry_run:
        report["dry_run"] = True
        return report
    package = model_required_package(config)
    if package and importlib.util.find_spec(package) is None:
        report["status"] = "blocked"
        report["problems"].append(f"{package} package is not installed")
        return report
    envs = model_api_key_envs(config)
    report["api_key_envs"] = envs
    missing = [env for env in envs if env and not os.getenv(env)]
    report["missing_api_key_envs"] = missing
    if missing and not config.options.get("api_key"):
        report["status"] = "blocked_by_credentials"
        report["problems"].append(f"missing API key env vars: {missing}")
    return report


def _check_grader(config: GraderConfig, *, force_dry_run: bool = False) -> dict[str, Any]:
    report = {
        "provider": config.provider,
        "model": config.model,
        "status": "ready",
        "problems": [],
    }
    if config.provider not in KNOWN_GRADER_PROVIDERS:
        report["status"] = "blocked"
        report["problems"].append(f"unknown grader provider: {config.provider}")
        return report
    if config.provider == "heuristic":
        return report
    model_report = _check_model(
        ModelConfig(provider=config.provider, model=config.model, options=config.options),
        force_dry_run=force_dry_run,
    )
    report.update(
        {
            "status": model_report["status"],
            "dry_run": model_report.get("dry_run", False),
            "api_key_envs": model_report.get("api_key_envs", []),
            "missing_api_key_envs": model_report.get("missing_api_key_envs", []),
            "problems": model_report["problems"],
        }
    )
    return report


def _collect_blocking(sections: dict[str, Any]) -> list[dict[str, Any]]:
    blocking: list[dict[str, Any]] = []
    _collect_status("dataset", sections["dataset"], blocking)
    for idx, mode in enumerate(sections["context_modes"]):
        _collect_status(f"context_modes[{idx}]", mode, blocking)
    for idx, retriever in enumerate(sections["retrievers"]):
        _collect_status(f"retrievers[{idx}].{retriever.get('name')}", retriever, blocking)
    for idx, model in enumerate(sections["models"]):
        _collect_status(f"models[{idx}].{model.get('model')}", model, blocking)
    _collect_status("grader", sections["grader"], blocking)
    return blocking


def _collect_status(path: str, item: dict[str, Any], blocking: list[dict[str, Any]]) -> None:
    status = item.get("status")
    if status in {"ready", "not_checked"}:
        return
    blocking.append({"path": path, "status": status, "problems": item.get("problems", [])})


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for idx, char in reversed(list(enumerate(text))):
        if char != "{":
            continue
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
    return None
