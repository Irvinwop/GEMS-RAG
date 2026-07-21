from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .analysis import validate_run
from .config import ExperimentConfig, load_experiment_config
from .runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ID = "MUTCD-150-v1.0"
BENCHMARK_SHA256 = "3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2"
BENCHMARK_QUESTION_COUNT = 150
BENCHMARK_DISTRIBUTION = {"T": 60, "TB": 30, "F": 30, "M": 30}
COMPARISON_RETRIEVERS = ("bm25", "graphrag_local", "paperqa2_chunks")
COMPARISON_RETRIEVER_KINDS = {
    "bm25": "bm25",
    "graphrag_local": "external_command",
    "paperqa2_chunks": "external_command",
}
COMPARISON_CONTEXT_MODES = ("injected",)
COMPARISON_TOP_K = 6
COMPARISON_MAX_EVIDENCE_CHARS = 9600


def comparison_contract(
    config: ExperimentConfig,
    *,
    root: Path = PROJECT_ROOT,
    expected_sha256: str = BENCHMARK_SHA256,
    expected_question_count: int = BENCHMARK_QUESTION_COUNT,
    expected_distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    qa_path = _resolve(root, config.dataset.qa_path)
    mrag_dir = _resolve(root, config.dataset.mrag_dir)
    distribution = expected_distribution or BENCHMARK_DISTRIBUTION
    benchmark = benchmark_fingerprint(
        qa_path,
        expected_sha256=expected_sha256,
        expected_question_count=expected_question_count,
        expected_distribution=distribution,
    )
    problems = list(benchmark["problems"])

    retriever_names = [retriever.name for retriever in config.retrievers]
    if retriever_names != list(COMPARISON_RETRIEVERS):
        problems.append(
            "retrievers must be exactly, in order: " + ", ".join(COMPARISON_RETRIEVERS)
        )
    for retriever in config.retrievers:
        expected_kind = COMPARISON_RETRIEVER_KINDS.get(retriever.name)
        if expected_kind is not None and retriever.kind != expected_kind:
            problems.append(
                f"retriever {retriever.name} must use kind {expected_kind}, got {retriever.kind}"
            )
        if retriever.top_k != COMPARISON_TOP_K:
            problems.append(
                f"retriever {retriever.name} must use top_k={COMPARISON_TOP_K}, got {retriever.top_k}"
            )
        if retriever.name in {"graphrag_local", "paperqa2_chunks"}:
            command = retriever.options.get("command")
            check_command = retriever.options.get("check_command")
            if not isinstance(command, list) or not command:
                problems.append(f"retriever {retriever.name} is missing its query command")
            if not isinstance(check_command, list) or not check_command:
                problems.append(f"retriever {retriever.name} is missing its readiness command")

    if tuple(config.context_modes) != COMPARISON_CONTEXT_MODES:
        problems.append("context_modes must be exactly: injected")
    if config.dataset.limit not in {None, BENCHMARK_QUESTION_COUNT}:
        problems.append(
            f"dataset limit must be omitted or {BENCHMARK_QUESTION_COUNT}, got {config.dataset.limit}"
        )
    if config.dataset.qa_ids:
        problems.append("dataset qa_ids must be omitted so all 150 locked questions are run")
    if config.max_evidence_chars != COMPARISON_MAX_EVIDENCE_CHARS:
        problems.append(
            "max_evidence_chars must be "
            f"{COMPARISON_MAX_EVIDENCE_CHARS}, got {config.max_evidence_chars}"
        )
    if not config.models:
        problems.append("at least one answer model is required")
    model_ids = [(model.provider, model.model) for model in config.models]
    if len(set(model_ids)) != len(model_ids):
        problems.append("answer model conditions must be unique")
    if config.grader.provider != "heuristic":
        problems.append("the run-time grader must be heuristic; GPT Pro grades the final bundle")

    manual_path = mrag_dir / "mutcd11theditionr1hl.pdf"
    chunks_path = mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    if not manual_path.is_file():
        problems.append(f"authoritative MUTCD manual is missing: {manual_path}")
    if not chunks_path.is_file():
        problems.append(f"canonical MUTCD chunks are missing: {chunks_path}")

    expected_rows = expected_question_count * len(COMPARISON_RETRIEVERS) * len(config.models)
    return {
        "ok": not problems,
        "status": "ready" if not problems else "blocked",
        "benchmark_id": BENCHMARK_ID,
        "benchmark": benchmark,
        "retrievers": list(COMPARISON_RETRIEVERS),
        "context_modes": list(COMPARISON_CONTEXT_MODES),
        "top_k": COMPARISON_TOP_K,
        "max_evidence_chars": COMPARISON_MAX_EVIDENCE_CHARS,
        "models": [
            {"provider": model.provider, "model": model.model}
            for model in config.models
        ],
        "expected_rows": expected_rows,
        "manual_path": str(manual_path),
        "chunks_path": str(chunks_path),
        "dry_run": config.dry_run,
        "problems": problems,
    }


def require_comparison_contract(config: ExperimentConfig, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    report = comparison_contract(config, root=root)
    if not report["ok"]:
        raise ValueError("invalid MUTCD comparison config: " + "; ".join(report["problems"]))
    return report


def run_comparison(
    config_path: Path,
    *,
    overwrite: bool = False,
    retry_errors: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    contract = require_comparison_contract(config, root=root)
    runs_path = run_experiment(
        config,
        overwrite=overwrite,
        resume=not overwrite and not retry_errors,
        retry_errors=retry_errors,
    )
    validation = validate_run(config, runs_path)
    return {
        "ok": validation["ok"],
        "status": "complete" if validation["ok"] else "needs_retry",
        "run_mode": "overwrite" if overwrite else "retry_errors" if retry_errors else "resume",
        "config": str(config_path.resolve()),
        "runs": str(runs_path.resolve()),
        "contract": contract,
        "validation": validation,
    }


def comparison_status(
    config_path: Path,
    *,
    runs_path: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    contract = comparison_contract(config, root=root)
    candidate = runs_path or config.output_dir / config.name / "runs.jsonl"
    candidate = _resolve(root, candidate)
    if not candidate.is_file():
        return {
            "ok": False,
            "status": "ready_to_run" if contract["ok"] else "blocked",
            "config": str(config_path.resolve()),
            "runs": str(candidate),
            "contract": contract,
            "validation": None,
        }
    validation = validate_run(config, candidate)
    return {
        "ok": contract["ok"] and validation["ok"],
        "status": "complete" if contract["ok"] and validation["ok"] else "needs_retry",
        "config": str(config_path.resolve()),
        "runs": str(candidate),
        "contract": contract,
        "validation": validation,
    }


def benchmark_fingerprint(
    path: Path,
    *,
    expected_sha256: str,
    expected_question_count: int,
    expected_distribution: dict[str, int],
) -> dict[str, Any]:
    problems: list[str] = []
    if not path.is_file():
        return {
            "ok": False,
            "path": str(path),
            "sha256": None,
            "question_count": 0,
            "distribution": {},
            "duplicate_ids": [],
            "invalid_lines": [],
            "problems": [f"locked question file is missing: {path}"],
        }

    digest = _sha256(path)
    if digest != expected_sha256:
        problems.append(
            f"question file SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    ids: list[str] = []
    distribution: dict[str, int] = {}
    invalid_lines: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines.append({"line": line_number, "error": str(exc)})
                continue
            question_id = str(row.get("question_id") or row.get("qa_id") or "").strip()
            question = str(row.get("question") or "").strip()
            if not question_id or not question:
                invalid_lines.append(
                    {"line": line_number, "error": "question_id and question are required"}
                )
                continue
            ids.append(question_id)
            prefix = "TB" if question_id.startswith("TB") else question_id[:1]
            distribution[prefix] = distribution.get(prefix, 0) + 1
    counts: dict[str, int] = {}
    for question_id in ids:
        counts[question_id] = counts.get(question_id, 0) + 1
    duplicate_ids = sorted(question_id for question_id, count in counts.items() if count > 1)
    if len(ids) != expected_question_count:
        problems.append(
            f"question count mismatch: expected {expected_question_count}, got {len(ids)}"
        )
    if distribution != expected_distribution:
        problems.append(
            f"question distribution mismatch: expected {expected_distribution}, got {distribution}"
        )
    if duplicate_ids:
        problems.append(f"duplicate question IDs: {', '.join(duplicate_ids[:10])}")
    if invalid_lines:
        problems.append(f"invalid question records: {len(invalid_lines)}")
    return {
        "ok": not problems,
        "path": str(path),
        "sha256": digest,
        "question_count": len(ids),
        "distribution": distribution,
        "duplicate_ids": duplicate_ids,
        "invalid_lines": invalid_lines,
        "problems": problems,
    }


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
