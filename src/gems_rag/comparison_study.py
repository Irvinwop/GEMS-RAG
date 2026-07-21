from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analysis import validate_run
from .config import ExperimentConfig, load_experiment_config
from .run_bundles import export_run_bundle, redact_secrets, run_row_id
from .runner import run_experiment
from .runtime_validity import operational_row_problems


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
DEFAULT_GRADER_SPEC = PROJECT_ROOT / "docs" / "MUTCD_RAG_EVALUATION_SPECIFICATION.md"


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
        if tuple(retriever.context_modes) != COMPARISON_CONTEXT_MODES:
            problems.append(
                f"retriever {retriever.name} context_modes must be exactly: injected"
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
    output_path: Path | None = None,
    grader_spec_path: Path = DEFAULT_GRADER_SPEC,
    create_bundle: bool = True,
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
    validation = validate_comparison_run(config, runs_path=runs_path, root=root)
    bundle = None
    if validation["ok"] and create_bundle:
        bundle = bundle_comparison(
            config_path,
            runs_path=runs_path,
            output_path=output_path,
            grader_spec_path=grader_spec_path,
            root=root,
        )
    return {
        "ok": validation["ok"],
        "status": _comparison_status(validation),
        "run_mode": "overwrite" if overwrite else "retry_errors" if retry_errors else "resume",
        "config": str(config_path.resolve()),
        "runs": str(runs_path.resolve()),
        "contract": contract,
        "validation": validation,
        "bundle": bundle,
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
    validation = validate_comparison_run(config, runs_path=candidate, root=root)
    return {
        "ok": contract["ok"] and validation["ok"],
        "status": _comparison_status(validation),
        "config": str(config_path.resolve()),
        "runs": str(candidate),
        "contract": contract,
        "validation": validation,
    }


def validate_comparison_run(
    config: ExperimentConfig,
    *,
    runs_path: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    contract = comparison_contract(config, root=root)
    candidate = runs_path or config.output_dir / config.name / "runs.jsonl"
    candidate = _resolve(root.resolve(), candidate)
    base = validate_run(config, candidate)
    rows, invalid_lines = _read_jsonl_lenient(candidate)
    invalid_rows = []
    for index, row in enumerate(rows, 1):
        problems = operational_row_problems(row)
        if problems:
            invalid_rows.append(
                {
                    "line": index,
                    "qa_id": row.get("qa_id"),
                    "condition": _condition(row),
                    "problems": problems,
                }
            )
    problems = list(base["problems"])
    if not contract["ok"]:
        problems.extend(contract["problems"])
    if config.dry_run or any(model.provider == "dry_run" for model in config.models):
        problems.append("dry-run answer rows cannot be packaged as final study results")
    if invalid_rows:
        problems.append(f"operationally invalid rows: {len(invalid_rows)}")
    if invalid_lines and not base.get("invalid_json_lines"):
        problems.append(f"invalid JSON lines: {len(invalid_lines)}")
    ok = base["ok"] and contract["ok"] and not invalid_rows and not invalid_lines and not (
        config.dry_run or any(model.provider == "dry_run" for model in config.models)
    )
    return {
        **base,
        "ok": ok,
        "status": "ready" if ok else "failed",
        "study_contract_ok": contract["ok"],
        "dry_run_final": config.dry_run or any(model.provider == "dry_run" for model in config.models),
        "operational_invalid_rows": len(invalid_rows),
        "operational_invalid_sample": invalid_rows[:20],
        "operational_invalid_json_lines": len(invalid_lines),
        "problems": list(dict.fromkeys(problems)),
    }


def bundle_comparison(
    config_path: Path,
    *,
    runs_path: Path | None = None,
    output_path: Path | None = None,
    grader_spec_path: Path = DEFAULT_GRADER_SPEC,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    candidate = runs_path or config.output_dir / config.name / "runs.jsonl"
    candidate = _resolve(root.resolve(), candidate)
    validation = validate_comparison_run(config, runs_path=candidate, root=root)
    if not validation["ok"]:
        raise ValueError(
            "final comparison bundle blocked: " + "; ".join(validation["problems"])
        )
    artifacts = _write_study_artifacts(
        config,
        config_path=config_path,
        runs_path=candidate,
        validation=validation,
        grader_spec_path=grader_spec_path,
        root=root,
    )
    output = output_path or candidate.parent / f"{config.name}-gpt-pro.zip"
    bundle = export_run_bundle(
        candidate,
        output_path=output,
        qa_path=_resolve(root.resolve(), config.dataset.qa_path),
        mode="gpt_pro",
        grader_spec_path=grader_spec_path,
    )
    return {**bundle, "validation": validation, "study_artifacts": artifacts}


def _write_study_artifacts(
    config: ExperimentConfig,
    *,
    config_path: Path,
    runs_path: Path,
    validation: dict[str, Any],
    grader_spec_path: Path,
    root: Path,
) -> list[str]:
    rows, invalid_lines = _read_jsonl_lenient(runs_path)
    if invalid_lines:
        raise ValueError("cannot write canonical study artifacts from invalid JSONL")
    run_dir = runs_path.parent
    answers = []
    retrieval = []
    for row in rows:
        identity = {
            "row_id": run_row_id(row),
            "qa_id": row.get("qa_id"),
            "question": row.get("question"),
            "condition": _condition(row),
            "run": row.get("run"),
        }
        answers.append(
            {
                **identity,
                "run_status": row.get("run_status"),
                "answer": row.get("answer"),
                "serialized_return": row.get("serialized_return"),
                "model_raw": row.get("model_raw"),
                "model_error": row.get("model_error"),
                "latency_s": row.get("latency_s"),
            }
        )
        retrieval.append(
            {
                **identity,
                "retrieval_error": row.get("retrieval_error"),
                "evidence": row.get("evidence"),
                "retrieval_debug": row.get("retrieval_debug"),
            }
        )

    _write_jsonl_atomic(run_dir / "canonical_answers.jsonl", answers)
    _write_jsonl_atomic(run_dir / "canonical_retrieval.jsonl", retrieval)
    _write_jsonl_atomic(run_dir / "canonical_errors.jsonl", [])
    provenance = _merge_provenance(run_dir, rows)
    provenance_path = run_dir / "merge_provenance.csv"
    if provenance:
        _write_csv_atomic(provenance_path, provenance)
    else:
        provenance_path.unlink(missing_ok=True)

    retry_archives = (
        [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": _sha256(path),
            }
            for path in sorted((run_dir / "retry_history").glob("*.jsonl"))
        ]
        if (run_dir / "retry_history").is_dir()
        else []
    )
    artifacts = [
        "canonical_answers.jsonl",
        "canonical_retrieval.jsonl",
        "canonical_errors.jsonl",
        *(["merge_provenance.csv"] if provenance else []),
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_id": BENCHMARK_ID,
        "benchmark_question_sha256": BENCHMARK_SHA256,
        "benchmark_questions": BENCHMARK_QUESTION_COUNT,
        "retrievers": list(COMPARISON_RETRIEVERS),
        "context_modes": list(COMPARISON_CONTEXT_MODES),
        "top_k": COMPARISON_TOP_K,
        "max_evidence_chars": COMPARISON_MAX_EVIDENCE_CHARS,
        "expected_rows": validation["expected_rows"],
        "canonical_rows": len(rows),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "runs_path": str(runs_path),
        "runs_sha256": _sha256(runs_path),
        "grader_specification": {
            "path": str(grader_spec_path.resolve()),
            "sha256": _sha256(grader_spec_path),
        },
        "models": [
            {
                "provider": model.provider,
                "model": model.model,
                "options": redact_secrets(model.options),
            }
            for model in config.models
        ],
        "validation": validation,
        "canonical_artifacts": artifacts,
        "retry_history": retry_archives,
        "source_authority": {
            "manual": str(
                _resolve(root.resolve(), config.dataset.mrag_dir)
                / "mutcd11theditionr1hl.pdf"
            ),
            "evaluator_annotations_in_bundle": False,
            "note": (
                "The locked runtime source is question-only. GPT Pro must apply the attached "
                "evaluation specification against the included MUTCD manual; no generated upstream "
                "answers are treated as gold."
            ),
        },
    }
    _write_json_atomic(run_dir / "study_manifest.json", manifest)
    return ["study_manifest.json", *artifacts]


def _merge_provenance(run_dir: Path, canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retry_dir = run_dir / "retry_history"
    if not retry_dir.is_dir():
        return []
    canonical = {_condition_key(row): row for row in canonical_rows}
    replacements: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for archive in sorted(retry_dir.glob("*.jsonl")):
        archived_rows, _invalid = _read_jsonl_lenient(archive)
        for row in archived_rows:
            key = _condition_key(row)
            reasons = operational_row_problems(
                row,
                require_serialized_return=False,
                require_run_status=False,
            )
            if not reasons or key not in canonical or key in replacements:
                continue
            condition = _condition(row)
            replacements[key] = {
                "qa_id": condition["qa_id"],
                "retriever": condition["retriever"],
                "context_mode": condition["context_mode"],
                "model_provider": condition["model_provider"],
                "model": condition["model"],
                "replaced_source": str(archive.relative_to(run_dir)),
                "selected_source": "runs.jsonl",
                "replacement_reason": "; ".join(reasons),
                "selected_run_id": str((canonical[key].get("run") or {}).get("run_id") or ""),
            }
    return list(replacements.values())


def _comparison_status(validation: dict[str, Any]) -> str:
    if validation["ok"]:
        return "complete"
    if validation.get("dry_run_final"):
        return "needs_real_run"
    return "needs_retry"


def _condition(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config") or {}
    return {
        "qa_id": str(row.get("qa_id") or ""),
        "retriever": str(config.get("retriever") or ""),
        "context_mode": str(config.get("context_mode") or ""),
        "model_provider": str(config.get("model_provider") or ""),
        "model": str(config.get("model") or ""),
    }


def _condition_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    condition = _condition(row)
    return tuple(
        condition[key]
        for key in ["qa_id", "retriever", "context_mode", "model_provider", "model"]
    )


def _read_jsonl_lenient(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        return [], [{"line": None, "error": f"missing run file: {path}"}]
    rows = []
    invalid = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(value, dict):
                invalid.append({"line": line_number, "error": "row is not a JSON object"})
                continue
            rows.append(value)
    return rows, invalid


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
