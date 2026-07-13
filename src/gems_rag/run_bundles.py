from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .data import load_qa_items
from .grading import RUBRIC_KEYS, normalize_judge_scores

IMAGE_PATH_KEYS = {"figure_image_path", "image_path", "image_paths", "page_image_path"}
SAFE_RUN_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
SECRET_KEYS = {"api_key", "apikey", "authorization", "password", "secret", "access_token", "refresh_token"}


def export_run_bundle(
    runs_path: Path,
    *,
    output_path: Path | None = None,
    qa_path: Path | None = None,
    mode: str = "gpt_pro",
) -> dict[str, Any]:
    if mode not in {"archive", "gpt_pro"}:
        raise ValueError(f"unsupported bundle mode: {mode}")
    runs_path = _runs_file(runs_path).resolve()
    if not runs_path.is_file():
        raise FileNotFoundError(runs_path)
    rows = _read_jsonl(runs_path)
    inferred_qa = qa_path or _infer_qa_path(runs_path.parent)
    qa_by_id = {}
    if inferred_qa is not None and inferred_qa.is_file():
        qa_by_id = {item.qa_id: item for item in load_qa_items(inferred_qa)}
    if mode == "gpt_pro" and not qa_by_id:
        raise ValueError("GPT Pro bundles require --qa-path or a materialized_config.json with dataset.qa_path")

    output_path = (output_path or runs_path.parent / f"{runs_path.parent.name}-{mode}.zip").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path == runs_path:
        raise ValueError("bundle output must differ from runs input")

    with tempfile.TemporaryDirectory(prefix="gems-rag-bundle-") as td:
        stage = Path(td)
        tasks, images = _build_tasks(rows, qa_by_id, stage)
        task_path = stage / "grading_tasks.jsonl"
        _write_jsonl(task_path, tasks)
        template_path = stage / "grades.template.jsonl"
        _write_jsonl(template_path, [_grade_template(task["row_id"]) for task in tasks])
        instructions_path = stage / "GRADING.md"
        instructions_path.write_text(_grading_instructions(), encoding="utf-8")
        _stage_run_artifacts(runs_path.parent, stage / "run")

        manifest = {
            "schema_version": 1,
            "bundle_mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "runs_path": str(runs_path),
            "qa_path": str(inferred_qa.resolve()) if inferred_qa and inferred_qa.is_file() else None,
            "rows": len(rows),
            "grading_tasks": len(tasks),
            "evidence_images": images,
            "rubric_keys": RUBRIC_KEYS,
            "files": {},
        }
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(stage))] = _sha256(path)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_zip(stage, output_path)

    return {
        "status": "complete",
        "mode": mode,
        "output": str(output_path),
        "rows": len(rows),
        "grading_tasks": len(tasks),
        "evidence_images": images,
        "bytes": output_path.stat().st_size,
    }


def import_pro_grades(
    runs_path: Path,
    grades_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    runs_path = _runs_file(runs_path).resolve()
    rows = _read_jsonl(runs_path)
    grades = _read_grades(grades_path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates = []
    for grade in grades:
        row_id = str(grade.get("row_id") or "")
        if not row_id:
            raise ValueError("every grade row requires row_id")
        if row_id in by_id:
            duplicates.append(row_id)
        by_id[row_id] = grade
    if duplicates:
        raise ValueError(f"duplicate grade row_id values: {', '.join(sorted(set(duplicates)))}")

    known_ids = {run_row_id(row) for row in rows}
    unknown = sorted(set(by_id) - known_ids)
    if unknown:
        raise ValueError(f"grade rows do not match this run: {', '.join(unknown[:5])}")

    output_path = (output_path or runs_path.parent / "gpt-pro-graded-runs.jsonl").resolve()
    if output_path == runs_path:
        raise ValueError("output path must differ from runs input")
    updated_rows = []
    imported = 0
    for row in rows:
        row_id = run_row_id(row)
        grade = by_id.get(row_id)
        if grade is None:
            updated_rows.append(row)
            continue
        updated = deepcopy(row)
        config = updated.setdefault("config", {})
        config["grader_provider"] = "gpt_pro"
        config["grader"] = str(grade.get("grader") or "gpt-pro-manual")
        updated["judge_scores"] = normalize_judge_scores(grade)
        updated["judge_confidence"] = _float_or_none(grade.get("judge_confidence"))
        updated["judge_explanation"] = grade.get("judge_explanation")
        if isinstance(grade.get("figure_metrics"), dict):
            updated["figure_metrics"] = grade["figure_metrics"]
        updated["grader_raw"] = {
            "external_subscription_grading": True,
            "imported_at": datetime.now(UTC).isoformat(),
            "row_id": row_id,
            "raw": redact_secrets(grade),
        }
        updated["judge_error"] = None
        updated_rows.append(updated)
        imported += 1
    _write_jsonl(output_path, updated_rows)
    missing = len(rows) - imported
    return {
        "ok": missing == 0,
        "status": "complete" if missing == 0 else "partial",
        "runs": str(runs_path),
        "grades": str(grades_path.resolve()),
        "output": str(output_path),
        "rows": len(rows),
        "grades_imported": imported,
        "grades_missing": missing,
    }


def run_row_id(row: dict[str, Any]) -> str:
    identity = {
        "qa_id": row.get("qa_id"),
        "config": row.get("config", {}),
        "run_id": (row.get("run") or {}).get("run_id"),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row.get("qa_id") or "row"))[:48]
    return f"{prefix}-{digest}"


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if _is_secret_key(key) else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def _build_tasks(rows: list[dict[str, Any]], qa_by_id: dict[str, Any], stage: Path) -> tuple[list[dict[str, Any]], int]:
    tasks = []
    copied_images: dict[str, str] = {}
    for row in rows:
        qa = qa_by_id.get(str(row.get("qa_id")))
        evidence = deepcopy(row.get("evidence") or [])
        evidence = _copy_evidence_images(evidence, stage, copied_images)
        tasks.append(
            redact_secrets(
                {
                    "row_id": run_row_id(row),
                    "qa_id": row.get("qa_id"),
                    "question": row.get("question") or (qa.question if qa else None),
                    "question_type": row.get("question_type") or (qa.question_type if qa else None),
                    "expected_refusal": row.get("expected_refusal") if "expected_refusal" in row else (qa.expected_refusal if qa else False),
                    "gold_answer": qa.gold_answer if qa else {},
                    "gold_references": qa.references if qa else [],
                    "gold_figures": qa.gold_figures if qa else [],
                    "rag_config": row.get("config", {}),
                    "rag_answer": row.get("answer", ""),
                    "model_error": row.get("model_error"),
                    "retrieval_error": row.get("retrieval_error"),
                    "retrieved_evidence": evidence,
                }
            )
        )
    return tasks, len(copied_images)


def _copy_evidence_images(evidence: list[dict[str, Any]], stage: Path, copied: dict[str, str]) -> list[dict[str, Any]]:
    for item in evidence:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = _rewrite_image_values(metadata, stage, copied)
    return evidence


def _rewrite_image_values(value: Any, stage: Path, copied: dict[str, str], key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _rewrite_image_values(item, stage, copied, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_image_values(item, stage, copied, key) for item in value]
    if key not in IMAGE_PATH_KEYS or not isinstance(value, str):
        return value
    source = Path(value).expanduser()
    if not source.is_file():
        return value
    digest = _sha256(source)
    relative = copied.get(digest)
    if relative is None:
        suffix = source.suffix.lower() if source.suffix else ".bin"
        relative = f"evidence_images/{digest[:20]}{suffix}"
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        copied[digest] = relative
    return relative


def _grade_template(row_id: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "grader": "gpt-pro-manual",
        "judge_scores": {key: {"score": None, "note": ""} for key in RUBRIC_KEYS},
        "judge_confidence": None,
        "judge_explanation": "",
        "figure_metrics": {
            "figure_recall": None,
            "figure_precision": None,
            "n_gold_figures": 0,
            "n_rag_figures": 0,
            "n_intersection": 0,
        },
    }


def _grading_instructions() -> str:
    keys = ", ".join(f"`{key}`" for key in RUBRIC_KEYS)
    return f"""# GEMS-RAG GPT Pro grading bundle

Grade each JSON object in `grading_tasks.jsonl` against its gold answer, references, and retrieved evidence. Open files under `evidence_images/` when a task references them.

Return one compact JSON object per line in a file named `grades.jsonl`. Start from `grades.template.jsonl`; preserve every `row_id` exactly. Score each rubric from 0 to 5, or `null` when it does not apply. Required rubric keys: {keys}.

Judge the answer, not merely retrieval quality. Penalize unsupported claims, invalid citations, and unfaithful quotations. Use `refusal_appropriateness` for out-of-scope questions and the figure rubrics only when figures are relevant. Do not include Markdown fences or prose outside the JSONL rows.
"""


def _stage_run_artifacts(run_dir: Path, target: Path) -> None:
    for source in sorted(run_dir.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in SAFE_RUN_SUFFIXES:
            continue
        relative = source.relative_to(run_dir)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".json":
            try:
                payload = redact_secrets(json.loads(source.read_text(encoding="utf-8")))
                destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if source.suffix.lower() == ".jsonl":
            try:
                _write_jsonl(destination, [redact_secrets(row) for row in _read_jsonl(source)])
                continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        text = source.read_text(encoding="utf-8", errors="replace")
        destination.write_text(_redact_text(text), encoding="utf-8")


def _redact_text(text: str) -> str:
    return re.sub(
        r"(?im)\b(OPENAI_API_KEY|ANTHROPIC_API_KEY|XAI_API_KEY|DASHSCOPE_API_KEY|GRAPHRAG_API_KEY)\s*[=:]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )


def _infer_qa_path(run_dir: Path) -> Path | None:
    config_path = run_dir / "materialized_config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = (payload.get("dataset") or {}).get("qa_path")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[2]
    return root / path


def _read_grades(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".zip":
        return _read_jsonl(path)
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if Path(name).name == "grades.jsonl"]
        if not candidates:
            raise ValueError("ZIP does not contain grades.jsonl")
        lines = archive.read(candidates[0]).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _runs_file(path: Path) -> Path:
    return path / "runs.jsonl" if path.is_dir() else path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_zip(stage: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(stage)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    return normalized in SECRET_KEYS or normalized.endswith("_api_key")


def _float_or_none(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
