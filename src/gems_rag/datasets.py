from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CURATED_GOLD_QA_PATH, DEFAULT_MRAG_DIR, MUTCD150_QA_PATH

DEFAULT_DATASET_ID = "mutcd150"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    qa_path: Path
    mrag_dir: Path
    includes_gold_answers: bool
    includes_gold_references: bool


DATASET_SPECS = (
    DatasetSpec(
        dataset_id="mutcd150",
        label="MUTCD-150",
        qa_path=MUTCD150_QA_PATH,
        mrag_dir=DEFAULT_MRAG_DIR,
        includes_gold_answers=False,
        includes_gold_references=False,
    ),
    DatasetSpec(
        dataset_id="curated49",
        label="Curated gold",
        qa_path=CURATED_GOLD_QA_PATH,
        mrag_dir=DEFAULT_MRAG_DIR,
        includes_gold_answers=True,
        includes_gold_references=True,
    ),
)


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    match = next((spec for spec in DATASET_SPECS if spec.dataset_id == dataset_id), None)
    if match is None:
        known = ", ".join(spec.dataset_id for spec in DATASET_SPECS)
        raise ValueError(f"unknown dataset {dataset_id!r}; choose one of: {known}")
    return match


def dataset_catalog(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    rows = []
    for spec in DATASET_SPECS:
        qa_path = _resolve(root, spec.qa_path)
        mrag_dir = _resolve(root, spec.mrag_dir)
        available = qa_path.is_file() and mrag_dir.is_dir()
        rows.append(
            {
                "id": spec.dataset_id,
                "label": spec.label,
                "qa_path": str(qa_path),
                "mrag_dir": str(mrag_dir),
                "qa_count": _jsonl_count(qa_path),
                "qa_sha256": _sha256(qa_path) if qa_path.is_file() else None,
                "available": available,
                "includes_gold_answers": spec.includes_gold_answers,
                "includes_gold_references": spec.includes_gold_references,
            }
        )
    return rows


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
