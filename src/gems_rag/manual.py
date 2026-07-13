from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MRAG_DIR = Path("data/extracted/MRAG-20260708T114057Z-3/MRAG")
DEFAULT_MANUSCRIPT_CATALOG = Path("configs/manuscript-rags.json")
DEFAULT_RETRIEVER_CATALOG = Path("configs/retriever-catalog.example.json")

NATIVE_INGESTION: dict[str, dict[str, str]] = {
    "megarag": {
        "mode": "native_pdf",
        "label": "Raw PDF to MegaRAG MMKG and page-image indexes",
    },
    "visrag": {
        "mode": "native_page_render",
        "label": "PDF page renders encoded by VisRAG",
    },
    "raganything": {
        "mode": "native_pdf",
        "label": "Raw PDF parsed by the RAG-Anything document pipeline",
    },
    "paperqa2": {
        "mode": "native_pdf",
        "label": "Raw PDF parsed by PaperQA2",
    },
}


def manual_status(
    *,
    root: Path = ROOT,
    mrag_dir: Path = DEFAULT_MRAG_DIR,
    manuscript_catalog: Path = DEFAULT_MANUSCRIPT_CATALOG,
    retriever_catalog: Path = DEFAULT_RETRIEVER_CATALOG,
) -> dict[str, Any]:
    root = root.resolve()
    mrag_dir = _resolve(root, mrag_dir)
    pdf_path = mrag_dir / "mutcd11theditionr1hl.pdf"
    cache_dir = mrag_dir / "mmrag_cache_v3"
    page_images_dir = mrag_dir / "page_images"
    figures_dir = mrag_dir / "figures"
    shared_dir = root / "data" / "working" / "mrag_corpus"
    shared_manifest = _read_json(shared_dir / "manifest.json")
    pdf_info = _pdf_info(pdf_path)
    page_images = _file_count(page_images_dir, {".png", ".jpg", ".jpeg", ".webp"})
    figure_files = _file_count(figures_dir, {".png", ".jpg", ".jpeg", ".webp"})
    raw_chunks = _jsonl_count(cache_dir / "chunks.jsonl")
    canonical_chunks = _jsonl_count(shared_dir / "chunks.jsonl")
    figure_records = _jsonl_count(cache_dir / "figures.jsonl")
    pdf_pages = _int_or_none(pdf_info.get("Pages"))

    checks = [
        _check("manual_pdf", pdf_path.is_file() and _looks_like_pdf(pdf_path), str(pdf_path)),
        _check("manual_pages", pdf_pages is not None and pdf_pages > 0, f"{pdf_pages or 0} PDF pages"),
        _check(
            "page_renders",
            pdf_pages is not None and page_images == pdf_pages,
            f"{page_images} renders for {pdf_pages or 0} PDF pages",
        ),
        _check("raw_chunks", raw_chunks > 0, f"{raw_chunks} extracted chunks"),
        _check("canonical_chunks", canonical_chunks > 0, f"{canonical_chunks} shared chunks"),
        _check("figures", figure_records > 0 and figure_files > 0, f"{figure_records} records, {figure_files} files"),
        _check("knowledge_graph", (cache_dir / "graph.gpickle").is_file(), str(cache_dir / "graph.gpickle")),
        _check("gold_qa", (mrag_dir / "eval" / "gold_qa.jsonl").is_file(), str(mrag_dir / "eval" / "gold_qa.jsonl")),
    ]
    ingestion = ingestion_matrix(
        root=root,
        manuscript_catalog=manuscript_catalog,
        retriever_catalog=retriever_catalog,
    )
    return {
        "schema_version": 1,
        "status": "ready" if all(check["ok"] for check in checks) else "incomplete",
        "manual": {
            "title": pdf_info.get("Title") or "MUTCD 11th Edition with Revision 1 Incorporated",
            "author": pdf_info.get("Author") or "Federal Highway Administration",
            "path": str(pdf_path),
            "sha256": _sha256(pdf_path) if pdf_path.is_file() else None,
            "bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
            "pages": pdf_pages,
            "pdf_version": pdf_info.get("PDF version"),
        },
        "artifacts": {
            "raw_chunks": raw_chunks,
            "canonical_chunks": canonical_chunks,
            "canonicalization": shared_manifest.get("chunk_canonicalization", {}),
            "page_images": page_images,
            "figure_records": figure_records,
            "figure_files": figure_files,
            "graph": str(cache_dir / "graph.gpickle"),
            "gold_qa": str(mrag_dir / "eval" / "gold_qa.jsonl"),
            "shared_corpus_manifest": str(shared_dir / "manifest.json"),
        },
        "checks": checks,
        "ingestion": ingestion,
    }


def ingestion_matrix(
    *,
    root: Path = ROOT,
    manuscript_catalog: Path = DEFAULT_MANUSCRIPT_CATALOG,
    retriever_catalog: Path = DEFAULT_RETRIEVER_CATALOG,
) -> dict[str, Any]:
    manuscript_payload = _read_json(_resolve(root, manuscript_catalog))
    retriever_payload = _read_json(_resolve(root, retriever_catalog))
    retrievers = {row["name"]: row for row in retriever_payload.get("retrievers", [])}
    rows: list[dict[str, Any]] = []
    native_method_ids: set[str] = set()
    for method in manuscript_payload.get("entries", []):
        if not method.get("coverage_required", True):
            continue
        method_retrievers = []
        for name in method.get("retrievers", []):
            retriever = retrievers.get(name, {})
            family = str(retriever.get("family") or retriever.get("kind") or "unknown")
            native = NATIVE_INGESTION.get(family)
            if native:
                native_method_ids.add(method["method_id"])
            method_retrievers.append(
                {
                    "name": name,
                    "family": family,
                    "shared_corpus": True,
                    "native": native,
                    "source": "verified MUTCD PDF derivative" if native is None else native["label"],
                }
            )
        rows.append(
            {
                "method_id": method["method_id"],
                "label": method["label"],
                "retrievers": method_retrievers,
                "manual_lineage": "verified",
            }
        )
    return {
        "default": "shared_corpus",
        "modes": {
            "shared_corpus": "One canonical manual-derived corpus for controlled cross-method comparison.",
            "native_pdf": "Use an upstream parser or visual PDF path when that research implementation supports it.",
        },
        "method_count": len(rows),
        "native_method_count": len(native_method_ids),
        "native_families": sorted(NATIVE_INGESTION),
        "methods": rows,
    }


def write_manual_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_count(path: Path, suffixes: set[str]) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def _pdf_info(path: Path) -> dict[str, str]:
    if not path.is_file() or shutil.which("pdfinfo") is None:
        return {}
    completed = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=False, timeout=30)
    if completed.returncode != 0:
        return {}
    result = {}
    for line in completed.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}
