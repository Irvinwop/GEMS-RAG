#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_chunks

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "visrag"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260708T114057Z-3" / "MRAG"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "visrag_index"
DEFAULT_MANIFEST = DEFAULT_WORKING_DIR / "visual_manifest.jsonl"
DEFAULT_EMBEDDINGS = DEFAULT_WORKING_DIR / "embeddings.npy"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "visrag" / "bin" / "python"
DEFAULT_MODEL = "openbmb/VisRAG-Ret"
INSTRUCTION = "Represent this query for retrieving relevant documents: "
REQUIRED_MODULES = ["torch", "transformers", "PIL", "numpy"]
EVIDENCE_KINDS = {"page", "figure"}


def main() -> int:
    args = _parse_args()
    if args.command in {"check", "index", "query"}:
        reexec_code = _maybe_reexec(args.python)
        if reexec_code is not None:
            return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["runnable"] else 2
    if args.command == "prepare":
        report = prepare_manifest(args.mrag_dir, args.manifest, scope=args.scope, limit=args.limit)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["records"] else 2
    if args.command == "index":
        return _index(args)
    if args.command == "query":
        return _query(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, index, or query VisRAG-Ret over MRAG page/figure images.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="Path to cloned OpenBMB VisRAG repository.")
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR, help="Extracted MRAG directory.")
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR, help="Ignored VisRAG working directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Prepared visual manifest JSONL.")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS, help="Numpy embedding matrix created by index.")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.getenv("VISRAG_PYTHON", str(DEFAULT_ENV_PYTHON))),
        help="Optional isolated Python with VisRAG dependencies. Defaults to data/working/venvs/visrag/bin/python when present.",
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or any torch device string.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model weights from Hugging Face.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the VisRAG adapter has dependencies, manifest, and embeddings.")

    prepare = sub.add_parser("prepare", help="Build a local visual manifest from extracted MRAG images.")
    prepare.add_argument("--scope", choices=["pages", "figures", "both"], default="pages")
    prepare.add_argument("--limit", type=int)

    index = sub.add_parser("index", help="Encode manifest images with VisRAG-Ret and save embeddings.")
    index.add_argument("--batch-size", type=int, default=4)
    index.add_argument("--limit", type=int, help="Encode only the first N manifest rows.")

    query = sub.add_parser("query", help="Query the saved VisRAG-Ret embedding index.")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    query.add_argument("--json", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def prepare_manifest(mrag_dir: Path, manifest: Path, *, scope: str = "pages", limit: int | None = None) -> dict[str, Any]:
    records = list(iter_visual_records(mrag_dir, scope=scope))
    if limit is not None:
        records = records[:limit]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "prepared": True,
        "mrag_dir": str(mrag_dir),
        "manifest": str(manifest),
        "scope": scope,
        "records": len(records),
        "pages": sum(1 for record in records if record["kind"] == "page"),
        "figures": sum(1 for record in records if record["kind"] == "figure"),
    }


def iter_visual_records(mrag_dir: Path, *, scope: str = "pages") -> Iterable[dict[str, Any]]:
    include_pages = scope in {"pages", "both"}
    include_figures = scope in {"figures", "both"}
    if include_pages:
        yield from _page_records(mrag_dir)
    if include_figures:
        yield from _figure_records(mrag_dir)


def _page_records(mrag_dir: Path) -> Iterable[dict[str, Any]]:
    page_dir = mrag_dir / "page_images"
    chunks_by_page = _chunks_by_page(mrag_dir)
    for path in sorted(page_dir.glob("page_*.png")):
        page_pdf = _page_number(path)
        chunks = chunks_by_page.get(page_pdf, [])
        section_ids = sorted({str(chunk.get("section_id")) for chunk in chunks if chunk.get("section_id")})
        chunk_ids = [str(chunk.get("chunk_id")) for chunk in chunks[:12] if chunk.get("chunk_id")]
        page_printed = sorted({str(chunk.get("page_printed")) for chunk in chunks if chunk.get("page_printed")})
        yield {
            "id": f"page:{page_pdf:04d}",
            "kind": "page",
            "image_path": str(path.resolve()),
            "text": _page_text(page_pdf, page_printed, section_ids, len(chunks)),
            "metadata": {
                "page_pdf": page_pdf,
                "page_printed": page_printed[:5],
                "section_ids": section_ids[:20],
                "chunk_ids_sample": chunk_ids,
                "chunk_count": len(chunks),
                "source": "mrag_page_images",
            },
        }


def _figure_records(mrag_dir: Path) -> Iterable[dict[str, Any]]:
    figures_path = mrag_dir / "mmrag_cache_v3" / "figures.jsonl"
    figures_dir = mrag_dir / "figures"
    if not figures_path.exists():
        return
    for record in _read_jsonl(figures_path):
        image_path = _local_figure_path(figures_dir, record)
        if image_path is None:
            continue
        figure_id = str(record.get("figure_id") or image_path.stem)
        yield {
            "id": f"figure:{figure_id}",
            "kind": "figure",
            "image_path": str(image_path.resolve()),
            "text": _figure_text(record),
            "metadata": {
                "figure_id": figure_id,
                "figure_kind": record.get("kind"),
                "canonical_id": record.get("canonical_id"),
                "page_pdf": record.get("page_pdf"),
                "page_printed": record.get("page_printed"),
                "caption": record.get("caption"),
                "title": record.get("title"),
                "sign_codes_depicted": record.get("sign_codes_depicted", []),
                "referenced_in_chunks": record.get("referenced_in_chunks", []),
                "source": "mrag_figures",
            },
        }


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    import_errors = _import_errors(REQUIRED_MODULES)
    manifest_rows = _count_jsonl(args.manifest) if args.manifest.exists() else 0
    embedding_rows = _embedding_rows(args.embeddings) if args.embeddings.exists() and not import_errors.get("numpy") else None
    repo_found = args.repo.exists()
    source_found = (args.repo / "src").exists()
    index_ready = bool(args.embeddings.exists() and manifest_rows and embedding_rows == manifest_rows)
    return {
        "runnable": repo_found and source_found and not import_errors and index_ready,
        "environment_ready": repo_found and source_found and not import_errors,
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "repo": str(args.repo),
        "repo_found": repo_found,
        "source_found": source_found,
        "mrag_dir": str(args.mrag_dir),
        "mrag_dir_found": args.mrag_dir.exists(),
        "manifest": str(args.manifest),
        "manifest_found": args.manifest.exists(),
        "manifest_rows": manifest_rows,
        "embeddings": str(args.embeddings),
        "embeddings_found": args.embeddings.exists(),
        "embedding_rows": embedding_rows,
        "index_ready": index_ready,
        "model_name_or_path": args.model_name_or_path,
        "missing_or_failed_imports": import_errors,
        "notes": "VisRAG-Ret indexing follows the upstream AutoModel/AutoTokenizer weighted-mean-pooling recipe and requires local model dependencies plus saved image embeddings before query is runnable.",
    }


def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    import_errors = report["missing_or_failed_imports"]
    if import_errors:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    records = _read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(json.dumps({"error": "empty_manifest", "manifest": str(args.manifest)}, indent=2), file=sys.stderr)
        return 2
    try:
        model, tokenizer, torch, np = _load_model(args)
        embeddings = []
        for batch in _batches(records, max(args.batch_size, 1)):
            images = [_open_image(record["image_path"]) for record in batch]
            embeddings.append(_encode(model, tokenizer, torch, np, images))
        matrix = np.concatenate(embeddings, axis=0)
    except Exception as exc:
        print(json.dumps({"error": "visrag_index_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings, matrix)
    print(json.dumps({"indexed": True, "records": len(records), "embeddings": str(args.embeddings)}, indent=2))
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if report["missing_or_failed_imports"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["index_ready"]:
        print(json.dumps({"error": "index_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    records = _read_jsonl(args.manifest)
    try:
        model, tokenizer, torch, np = _load_model(args)
        query_embedding = _encode(model, tokenizer, torch, np, [INSTRUCTION + args.question])[0]
        embeddings = np.load(args.embeddings)
        scores = embeddings @ query_embedding.T
        order = np.argsort(-scores)[: args.top_k]
    except Exception as exc:
        print(json.dumps({"error": "visrag_query_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    contexts = [_context_from_record(records[int(idx)], float(scores[int(idx)])) for idx in order]
    payload = {
        "question": args.question,
        "model_name_or_path": args.model_name_or_path,
        "contexts": contexts,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for context in contexts:
            print(f"{context['score']:.4f}\t{context['name']}\t{context['image_path']}")
    return 0


def _load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo / "src"))
    sys.path.insert(0, str(args.repo / "timm_modified"))
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    dtype = _torch_dtype(torch, args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(args.model_name_or_path, **model_kwargs)
    device = _device(torch, args.device)
    model.to(device)
    model.eval()
    return model, tokenizer, torch, np


def _maybe_reexec(python: Path) -> int | None:
    if not python.exists():
        return None
    try:
        if python.resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        return None
    completed = subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT, check=False)
    return completed.returncode


def _encode(model: Any, tokenizer: Any, torch: Any, np: Any, text_or_image_list: list[Any]) -> Any:
    device = next(model.parameters()).device
    with torch.no_grad():
        if isinstance(text_or_image_list[0], str):
            inputs = {"text": text_or_image_list, "image": [None] * len(text_or_image_list), "tokenizer": tokenizer}
        else:
            inputs = {"text": [""] * len(text_or_image_list), "image": text_or_image_list, "tokenizer": tokenizer}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        attention_mask = outputs.attention_mask.to(device)
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        summed = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        denom = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = summed / denom
        reps = torch.nn.functional.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return np.asarray(reps)


def _context_from_record(record: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "name": record["id"],
        "kind": record["kind"] if record["kind"] in EVIDENCE_KINDS else "tool_trace",
        "text": record.get("text") or record["id"],
        "score": score,
        "image_path": record.get("image_path"),
        "metadata": dict(record.get("metadata") or {}),
    }


def _chunks_by_page(mrag_dir: Path) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not (mrag_dir / "mmrag_cache_v3" / "chunks.jsonl").exists():
        return groups
    for chunk in load_chunks(mrag_dir):
        page = chunk.get("page_pdf")
        if isinstance(page, int):
            groups[page].append(chunk)
    return groups


def _page_text(page_pdf: int, page_printed: list[str], section_ids: list[str], chunk_count: int) -> str:
    printed = f" printed page {', '.join(page_printed[:3])}" if page_printed else ""
    sections = f" Sections: {', '.join(section_ids[:10])}." if section_ids else ""
    return f"MUTCD document page image {page_pdf}{printed}.{sections} Text chunks on page: {chunk_count}."


def _figure_text(record: dict[str, Any]) -> str:
    label = str(record.get("figure_id") or record.get("caption") or "MRAG figure")
    title = str(record.get("title") or "").strip()
    page = record.get("page_pdf")
    text = f"{label} image"
    if title:
        text += f": {title}"
    if page:
        text += f" on PDF page {page}"
    return text + "."


def _local_figure_path(figures_dir: Path, record: dict[str, Any]) -> Path | None:
    raw_path = str(record.get("image_path") or "")
    if raw_path:
        candidate = figures_dir / Path(raw_path).name
        if candidate.exists():
            return candidate
    kind = str(record.get("kind") or "figure").lower()
    canonical = str(record.get("canonical_id") or "").replace(" ", "-")
    page = record.get("page_pdf")
    if canonical and isinstance(page, int):
        pattern = f"{kind}_{canonical}_p{page:04d}.png"
        candidate = figures_dir / pattern
        if candidate.exists():
            return candidate
    return None


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[requested]


def _open_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _embedding_rows(path: Path) -> int | None:
    try:
        import numpy as np

        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return None


def _import_errors(module_names: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in module_names:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as exc:
            errors[name] = repr(exc)
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
