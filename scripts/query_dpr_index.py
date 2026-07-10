#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "dpr"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data" / "working" / "dpr_index" / "context_embeddings.npy"
DEFAULT_METADATA = ROOT / "data" / "working" / "dpr_index" / "chunks.jsonl"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "dpr" / "bin" / "python"
REQUIRED_MODULES = ["numpy", "torch", "transformers"]


def main() -> int:
    args = _parse_args()
    reexec_code = _maybe_reexec(args.python)
    if reexec_code is not None:
        return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "index":
        return _index(args)
    if args.command == "query":
        return _query(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index and query MUTCD chunks with the original DPR encoders.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--python", type=Path, default=Path(os.getenv("DPR_PYTHON", str(DEFAULT_ENV_PYTHON))))
    parser.add_argument("--context-model", default="facebook/dpr-ctx_encoder-single-nq-base")
    parser.add_argument("--question-model", default="facebook/dpr-question_encoder-single-nq-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    index = sub.add_parser("index")
    index.add_argument("--force", action="store_true")
    query = sub.add_parser("query")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    repo_found = args.repo.exists()
    chunks_found = args.chunks.exists()
    environment_ready = repo_found and chunks_found and not missing
    index_ready = args.embeddings.exists() and args.metadata.exists()
    return {
        "runnable": environment_ready and index_ready,
        "environment_ready": environment_ready,
        "index_ready": index_ready,
        "repo": str(args.repo),
        "repo_found": repo_found,
        "chunks": str(args.chunks),
        "chunks_found": chunks_found,
        "embeddings": str(args.embeddings),
        "metadata": str(args.metadata),
        "context_model": getattr(args, "context_model", "facebook/dpr-ctx_encoder-single-nq-base"),
        "question_model": getattr(args, "question_model", "facebook/dpr-question_encoder-single-nq-base"),
        "missing_or_failed_imports": {name: "not installed" for name in missing},
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
    }


def _index(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["environment_ready"]:
        print(json.dumps({"error": "environment_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    if report["index_ready"] and not args.force:
        print(json.dumps({"status": "already_indexed", **report}, indent=2))
        return 0

    import numpy as np
    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

    records = list(_read_jsonl(args.chunks))
    if not records:
        print(json.dumps({"error": "empty_corpus", "chunks": str(args.chunks)}), file=sys.stderr)
        return 2
    device = _device(args.device, torch)
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.context_model)
    model = DPRContextEncoder.from_pretrained(args.context_model).to(device).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            titles = [str(record.get("section_title") or record.get("section_id") or "") for record in batch]
            texts = [str(record.get("text") or "") for record in batch]
            encoded = tokenizer(titles, texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batches.append(model(**encoded).pooler_output.detach().cpu().numpy().astype("float32"))
    embeddings = np.concatenate(batches, axis=0)

    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings, embeddings)
    with args.metadata.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": "indexed",
                "rows": len(records),
                "dimensions": int(embeddings.shape[1]),
                "embeddings": str(args.embeddings),
                "metadata": str(args.metadata),
                "device": str(device),
            },
            indent=2,
        )
    )
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "adapter_not_ready", **report}, indent=2), file=sys.stderr)
        return 2

    import numpy as np
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

    records = list(_read_jsonl(args.metadata))
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if len(records) != embeddings.shape[0]:
        print(
            json.dumps(
                {"error": "index_metadata_mismatch", "embedding_rows": int(embeddings.shape[0]), "metadata_rows": len(records)}
            ),
            file=sys.stderr,
        )
        return 2
    device = _device(args.device, torch)
    tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(args.question_model)
    model = DPRQuestionEncoder.from_pretrained(args.question_model).to(device).eval()
    with torch.no_grad():
        encoded = tokenizer(args.question, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        query_vector = model(**encoded).pooler_output[0].detach().cpu().numpy().astype("float32")
    scores = embeddings @ query_vector
    indices = _top_indices(scores, args.top_k)
    chunks = [{**records[index], "score": float(scores[index])} for index in indices]
    print(
        json.dumps(
            {
                "question": args.question,
                "chunks": chunks,
                "debug": {
                    "method": "dpr",
                    "context_model": args.context_model,
                    "question_model": args.question_model,
                    "upstream_repo": "facebookresearch/DPR",
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


def _top_indices(scores: Iterable[float], top_k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[: max(top_k, 0)]


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _device(requested: str, torch: Any):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


if __name__ == "__main__":
    raise SystemExit(main())
