#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CORPUS = ROOT / "indexes" / "corpus" / "chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "rebuilt_indexes"
METHODS = ("bm25", "graphrag", "paperqa")


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    unknown = [method for method in methods if method not in METHODS]
    if not methods or unknown:
        raise argparse.ArgumentTypeError(
            f"methods must be a subset of {','.join(METHODS)}; unknown={unknown}"
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the BM25, GraphRAG, and PaperQA comparison indexes with "
            "the OpenAI API. Completed stages are resumed from state.json."
        )
    )
    parser.add_argument("--methods", type=parse_methods, default=METHODS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--graphrag-completion-id",
        default=os.getenv("GRAPHRAG_LLM_MODEL"),
    )
    parser.add_argument(
        "--graphrag-embedding-id",
        default=os.getenv("GRAPHRAG_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--paperqa-embedding-id",
        default=os.getenv("PAPERQA_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--graphrag-python",
        type=Path,
        default=ROOT / ".venv-graphrag" / "bin" / "python",
    )
    parser.add_argument(
        "--paperqa-python",
        type=Path,
        default=ROOT / ".venv-paperqa" / "bin" / "python",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry a stage whose last recorded attempt failed.",
    )
    args = parser.parse_args()
    if "graphrag" in args.methods:
        if not args.graphrag_completion_id:
            parser.error(
                "--graphrag-completion-id or GRAPHRAG_LLM_MODEL is required"
            )
        if not args.graphrag_embedding_id:
            parser.error(
                "--graphrag-embedding-id or GRAPHRAG_EMBEDDING_MODEL is required"
            )
    if "paperqa" in args.methods and not args.paperqa_embedding_id:
        parser.error(
            "--paperqa-embedding-id or PAPERQA_EMBEDDING_MODEL is required"
        )
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "api": "openai",
        "base_url": OPENAI_BASE_URL,
        "methods": list(args.methods),
        "corpus": {
            "sha256": sha256(args.corpus),
            "bytes": args.corpus.stat().st_size,
        },
        "graphrag_completion_id": args.graphrag_completion_id,
        "graphrag_embedding_id": args.graphrag_embedding_id,
        "paperqa_embedding_id": args.paperqa_embedding_id,
    }


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    pipelines = ROOT / "pipelines"
    result: list[tuple[str, list[str]]] = []
    if "bm25" in args.methods:
        result.append(("bm25:ready", []))
    if "graphrag" in args.methods:
        graph_root = args.output / "graphrag"
        common = [
            sys.executable,
            str(pipelines / "query_graphrag.py"),
            "--python",
            str(args.graphrag_python),
            "--working-dir",
            str(graph_root),
            "--api-key-env",
            "OPENAI_API_KEY",
            "--base-url",
            OPENAI_BASE_URL,
            "--embedding-base-url",
            OPENAI_BASE_URL,
        ]
        result.extend(
            [
                (
                    "graphrag:prepare",
                    [
                        *common,
                        "prepare",
                        "--chunks",
                        str(args.corpus),
                    ],
                ),
                (
                    "graphrag:init",
                    [
                        *common,
                        "init",
                        "--llm-model",
                        args.graphrag_completion_id,
                        "--embedding-model",
                        args.graphrag_embedding_id,
                    ],
                ),
                ("graphrag:index", [*common, "index", "--method", "standard"]),
            ]
        )
    if "paperqa" in args.methods:
        result.append(
            (
                "paperqa:index",
                [
                    str(args.paperqa_python),
                    str(pipelines / "query_paperqa.py"),
                    "--index",
                    str(args.output / "paperqa" / "docs.pkl"),
                    "--api-key-env",
                    "OPENAI_API_KEY",
                    "--base-url",
                    OPENAI_BASE_URL,
                    "index",
                    "--ingestion-mode",
                    "shared_corpus",
                    "--chunks",
                    str(args.corpus),
                    "--embedding",
                    args.paperqa_embedding_id,
                ],
            )
        )
    return result


def load_state(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "configuration": expected,
            "stages": {},
            "complete": False,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("configuration") != expected:
        raise ValueError(
            f"{path} belongs to a different build configuration; "
            "use a different --output directory"
        )
    return state


def run_stage(
    *,
    name: str,
    command: list[str],
    state: dict[str, Any],
    state_path: Path,
    env: dict[str, str],
    retry_failed: bool,
) -> None:
    previous = state["stages"].get(name, {})
    if previous.get("status") == "complete":
        print(f"[skip] {name}", flush=True)
        return
    if previous.get("status") == "failed" and not retry_failed:
        raise RuntimeError(
            f"{name} previously failed; pass --retry-failed after correcting it"
        )
    if not command:
        state["stages"][name] = {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(state_path, state)
        print(f"[done] {name}", flush=True)
        return

    state["stages"][name] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(state_path, state)
    print(f"[run] {name}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        check=False,
    )
    state["stages"][name] = {
        "status": "complete" if completed.returncode == 0 else "failed",
        "completed_at": datetime.now(UTC).isoformat(),
        "returncode": completed.returncode,
    }
    atomic_write_json(state_path, state)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} exited with status {completed.returncode}")
    print(f"[done] {name}", flush=True)


def main() -> int:
    args = parse_args()
    args.corpus = args.corpus.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and any(method != "bm25" for method in args.methods):
        raise RuntimeError("OPENAI_API_KEY is required")

    args.output.mkdir(parents=True, exist_ok=True)
    expected = fingerprint(args)
    state_path = args.output / "state.json"
    state = load_state(state_path, expected)
    env = {
        **os.environ,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "GRAPHRAG_API_BASE": OPENAI_BASE_URL,
        "GRAPHRAG_EMBEDDING_API_BASE": OPENAI_BASE_URL,
    }
    if api_key:
        env["GRAPHRAG_API_KEY"] = api_key

    for name, command in commands(args):
        run_stage(
            name=name,
            command=command,
            state=state,
            state_path=state_path,
            env=env,
            retry_failed=args.retry_failed,
        )
    state["complete"] = True
    state["completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(state_path, state)
    print(json.dumps({"complete": True, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
