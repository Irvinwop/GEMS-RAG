#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "benchmark" / "questions.jsonl"
DEFAULT_OUTPUT = ROOT / "runs" / "comparison"
OPENAI_BASE_URL = "https://api.openai.com/v1"
METHODS = ("bm25", "graphrag", "paperqa", "gems-rag")
DEFAULT_METHODS = ("bm25", "graphrag", "paperqa")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [method for method in methods if method not in METHODS]
    if not methods or unknown:
        choices = ", ".join(METHODS)
        raise argparse.ArgumentTypeError(
            f"methods must be a comma-separated subset of {choices}; unknown: {unknown}"
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and resume the MUTCD retrieval comparison. Each completed "
            "question-method pair is persisted atomically."
        )
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated method IDs. The comparison defaults to "
            "bm25,graphrag,paperqa; add gems-rag when needed."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL),
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("GRAPHRAG_EMBEDDING_API_BASE"),
    )
    parser.add_argument(
        "--graphrag-working-dir",
        type=Path,
        default=ROOT / "indexes" / "graphrag",
    )
    parser.add_argument(
        "--paperqa-index",
        type=Path,
        default=ROOT / "indexes" / "paperqa" / "docs.pkl",
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
        "--gems-rag-python",
        type=Path,
        default=ROOT / ".venv-gems-rag" / "bin" / "python",
    )
    parser.add_argument(
        "--graphrag-embedding-model",
        default=os.getenv("GRAPHRAG_QUERY_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--graphrag-llm-model",
        default=os.getenv("GRAPHRAG_QUERY_LLM_MODEL"),
    )
    parser.add_argument(
        "--paperqa-embedding-model",
        default=os.getenv("PAPERQA_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--paperqa-llm-model",
        default=os.getenv("PAPERQA_LLM_MODEL"),
    )
    parser.add_argument(
        "--paperqa-summary-model",
        default=os.getenv("PAPERQA_SUMMARY_MODEL"),
    )
    parser.add_argument("--gems-rag-mode", default="full")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if "graphrag" in args.methods and not args.graphrag_embedding_model:
        parser.error(
            "--graphrag-embedding-model or GRAPHRAG_QUERY_EMBEDDING_MODEL "
            "is required"
        )
    if "paperqa" in args.methods:
        required = {
            "--paperqa-embedding-model": args.paperqa_embedding_model,
            "--paperqa-llm-model": args.paperqa_llm_model,
            "--paperqa-summary-model": args.paperqa_summary_model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"required for paperqa: {', '.join(missing)}")
    return args


def question_id(row: dict[str, Any], index: int) -> str:
    value = row.get("question_id") or row.get("qa_id")
    return str(value or f"question_{index + 1:04d}")


def row_path(output: Path, method: str, qa_id: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", qa_id).strip("._") or "question"
    suffix = hashlib.sha256(qa_id.encode("utf-8")).hexdigest()[:10]
    return output / "rows" / method / f"{slug}-{suffix}.json"


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "questions_sha256": sha256_file(args.questions),
        "methods": list(args.methods),
        "top_k": args.top_k,
        "api_key_env": args.api_key_env,
        "base_url": args.base_url,
        "embedding_base_url": args.embedding_base_url,
        "graphrag_working_dir": str(args.graphrag_working_dir),
        "paperqa_index": str(args.paperqa_index),
        "graphrag_python": str(args.graphrag_python),
        "paperqa_python": str(args.paperqa_python),
        "gems_rag_python": str(args.gems_rag_python),
        "graphrag_embedding_model": args.graphrag_embedding_model,
        "graphrag_llm_model": args.graphrag_llm_model,
        "paperqa_embedding_model": args.paperqa_embedding_model,
        "paperqa_llm_model": args.paperqa_llm_model,
        "paperqa_summary_model": args.paperqa_summary_model,
        "gems_rag_mode": args.gems_rag_mode,
    }


def append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None and str(value):
        command.extend([name, str(value)])


def build_command(
    args: argparse.Namespace,
    method: str,
    question: str,
) -> list[str]:
    pipelines = ROOT / "pipelines"
    if method == "bm25":
        return [
            sys.executable,
            str(pipelines / "query_bm25.py"),
            "query",
            "--question",
            question,
            "--top-k",
            str(args.top_k),
        ]

    if method == "graphrag":
        command = [
            sys.executable,
            str(pipelines / "query_graphrag.py"),
            "--python",
            str(args.graphrag_python),
            "--api-key-env",
            args.api_key_env,
            "--working-dir",
            str(args.graphrag_working_dir),
        ]
        append_option(command, "--base-url", args.base_url)
        append_option(
            command,
            "--embedding-base-url",
            args.embedding_base_url or args.base_url,
        )
        append_option(
            command,
            "--query-embedding-model",
            args.graphrag_embedding_model,
        )
        append_option(command, "--query-llm-model", args.graphrag_llm_model)
        command.extend(
            [
                "query",
                "--question",
                question,
                "--method",
                "local",
                "--context-only",
                "--top-k",
                str(args.top_k),
                "--json",
            ]
        )
        return command

    if method == "paperqa":
        command = [
            str(args.paperqa_python),
            str(pipelines / "query_paperqa.py"),
            "--api-key-env",
            args.api_key_env,
            "--index",
            str(args.paperqa_index),
        ]
        append_option(command, "--base-url", args.base_url)
        command.extend(
            [
                "query",
                "--question",
                question,
                "--top-k",
                str(args.top_k),
                "--context-only",
                "--embedding",
                args.paperqa_embedding_model,
                "--llm",
                args.paperqa_llm_model,
                "--summary-llm",
                args.paperqa_summary_model,
            ]
        )
        return command

    if method == "gems-rag":
        return [
            sys.executable,
            str(pipelines / "query_gems_rag.py"),
            "--python",
            str(args.gems_rag_python),
            "retrieve",
            "--question",
            question,
            "--top-k",
            str(args.top_k),
            "--mode",
            args.gems_rag_mode,
            "--persistent",
        ]
    raise ValueError(f"unknown method: {method}")


def decode_output(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("adapter stdout did not contain a JSON object")


def execute_pair(
    args: argparse.Namespace,
    *,
    qa_id: str,
    question: str,
    method: str,
    configuration_sha256: str,
) -> dict[str, Any]:
    command = build_command(args, method, question)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            check=False,
            timeout=args.timeout_seconds,
        )
        output = decode_output(completed.stdout) if completed.returncode == 0 else None
        error = None
        if completed.returncode != 0:
            error = f"adapter exited with status {completed.returncode}"
        status = "ok" if error is None else "error"
        stderr = completed.stderr[-20000:]
        stdout_tail = completed.stdout[-20000:] if error else None
        return_code = completed.returncode
    except Exception as exc:
        output = None
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        stderr = ""
        stdout_tail = None
        return_code = None
    return {
        "schema_version": 1,
        "question_id": qa_id,
        "question": question,
        "method": method,
        "status": status,
        "configuration_sha256": configuration_sha256,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "return_code": return_code,
        "output": output,
        "error": error,
        "stderr": stderr,
        "stdout_tail": stdout_tail,
    }


def collect_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in (output / "rows").glob("*/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def publish_state(
    output: Path,
    *,
    expected_pairs: int,
    question_order: dict[str, int],
) -> None:
    rows = collect_rows(output)
    rows.sort(
        key=lambda row: (
            question_order.get(str(row.get("question_id")), 1_000_000),
            METHOD_ORDER.get(str(row.get("method")), 1_000_000),
        )
    )
    atomic_write_jsonl(output / "results.jsonl", rows)
    completed = sum(row.get("status") == "ok" for row in rows)
    errors = sum(row.get("status") == "error" for row in rows)
    atomic_write_json(
        output / "state.json",
        {
            "schema_version": 1,
            "expected_pairs": expected_pairs,
            "persisted_pairs": len(rows),
            "completed_pairs": completed,
            "error_pairs": errors,
            "remaining_pairs": max(0, expected_pairs - completed),
            "complete": completed == expected_pairs and errors == 0,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def main() -> int:
    args = parse_args()
    if not args.questions.is_file():
        raise FileNotFoundError(args.questions)
    allow_ids = set(args.question_id)
    questions = []
    for index, row in enumerate(read_jsonl(args.questions)):
        qa_id = question_id(row, index)
        if allow_ids and qa_id not in allow_ids:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"question is empty: {qa_id}")
        questions.append((qa_id, question))
        if args.limit is not None and len(questions) >= args.limit:
            break
    if not questions:
        raise ValueError("no questions selected")

    args.output.mkdir(parents=True, exist_ok=True)
    run_configuration = configuration(args)
    configuration_sha256 = stable_hash(run_configuration)
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior = manifest.get("configuration_sha256")
        if prior != configuration_sha256:
            raise ValueError(
                "the output directory belongs to a different retrieval "
                "configuration; choose a new --output directory"
            )
    else:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "configuration_sha256": configuration_sha256,
                "configuration": run_configuration,
            },
        )

    question_order = {qa_id: index for index, (qa_id, _) in enumerate(questions)}
    expected_pairs = len(questions) * len(args.methods)
    try:
        for qa_id, question in questions:
            for method in args.methods:
                destination = row_path(args.output, method, qa_id)
                existing = None
                if destination.is_file():
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                if existing and (
                    existing.get("status") == "ok"
                    or (
                        existing.get("status") == "error"
                        and not args.retry_errors
                    )
                ):
                    continue
                print(f"[{method}] {qa_id}", flush=True)
                result = execute_pair(
                    args,
                    qa_id=qa_id,
                    question=question,
                    method=method,
                    configuration_sha256=configuration_sha256,
                )
                atomic_write_json(destination, result)
                publish_state(
                    args.output,
                    expected_pairs=expected_pairs,
                    question_order=question_order,
                )
    except KeyboardInterrupt:
        publish_state(
            args.output,
            expected_pairs=expected_pairs,
            question_order=question_order,
        )
        return 130

    publish_state(
        args.output,
        expected_pairs=expected_pairs,
        question_order=question_order,
    )
    state = json.loads((args.output / "state.json").read_text(encoding="utf-8"))
    print(json.dumps(state, indent=2))
    return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
