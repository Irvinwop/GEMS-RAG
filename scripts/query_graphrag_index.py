#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "graphrag"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "graphrag_index"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "graphrag" / "bin" / "python"


def main() -> int:
    args = _parse_args()
    env = _env(args.repo)
    _apply_local_api_key(args, env)
    if args.command == "check":
        return _check(args, env)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "init":
        return _run_graphrag(args, env, ["init", "--root", str(args.working_dir), "--force", "--model", args.llm_model, "--embedding", args.embedding_model])
    if args.command == "index":
        return _run_graphrag(args, env, ["index", "--root", str(args.working_dir), "--method", args.method])
    if args.command == "query":
        cmd = [
            "query",
            args.question,
            "--root",
            str(args.working_dir),
            "--method",
            args.method,
            "--response-type",
            args.response_type,
        ]
        if args.data:
            cmd.extend(["--data", str(args.data)])
        completed = _graphrag_subprocess(args, env, cmd)
        stdout = completed.stdout.strip()
        if args.json:
            print(json.dumps({"question": args.question, "method": args.method, "result": stdout}, ensure_ascii=False))
        else:
            print(stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, index, or query Microsoft GraphRAG over exported MRAG chunks.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--python", default=_default_python(), help="Python executable used to run the GraphRAG CLI.")
    parser.add_argument("--api-key-env", default="GRAPHRAG_API_KEY", help="Provider API-key env var expected by the generated GraphRAG .env/settings.")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check whether GraphRAG imports from the cloned source tree.")

    prepare = sub.add_parser("prepare", help="Write GraphRAG input text from exported MRAG chunks.")
    prepare.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    prepare.add_argument("--force", action="store_true")

    init = sub.add_parser("init", help="Run GraphRAG init in the ignored working directory.")
    init.add_argument("--llm-model", default=os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini"))
    init.add_argument("--embedding-model", default=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small"))

    index = sub.add_parser("index", help="Run GraphRAG indexing.")
    index.add_argument("--method", default="standard", choices=["standard", "fast"])

    query = sub.add_parser("query", help="Query an indexed GraphRAG workspace.")
    query.add_argument("--question", required=True)
    query.add_argument("--method", default="local", choices=["local", "global", "drift", "basic"])
    query.add_argument("--response-type", default="Multiple Paragraphs")
    query.add_argument("--data", type=Path)
    query.add_argument("--json", action="store_true")
    return parser.parse_args()


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(repo / "packages" / "graphrag")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _apply_local_api_key(args: argparse.Namespace, env: dict[str, str]) -> None:
    if os.getenv(args.api_key_env):
        return
    if args.allow_missing_api_key:
        env[args.api_key_env] = "local"


def _check(args: argparse.Namespace, env: dict[str, str]) -> int:
    version = _python_version(args.python)
    compatible = _python_is_compatible(version)
    completed = _graphrag_subprocess(args, env, ["--help"]) if compatible else None
    cli_runnable = bool(compatible and completed and completed.returncode == 0)
    api_key_present = bool(os.getenv(args.api_key_env))
    api_key_usable = api_key_present or bool(args.allow_missing_api_key)
    settings_found = (args.working_dir / "settings.yaml").exists()
    env_file_found = (args.working_dir / ".env").exists()
    index_files = _index_files(args.working_dir)
    environment_ready = args.repo.exists() and cli_runnable
    index_ready = settings_found and bool(index_files)
    report = {
        "runnable": environment_ready and api_key_usable and index_ready,
        "environment_ready": environment_ready,
        "cli_runnable": cli_runnable,
        "repo": str(args.repo),
        "repo_found": args.repo.exists(),
        "working_dir": str(args.working_dir),
        "working_dir_exists": args.working_dir.exists(),
        "settings_found": settings_found,
        "env_file_found": env_file_found,
        "index_ready": index_ready,
        "index_file_count": len(index_files),
        "index_files_sample": index_files[:20],
        "python": str(args.python),
        "python_version": version,
        "python_compatible": compatible,
        "requires_python": ">=3.11,<3.14",
        "api_key_env": args.api_key_env,
        "api_key_present": api_key_present,
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "api_key_usable": api_key_usable,
        "returncode": completed.returncode if completed else None,
        "stderr": completed.stderr[-4000:] if completed else "GraphRAG upstream requires Python >=3.11,<3.14; set GRAPHRAG_PYTHON to a compatible interpreter.",
        "notes": "GraphRAG CLI is usable when cli_runnable is true; query runs also need generated settings, output artifacts, and GRAPHRAG_API_KEY or a provider-specific override.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["runnable"] else 2


def _prepare(args: argparse.Namespace) -> int:
    input_dir = args.working_dir / "input"
    if input_dir.exists() and args.force:
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    out_path = input_dir / "mutcd_chunks.txt"
    if out_path.exists() and not args.force:
        print(json.dumps({"prepared": True, "path": str(out_path), "skipped": True}))
        return 0
    count = 0
    with args.chunks.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(f"\n\n--- {row['doc_id']} ---\n")
            dst.write(row["text"].strip())
            dst.write("\n")
            count += 1
    print(json.dumps({"prepared": True, "chunks": count, "path": str(out_path)}))
    return 0


def _run_graphrag(args: argparse.Namespace, env: dict[str, str], command: list[str]) -> int:
    completed = _graphrag_subprocess(args, env, command)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


def _graphrag_subprocess(args: argparse.Namespace, env: dict[str, str], command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [args.python, "-c", "from graphrag.cli.main import app; app()", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )


def _python_version(python: str) -> dict[str, Any]:
    completed = subprocess.run(
        [python, "-c", "import sys, json; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'micro': sys.version_info.micro, 'executable': sys.executable}))"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-1000:] or completed.stdout[-1000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": completed.stdout[-1000:]}


def _python_is_compatible(version: dict[str, Any]) -> bool:
    major = version.get("major")
    minor = version.get("minor")
    return major == 3 and isinstance(minor, int) and 11 <= minor < 14


def _index_files(working_dir: Path) -> list[str]:
    output_dir = working_dir / "output"
    candidates: list[Path] = []
    if output_dir.exists():
        candidates.extend(path for path in output_dir.rglob("*.parquet") if path.is_file())
    candidates.extend(path for path in working_dir.glob("*.parquet") if path.is_file())
    return sorted(str(path.relative_to(working_dir)) for path in candidates)


def _default_python() -> str:
    if os.getenv("GRAPHRAG_PYTHON"):
        return os.environ["GRAPHRAG_PYTHON"]
    if DEFAULT_ENV_PYTHON.exists():
        return str(DEFAULT_ENV_PYTHON)
    return sys.executable


if __name__ == "__main__":
    raise SystemExit(main())
