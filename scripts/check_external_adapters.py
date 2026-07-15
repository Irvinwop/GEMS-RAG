#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    {
        "name": "qdrant_hash_vector_command",
        "command": [".venv/bin/python", "scripts/query_vector_db.py", "check"],
        "required_for": ["qdrant_hash_vector_command"],
    },
    {
        "name": "dpr",
        "command": [".venv/bin/python", "scripts/query_dpr_index.py", "check"],
        "required_for": ["dpr_dense", "canonical_rag_dpr"],
    },
    {
      "name": "gfmrag",
        "command": [".venv/bin/python", "scripts/query_gfmrag_index.py", "check"],
        "required_for": ["gfm_rag"],
    },
    {
        "name": "megarag",
        "command": [".venv/bin/python", "scripts/query_megarag_index.py", "check"],
        "required_for": ["megarag_hybrid_context"],
    },
    {
        "name": "mrag_reference",
        "command": [".venv/bin/python", "scripts/query_mrag_reference.py", "check", "--mode", "full"],
        "required_for": ["gems_full", "gems_no_graph", "gems_no_visual", "gems_no_rule", "gems_no_hierarchy"],
    },
    {
        "name": "graphrag",
        "command": [".venv/bin/python", "scripts/query_graphrag_index.py", "check"],
        "required_for": ["graphrag_local"],
    },
    {
        "name": "lightrag",
        "command": [".venv/bin/python", "scripts/query_lightrag_index.py", "check"],
        "required_for": ["lightrag_hybrid_context"],
    },
    {
        "name": "raganything",
        "command": [".venv/bin/python", "scripts/query_raganything_index.py", "check"],
        "required_for": ["raganything_hybrid"],
    },
    {
        "name": "hipporag",
        "command": [".venv/bin/python", "scripts/query_hipporag_index.py", "check"],
        "required_for": ["hipporag"],
    },
    {
        "name": "visrag",
        "command": [".venv/bin/python", "scripts/query_visrag_index.py", "check"],
        "required_for": ["visrag_pages"],
    },
    {
        "name": "paperqa2",
        "command": [".venv/bin/python", "scripts/query_paperqa_index.py", "check"],
        "required_for": ["paperqa2_chunks"],
    },
]
LOCAL_OPENAI_ADAPTERS = {"graphrag", "hipporag", "lightrag", "megarag", "raganything", "paperqa2"}


def main() -> int:
    args = _parse_args()
    checks = [_with_local_openai_options(item, args) for item in CHECKS]
    results = [_run_check(item, args.timeout_s) for item in checks]
    report = {
        "root": str(ROOT),
        "local_openai_mode": bool(args.allow_missing_api_key),
        "local_openai_base_url": args.local_openai_base_url,
        "ready": [item["name"] for item in results if item["ok"]],
        "environment_ready": [item["name"] for item in results if _environment_ready(item)],
        "blocked_by_credentials": [item["name"] for item in results if _blocked_by_credentials(item)],
        "blocked_by_model_service": [item["name"] for item in results if _blocked_by_model_service(item)],
        "not_ready": [item["name"] for item in results if not item["ok"]],
        "checks": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["not_ready"]:
        return 2
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check readiness of command-backed external RAG adapters.")
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any adapter is not ready.")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use local dummy-key mode for compatible OpenAI-style adapters.")
    parser.add_argument("--local-openai-base-url", default="http://localhost:8000/v1", help="Base URL passed to compatible local OpenAI-style adapter checks.")
    return parser.parse_args()


def _with_local_openai_options(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_missing_api_key or item["name"] not in LOCAL_OPENAI_ADAPTERS:
        return item
    command = list(item["command"])
    if item["name"] in {"graphrag", "hipporag", "megarag", "paperqa2"}:
        command[2:2] = ["--base-url", args.local_openai_base_url, "--allow-missing-api-key"]
    elif item["name"] in {"lightrag", "raganything"}:
        command.extend(["--base-url", args.local_openai_base_url, "--allow-missing-api-key"])
    return {**item, "command": command, "local_openai_mode": True}


def _run_check(item: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            item["command"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {**item, "ok": False, "error": repr(exc)}
    parsed = _parse_json(completed.stdout)
    ok = completed.returncode == 0
    if isinstance(parsed, dict) and "runnable" in parsed:
        ok = bool(parsed["runnable"])
    return {
        **item,
        "ok": ok,
        "returncode": completed.returncode,
        "stdout_json": parsed,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _environment_ready(item: dict[str, Any]) -> bool:
    parsed = item.get("stdout_json")
    if not isinstance(parsed, dict):
        return False
    if "environment_ready" in parsed:
        return bool(parsed["environment_ready"])
    if parsed.get("cli_runnable") is True:
        return True
    import_errors = parsed.get("missing_or_failed_imports")
    if import_errors == {} and parsed.get("repo_found", True):
        return True
    return bool(parsed.get("runnable"))


def _blocked_by_credentials(item: dict[str, Any]) -> bool:
    parsed = item.get("stdout_json")
    if not isinstance(parsed, dict) or not _environment_ready(item):
        return False
    return parsed.get(
        "credential_available",
        parsed.get("api_key_usable", parsed.get("api_key_present")),
    ) is False


def _blocked_by_model_service(item: dict[str, Any]) -> bool:
    parsed = item.get("stdout_json")
    if not isinstance(parsed, dict) or not _environment_ready(item):
        return False
    return parsed.get("model_service_ready") is False and parsed.get("credential_available") is not False


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for idx, char in reversed(list(enumerate(text))):
        if char != "{":
            continue
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())
