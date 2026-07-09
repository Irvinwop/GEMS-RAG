from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
HARNESS_PYTHON = ".venv/bin/python"
ADAPTER_SCRIPT_MAP = {
    "scripts/query_vector_db.py": "qdrant_hash_vector_command",
    "scripts/query_mrag_reference.py": "mrag_reference",
    "scripts/query_graphrag_index.py": "graphrag",
    "scripts/query_lightrag_index.py": "lightrag",
    "scripts/query_raganything_index.py": "raganything",
    "scripts/query_hipporag_index.py": "hipporag",
    "scripts/query_visrag_index.py": "visrag",
    "scripts/query_paperqa_index.py": "paperqa2",
}


@dataclass(frozen=True)
class AdapterPlan:
    name: str
    check_command: list[str]
    build_commands: list[list[str]]
    notes: str


Runner = Callable[..., subprocess.CompletedProcess[str]]


def add_external_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only", help="Comma-separated adapter names to build. Defaults to all known adapters.")
    parser.add_argument("--config", type=Path, help="Restrict setup to external adapters referenced by an experiment config.")
    parser.add_argument("--skip", help="Comma-separated adapter names to skip.")
    parser.add_argument("--dry-run", action="store_true", help="Run prechecks and print commands without executing index builds.")
    parser.add_argument("--force", action="store_true", help="Run build commands even if an adapter already reports query-ready.")
    parser.add_argument("--no-precheck", action="store_true", help="Run build commands without checking environment readiness first.")
    parser.add_argument("--allow-failures", action="store_true", help="Exit zero even if a build command fails.")
    parser.add_argument("--strict-skips", action="store_true", help="Exit non-zero when any selected adapter is skipped.")
    parser.add_argument("--timeout-s", type=int, default=3600, help="Timeout for each build command.")
    parser.add_argument("--check-timeout-s", type=int, default=60, help="Timeout for each readiness check.")
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use dummy local-key mode for OpenAI-compatible adapters.")
    parser.add_argument("--local-openai-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--graphrag-method", default="standard", choices=["standard", "fast"])
    parser.add_argument("--visrag-scope", default="pages", choices=["pages", "figures", "both"])
    parser.add_argument("--visrag-limit", type=int, help="Limit VisRAG manifest/index rows for smoke builds.")
    parser.add_argument("--visrag-batch-size", type=int, default=4)
    parser.add_argument("--hipporag-limit", type=int, help="Limit HippoRAG docs for smoke builds.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ignored local indexes for command-backed external RAG adapters.")
    add_external_index_args(parser)
    args = parser.parse_args(argv)
    report = build_external_indexes(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return external_index_exit_code(report, args)


def external_index_exit_code(report: dict[str, Any], args: argparse.Namespace) -> int:
    if report["failed"] and not args.allow_failures:
        return 2
    if args.strict_skips and report["skipped"] and not args.allow_failures:
        return 2
    return 0


def build_external_indexes(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    plans = _selected_plans(_adapter_plans(args), only=args.only, skip=args.skip, config=getattr(args, "config", None))
    results = [_run_adapter(plan, args, runner=runner) for plan in plans]
    setup_plan = [_setup_plan_item(result) for result in results]
    return {
        "root": str(ROOT),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "selected": [result["name"] for result in results],
        "query_ready": [
            result["name"]
            for result in results
            if result["status"] in {"already_ready", "built", "check_only_ready"}
        ],
        "needs_index": [result["name"] for result in results if result["status"] == "would_run"],
        "needs_environment": [result["name"] for result in results if result["status"].startswith("skipped")],
        "check_only_not_ready": [result["name"] for result in results if result["status"] == "check_only_not_ready"],
        "built": [result["name"] for result in results if result["status"] == "built"],
        "already_ready": [result["name"] for result in results if result["status"] == "already_ready"],
        "check_only": [result["name"] for result in results if result["status"].startswith("check_only")],
        "would_run": [result["name"] for result in results if result["status"] == "would_run"],
        "skipped": [result["name"] for result in results if result["status"].startswith("skipped")],
        "failed": [result["name"] for result in results if result["status"] == "failed"],
        "setup_plan": setup_plan,
        "results": results,
    }


def _run_adapter(plan: AdapterPlan, args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": plan.name,
        "notes": plan.notes,
        "check_command": plan.check_command,
        "build_commands": plan.build_commands,
        "commands": [],
    }
    precheck = None if args.no_precheck else _run_command(plan.check_command, args.check_timeout_s, runner=runner)
    if precheck is not None:
        result["precheck"] = precheck
    parsed = precheck.get("stdout_json") if precheck else None
    environment_ready = True if args.no_precheck else _environment_ready(parsed)
    already_ready = False if args.no_precheck else bool(isinstance(parsed, dict) and parsed.get("runnable") is True)

    if not plan.build_commands:
        result["status"] = "check_only_ready" if already_ready else "check_only_not_ready"
        return result
    if already_ready and not args.force:
        result["status"] = "already_ready"
        return result
    if not environment_ready:
        result["status"] = "skipped_not_environment_ready"
        return result
    if args.dry_run:
        result["status"] = "would_run"
        return result

    for command in plan.build_commands:
        command_result = _run_command(command, args.timeout_s, runner=runner)
        result["commands"].append(command_result)
        if command_result["returncode"] != 0:
            result["status"] = "failed"
            return result

    if not args.no_precheck:
        final_check = _run_command(plan.check_command, args.check_timeout_s, runner=runner)
        result["final_check"] = final_check
        final_json = final_check.get("stdout_json")
        if not (isinstance(final_json, dict) and final_json.get("runnable") is True):
            result["status"] = "failed"
            return result
    result["status"] = "built"
    return result


def _run_command(command: list[str], timeout_s: int, *, runner: Runner) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {
            "command": command,
            "returncode": 127,
            "duration_s": round(time.monotonic() - started, 3),
            "error": repr(exc),
            "stdout_tail": "",
            "stderr_tail": "",
            "stdout_json": None,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_s": round(time.monotonic() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "stdout_json": _parse_json(completed.stdout),
    }


def _setup_plan_item(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", "unknown"))
    action = _setup_action(status)
    commands = result.get("build_commands") if action == "run_build_commands" else []
    return {
        "name": result["name"],
        "status": status,
        "action": action,
        "commands": commands,
        "notes": result.get("notes", ""),
    }


def _setup_action(status: str) -> str:
    if status in {"already_ready", "built", "check_only_ready"}:
        return "none"
    if status == "would_run":
        return "run_build_commands"
    if status.startswith("skipped"):
        return "install_environment"
    if status == "check_only_not_ready":
        return "install_environment_or_credentials"
    if status == "failed":
        return "inspect_failure"
    return "inspect_status"


def _adapter_plans(args: argparse.Namespace) -> dict[str, AdapterPlan]:
    return {
        "qdrant_hash_vector_command": AdapterPlan(
            name="qdrant_hash_vector_command",
            check_command=[HARNESS_PYTHON, "scripts/query_vector_db.py", "check"],
            build_commands=[],
            notes="The local Qdrant hash-vector command wrapper builds its ignored index lazily during search.",
        ),
        "mrag_reference": AdapterPlan(
            name="mrag_reference",
            check_command=[HARNESS_PYTHON, "scripts/query_mrag_reference.py", "check"],
            build_commands=[],
            notes="Reference MRAG uses the extracted MRAG cache directly; there is no separate index command.",
        ),
        "graphrag": AdapterPlan(
            name="graphrag",
            check_command=_graphrag_command(args, "check"),
            build_commands=[
                _graphrag_command(args, "prepare", ["--force"]),
                _graphrag_command(args, "init"),
                _graphrag_command(args, "index", ["--method", args.graphrag_method]),
            ],
            notes="Runs GraphRAG prepare, init, and index in data/working/graphrag_index/.",
        ),
        "lightrag": AdapterPlan(
            name="lightrag",
            check_command=_openai_subcommand(args, "scripts/query_lightrag_index.py", "check"),
            build_commands=[
                _openai_subcommand(args, "scripts/query_lightrag_index.py", "index", ["--force"] if args.force else [])
            ],
            notes="Indexes data/working/mrag_corpus/lightrag_corpus.txt into the ignored LightRAG working dir.",
        ),
        "raganything": AdapterPlan(
            name="raganything",
            check_command=_openai_subcommand(args, "scripts/query_raganything_index.py", "check"),
            build_commands=[
                _openai_subcommand(args, "scripts/query_raganything_index.py", "index", ["--force"] if args.force else [])
            ],
            notes="Indexes the exported RAG-Anything content list into the ignored RAG-Anything working dir.",
        ),
        "hipporag": AdapterPlan(
            name="hipporag",
            check_command=[HARNESS_PYTHON, "scripts/query_hipporag_index.py", "check"],
            build_commands=[
                _with_optional_limit([HARNESS_PYTHON, "scripts/query_hipporag_index.py", "index"], args.hipporag_limit)
            ],
            notes="Indexes exported MRAG chunks through the cloned HippoRAG package.",
        ),
        "visrag": AdapterPlan(
            name="visrag",
            check_command=[HARNESS_PYTHON, "scripts/query_visrag_index.py", "check"],
            build_commands=[
                _with_optional_limit(
                    [HARNESS_PYTHON, "scripts/query_visrag_index.py", "prepare", "--scope", args.visrag_scope],
                    args.visrag_limit,
                ),
                _with_optional_limit(
                    [
                        HARNESS_PYTHON,
                        "scripts/query_visrag_index.py",
                        "index",
                        "--batch-size",
                        str(args.visrag_batch_size),
                    ],
                    args.visrag_limit,
                ),
            ],
            notes="Prepares a visual manifest and encodes it with VisRAG-Ret.",
        ),
        "paperqa2": AdapterPlan(
            name="paperqa2",
            check_command=_paperqa_command(args, "check"),
            build_commands=[_paperqa_command(args, "index", ["--defer-embedding"])],
            notes="Builds a deferred-embedding PaperQA2 Docs pickle over exported chunks.",
        ),
    }


def _selected_plans(plans: dict[str, AdapterPlan], *, only: str | None, skip: str | None, config: Path | None = None) -> list[AdapterPlan]:
    if only and config is not None:
        raise SystemExit("--only and --config are mutually exclusive")
    only_names = _adapter_names_from_config(config) if config is not None else (_name_set(only) or set(plans))
    skip_names = _name_set(skip)
    unknown = sorted((only_names | skip_names) - set(plans))
    if unknown:
        raise SystemExit(f"unknown adapter name(s): {', '.join(unknown)}")
    return [plans[name] for name in plans if name in only_names and name not in skip_names]


def _name_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _adapter_names_from_config(path: Path) -> set[str]:
    config = load_experiment_config(path)
    names: set[str] = set()
    for retriever in config.retrievers:
        if retriever.kind != "external_command":
            continue
        adapter = _adapter_name_from_command(retriever.options.get("command"))
        if adapter:
            names.add(adapter)
    return names


def _adapter_name_from_command(command: Any) -> str | None:
    if isinstance(command, str):
        parts = shlex.split(command)
    elif isinstance(command, Sequence):
        parts = [str(part) for part in command]
    else:
        return None
    for part in parts:
        normalized = str(part)
        for script, adapter in ADAPTER_SCRIPT_MAP.items():
            if normalized == script or normalized.endswith(f"/{script}"):
                return adapter
    return None


def _graphrag_command(args: argparse.Namespace, subcommand: str, extra: Sequence[str] = ()) -> list[str]:
    command = [HARNESS_PYTHON, "scripts/query_graphrag_index.py"]
    if args.allow_missing_api_key:
        command.append("--allow-missing-api-key")
    command.append(subcommand)
    command.extend(extra)
    return command


def _openai_subcommand(args: argparse.Namespace, script: str, subcommand: str, extra: Sequence[str] = ()) -> list[str]:
    command = [HARNESS_PYTHON, script, subcommand]
    command.extend(extra)
    if args.allow_missing_api_key:
        command.extend(["--base-url", args.local_openai_base_url, "--allow-missing-api-key"])
    return command


def _paperqa_command(args: argparse.Namespace, subcommand: str, extra: Sequence[str] = ()) -> list[str]:
    command = [HARNESS_PYTHON, "scripts/query_paperqa_index.py"]
    if args.allow_missing_api_key:
        command.extend(["--base-url", args.local_openai_base_url, "--allow-missing-api-key"])
    command.append(subcommand)
    command.extend(extra)
    return command


def _with_optional_limit(command: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return command
    return [*command, "--limit", str(limit)]


def _environment_ready(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "environment_ready" in parsed:
        return bool(parsed["environment_ready"])
    if parsed.get("cli_runnable") is True:
        return True
    if "missing_required_modules" in parsed:
        return (
            not parsed.get("missing_required_modules")
            and not parsed.get("missing_alternative_groups")
            and parsed.get("repo_found", True)
        )
    import_errors = parsed.get("missing_or_failed_imports")
    if import_errors == {} and parsed.get("repo_found", True):
        return True
    return bool(parsed.get("runnable"))


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
