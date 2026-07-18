#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.endpoint import probe_openai_endpoint
from gems_rag.index_completion import (
    completion_marker_matches,
    file_identity,
    publish_completion_marker,
    read_completion_marker,
)

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "graphrag"
DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "graphrag_index"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "graphrag" / "bin" / "python"
INDEX_SENTINEL = ".gems_rag_graphrag_index.json"
COMMUNITY_PROMPT_NAMES = (
    "community_report_graph.txt",
    "community_report_text.txt",
)
COMMUNITY_FINDINGS_PREFIX = "- DETAILED FINDINGS:"
COMMUNITY_FINDINGS_INSTRUCTION = (
    "- DETAILED FINDINGS: A list of 2-4 distinct key insights about the community. "
    "Each insight must have a short summary and one concise evidence-grounded paragraph. "
    "Do not repeat or restate a finding."
)
INDEX_PROMPT_NAMES = (
    "extract_graph.txt",
    "summarize_descriptions.txt",
    "extract_claims.txt",
    *COMMUNITY_PROMPT_NAMES,
)
DEFAULT_ENTITY_TYPES = (
    "organization",
    "person",
    "geo",
    "event",
    "traffic_control_device",
    "facility",
    "road_user",
    "regulation",
    "standard",
    "concept",
)


def main() -> int:
    args = _parse_args()
    env = _env(args.repo)
    _apply_local_api_key(args, env)
    if args.command == "check":
        return _check(args, env)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "init":
        return _init(args, env)
    if args.command == "index":
        return _index(args, env)
    if args.command == "query":
        if not _index_ready(args):
            print(
                json.dumps(
                    {
                        "error": "graphrag_index_not_ready",
                        "working_dir": str(args.working_dir),
                        "limit": args.limit,
                    }
                ),
                file=sys.stderr,
            )
            return 2
        if args.json:
            completed = _graphrag_query_json_subprocess(args, env)
            if completed.returncode == 0:
                payload = _query_payload_from_stdout(args, completed.stdout)
                if payload is not None:
                    print(json.dumps(payload, ensure_ascii=False))
                    if completed.stderr:
                        print(completed.stderr, file=sys.stderr, end="")
                    return 0
        cmd = [
            "query",
            args.question,
            "--root",
            str(args.working_dir),
            "--method",
            args.method,
            "--community-level",
            str(args.community_level),
            "--response-type",
            args.response_type,
        ]
        if args.dynamic_community_selection:
            cmd.append("--dynamic-community-selection")
        if args.data:
            cmd.extend(["--data", str(args.data)])
        completed = _graphrag_subprocess(args, env, cmd)
        stdout = completed.stdout.strip()
        if args.json:
            print(json.dumps({"question": args.question, "method": args.method, "top_k": args.top_k, "result": stdout}, ensure_ascii=False))
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
    parser.add_argument(
        "--api-key-env",
        default="GRAPHRAG_API_KEY",
        help="Provider API-key env var; GRAPHRAG_API_KEY falls back to OPENAI_API_KEY.",
    )
    parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("GRAPHRAG_API_BASE") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("GRAPHRAG_EMBEDDING_API_BASE"),
        help="Optional separate OpenAI-compatible embedding endpoint; defaults to --base-url.",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Hard ceiling for each GraphRAG completion model call.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check whether GraphRAG imports from the cloned source tree.")
    check.add_argument("--limit", type=int, help="Expected smoke-index input limit.")

    prepare = sub.add_parser("prepare", help="Write GraphRAG input text from exported MRAG chunks.")
    prepare.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--limit", type=int, help="Prepare only the first N chunks for a smoke index.")

    init = sub.add_parser("init", help="Run GraphRAG init in the ignored working directory.")
    init.add_argument("--llm-model", default=os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini"))
    init.add_argument("--embedding-model", default=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    init.add_argument(
        "--entity-types",
        default=",".join(DEFAULT_ENTITY_TYPES),
        help="Comma-separated GraphRAG entity types used for MUTCD graph extraction.",
    )
    init.add_argument(
        "--max-gleanings",
        type=int,
        default=0,
        help="Optional follow-up extraction passes per chunk; zero avoids prompt-example leakage with small local models.",
    )
    init.add_argument(
        "--community-report-max-length",
        type=int,
        default=300,
        help="Maximum community-report word target inserted into the prompt.",
    )
    init.add_argument(
        "--community-report-max-tokens",
        type=int,
        default=768,
        help="Hard completion-token ceiling for the dedicated community-report model profile.",
    )
    init.add_argument(
        "--community-report-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for deterministic community-report generation.",
    )
    init.add_argument(
        "--keep-community-prompt-examples",
        action="store_true",
        help="Retain GraphRAG's upstream few-shot community examples instead of removing them for local models.",
    )

    index = sub.add_parser("index", help="Run GraphRAG indexing.")
    index.add_argument("--method", default="standard", choices=["standard", "fast"])
    index.add_argument("--limit", type=int, help="Input limit used by the prepared smoke index.")

    query = sub.add_parser("query", help="Query an indexed GraphRAG workspace.")
    query.add_argument("--question", required=True)
    query.add_argument("--method", default="local", choices=["local", "global", "drift", "basic"])
    query.add_argument("--top-k", type=int, default=6, help="Maximum number of structured context records to emit in JSON mode.")
    query.add_argument("--community-level", type=int, default=2)
    query.add_argument("--dynamic-community-selection", action="store_true")
    query.add_argument("--response-type", default="Multiple Paragraphs")
    query.add_argument("--data", type=Path)
    query.add_argument("--json", action="store_true")
    query.add_argument("--limit", type=int, help="Expected smoke-index input limit.")
    args = parser.parse_args()
    if args.llm_max_tokens is not None and args.llm_max_tokens <= 0:
        parser.error("--llm-max-tokens must be positive")
    if getattr(args, "max_gleanings", None) is not None and args.max_gleanings < 0:
        parser.error("--max-gleanings must be non-negative")
    if (
        getattr(args, "community_report_max_length", None) is not None
        and args.community_report_max_length <= 0
    ):
        parser.error("--community-report-max-length must be positive")
    if (
        getattr(args, "community_report_max_tokens", None) is not None
        and args.community_report_max_tokens <= 0
    ):
        parser.error("--community-report-max-tokens must be positive")
    if (
        getattr(args, "community_report_temperature", None) is not None
        and not 0 <= args.community_report_temperature <= 2
    ):
        parser.error("--community-report-temperature must be between 0 and 2")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(repo / "packages" / "graphrag")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _apply_local_api_key(args: argparse.Namespace, env: dict[str, str]) -> None:
    api_key = os.getenv(args.api_key_env)
    if not api_key and args.api_key_env == "GRAPHRAG_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and args.allow_missing_api_key:
        api_key = "local"
    if api_key:
        env[args.api_key_env] = api_key
        env["GRAPHRAG_API_KEY"] = api_key


def _check(args: argparse.Namespace, env: dict[str, str]) -> int:
    version = _python_version(args.python)
    compatible = _python_is_compatible(version)
    completed = _graphrag_subprocess(args, env, ["--help"]) if compatible else None
    cli_runnable = bool(compatible and completed and completed.returncode == 0)
    api_key = env.get("GRAPHRAG_API_KEY")
    api_key_present = bool(api_key)
    credential_available = api_key_present or bool(args.allow_missing_api_key)
    endpoint = probe_openai_endpoint(
        args.base_url,
        api_key=api_key or ("local" if args.allow_missing_api_key else None),
    )
    embedding_base_url = args.embedding_base_url or args.base_url
    embedding_endpoint = (
        endpoint
        if embedding_base_url == args.base_url
        else probe_openai_endpoint(
            embedding_base_url,
            api_key=api_key or ("local" if args.allow_missing_api_key else None),
        )
    )
    endpoint_usable = endpoint["usable"] if endpoint["checked"] else True
    embedding_endpoint_usable = (
        embedding_endpoint["usable"] if embedding_endpoint["checked"] else True
    )
    api_key_usable = credential_available and endpoint_usable and embedding_endpoint_usable
    settings_found = (args.working_dir / "settings.yaml").exists()
    env_file_found = (args.working_dir / ".env").exists()
    index_files = _index_files(args.working_dir)
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel = read_completion_marker(sentinel_path)
    sentinel_matches_input = completion_marker_matches(sentinel_path, _index_identity(args))
    sentinel_files_present = _sentinel_files_present(sentinel, index_files)
    environment_ready = args.repo.exists() and cli_runnable
    index_ready = settings_found and bool(index_files) and sentinel_matches_input and sentinel_files_present
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
        "sentinel": str(sentinel_path),
        "sentinel_found": sentinel_path.is_file(),
        "sentinel_matches_input": sentinel_matches_input,
        "sentinel_files_present": sentinel_files_present,
        "python": str(args.python),
        "python_version": version,
        "python_compatible": compatible,
        "requires_python": ">=3.11,<3.14",
        "api_key_env": args.api_key_env,
        "api_key_envs": (
            ["GRAPHRAG_API_KEY", "OPENAI_API_KEY"]
            if args.api_key_env == "GRAPHRAG_API_KEY"
            else [args.api_key_env]
        ),
        "api_key_present": api_key_present,
        "allow_missing_api_key": bool(args.allow_missing_api_key),
        "credential_available": credential_available,
        "api_key_usable": api_key_usable,
        "base_url": args.base_url,
        "embedding_base_url": embedding_base_url,
        "endpoint": endpoint,
        "embedding_endpoint": embedding_endpoint,
        "endpoint_reachable": endpoint["reachable"],
        "endpoint_usable": endpoint["usable"],
        "embedding_endpoint_reachable": embedding_endpoint["reachable"],
        "embedding_endpoint_usable": embedding_endpoint["usable"],
        "model_service_ready": api_key_usable,
        "returncode": completed.returncode if completed else None,
        "stderr": completed.stderr[-4000:] if completed else "GraphRAG upstream requires Python >=3.11,<3.14; set GRAPHRAG_PYTHON to a compatible interpreter.",
        "notes": "GraphRAG CLI is usable when cli_runnable is true; its generated settings use GRAPHRAG_API_KEY, which defaults to OPENAI_API_KEY in this harness.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["runnable"] else 2


def _index(args: argparse.Namespace, env: dict[str, str]) -> int:
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel_path.unlink(missing_ok=True)
    code = _run_graphrag(
        args,
        env,
        ["index", "--root", str(args.working_dir), "--method", args.method],
    )
    if code != 0:
        return code
    index_files = _index_files(args.working_dir)
    if not index_files:
        print(json.dumps({"error": "graphrag_index_produced_no_artifacts"}), file=sys.stderr)
        return 2
    publish_completion_marker(
        sentinel_path,
        _index_identity(args),
        method=args.method,
        index_files=index_files,
    )
    return 0


def _init(args: argparse.Namespace, env: dict[str, str]) -> int:
    code = _run_graphrag(
        args,
        env,
        [
            "init",
            "--root",
            str(args.working_dir),
            "--force",
            "--model",
            args.llm_model,
            "--embedding",
            args.embedding_model,
        ],
    )
    reasoning_effort = getattr(args, "reasoning_effort", None)
    llm_max_tokens = getattr(args, "llm_max_tokens", None)
    max_gleanings = getattr(args, "max_gleanings", None)
    community_report_max_length = getattr(args, "community_report_max_length", None)
    community_report_max_tokens = getattr(args, "community_report_max_tokens", None)
    community_report_temperature = getattr(args, "community_report_temperature", None)
    if code != 0:
        return code
    if (
        args.base_url
        or args.embedding_base_url
        or reasoning_effort
        or llm_max_tokens
        or max_gleanings is not None
        or community_report_max_length is not None
        or community_report_max_tokens is not None
        or community_report_temperature is not None
    ):
        code = _configure_api_base(
            args.working_dir / "settings.yaml",
            args.base_url,
            embedding_base_url=args.embedding_base_url,
            reasoning_effort=reasoning_effort,
            llm_max_tokens=llm_max_tokens,
            entity_types=[part.strip() for part in args.entity_types.split(",") if part.strip()],
            max_gleanings=max_gleanings,
            community_report_max_length=community_report_max_length,
            community_report_max_tokens=community_report_max_tokens,
            community_report_temperature=community_report_temperature,
        )
        if code != 0:
            return code
    if not getattr(args, "keep_community_prompt_examples", False):
        return _sanitize_community_prompts(args.working_dir)
    return 0


def _sanitize_community_prompts(working_dir: Path) -> int:
    sanitized: list[str] = []
    try:
        for name in COMMUNITY_PROMPT_NAMES:
            path = working_dir / "prompts" / name
            original = path.read_text(encoding="utf-8")
            without_example = _remove_few_shot_example(original)
            compact = _compact_community_prompt(without_example)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write(compact)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            sanitized.append(str(path))
    except Exception as exc:
        print(json.dumps({"error": "sanitize_graphrag_prompts_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"community_prompt_examples_removed": True, "prompts": sanitized}))
    return 0


def _remove_few_shot_example(prompt: str) -> str:
    example_marker = "# Example Input"
    real_data_marker = "# Real Data"
    example_start = prompt.find(example_marker)
    real_data_start = prompt.find(real_data_marker, example_start + len(example_marker))
    if example_start < 0 or real_data_start < 0:
        raise ValueError("community prompt does not contain the expected example and real-data markers")
    return f"{prompt[:example_start].rstrip()}\n\n{prompt[real_data_start:].lstrip()}"


def _compact_community_prompt(prompt: str) -> str:
    lines = prompt.splitlines()
    replacements = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith(COMMUNITY_FINDINGS_PREFIX):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}{COMMUNITY_FINDINGS_INSTRUCTION}"
            replacements += 1
    if replacements == 0:
        raise ValueError("community prompt does not contain a detailed-findings instruction")
    compact = "\n".join(lines)
    return f"{compact}\n" if prompt.endswith("\n") else compact


def _model_cache_partition(prefix: str, model: dict[str, Any]) -> str:
    identity = {
        key: value
        for key, value in model.items()
        if key not in {"api_key", "metrics", "rate_limit", "retry"}
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _configure_model_cache_partitions(payload: dict[str, Any]) -> None:
    workflows = (
        ("extract_graph", "completion_models", "completion_model_id", "default_completion_model"),
        ("summarize_descriptions", "completion_models", "completion_model_id", "default_completion_model"),
        ("extract_claims", "completion_models", "completion_model_id", "default_completion_model"),
        ("community_reports", "completion_models", "completion_model_id", "default_completion_model"),
        ("embed_text", "embedding_models", "embedding_model_id", "default_embedding_model"),
    )
    for workflow_name, models_name, model_id_name, default_model_id in workflows:
        workflow = payload.get(workflow_name)
        models = payload.get(models_name)
        if not isinstance(workflow, dict) or not isinstance(models, dict):
            continue
        model_id = workflow.get(model_id_name, default_model_id)
        model = models.get(model_id)
        if not isinstance(model, dict):
            raise ValueError(f"missing {model_id} for {workflow_name} cache partition")
        workflow["model_instance_name"] = _model_cache_partition(workflow_name, model)


def _configure_api_base(
    settings_path: Path,
    base_url: str | None,
    *,
    embedding_base_url: str | None = None,
    reasoning_effort: str | None = None,
    llm_max_tokens: int | None = None,
    entity_types: list[str] | None = None,
    max_gleanings: int | None = None,
    community_report_max_length: int | None = None,
    community_report_max_tokens: int | None = None,
    community_report_temperature: float | None = None,
) -> int:
    try:
        import yaml

        payload = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        for section in ["completion_models", "embedding_models"]:
            models = payload.get(section) if isinstance(payload, dict) else None
            if not isinstance(models, dict):
                raise ValueError(f"missing {section} in {settings_path}")
            for model in models.values():
                if isinstance(model, dict):
                    model_base_url = (
                        embedding_base_url or base_url
                        if section == "embedding_models"
                        else base_url
                    )
                    if model_base_url:
                        model["api_base"] = model_base_url
                    if section == "completion_models" and (
                        reasoning_effort or llm_max_tokens
                    ):
                        call_args = model.get("call_args") or {}
                        if not isinstance(call_args, dict):
                            raise ValueError("completion model call_args must be a mapping")
                        if reasoning_effort:
                            call_args["reasoning_effort"] = reasoning_effort
                        if llm_max_tokens:
                            call_args["max_tokens"] = llm_max_tokens
                        model["call_args"] = call_args
        if entity_types or max_gleanings is not None:
            extract_graph = payload.get("extract_graph") if isinstance(payload, dict) else None
            if not isinstance(extract_graph, dict):
                raise ValueError(f"missing extract_graph in {settings_path}")
            if entity_types:
                extract_graph["entity_types"] = entity_types
            if max_gleanings is not None:
                extract_graph["max_gleanings"] = max_gleanings
        if (
            community_report_max_length is not None
            or community_report_max_tokens is not None
            or community_report_temperature is not None
        ):
            community_reports = payload.get("community_reports") if isinstance(payload, dict) else None
            if not isinstance(community_reports, dict):
                raise ValueError(f"missing community_reports in {settings_path}")
            if community_report_max_length is not None:
                community_reports["max_length"] = community_report_max_length
            if community_report_max_tokens is not None or community_report_temperature is not None:
                completion_models = payload.get("completion_models")
                if not isinstance(completion_models, dict) or not completion_models:
                    raise ValueError(f"missing completion_models in {settings_path}")
                source_model = completion_models.get("default_completion_model") or next(
                    iter(completion_models.values())
                )
                if not isinstance(source_model, dict):
                    raise ValueError("completion model must be a mapping")
                report_model = copy.deepcopy(source_model)
                report_call_args = report_model.get("call_args") or {}
                if not isinstance(report_call_args, dict):
                    raise ValueError("community report model call_args must be a mapping")
                if community_report_max_tokens is not None:
                    report_call_args["max_tokens"] = community_report_max_tokens
                if community_report_temperature is not None:
                    report_call_args["temperature"] = community_report_temperature
                report_model["call_args"] = report_call_args
                report_model_id = "community_report_completion_model"
                completion_models[report_model_id] = report_model
                community_reports["completion_model_id"] = report_model_id
        _configure_model_cache_partitions(payload)
        settings_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"error": "configure_api_base_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "configured": True,
                "settings": str(settings_path),
                "api_base": base_url,
                "embedding_api_base": embedding_base_url or base_url,
                "reasoning_effort": reasoning_effort,
                "llm_max_tokens": llm_max_tokens,
                "entity_types": entity_types,
                "max_gleanings": max_gleanings,
                "community_report_max_length": community_report_max_length,
                "community_report_max_tokens": community_report_max_tokens,
                "community_report_temperature": community_report_temperature,
            }
        )
    )
    return 0


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
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with args.chunks.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                row = json.loads(line)
                dst.write(f"\n\n--- {row['doc_id']} ---\n")
                dst.write(row["text"].strip())
                dst.write("\n")
                count += 1
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(json.dumps({"prepared": True, "chunks": count, "limit": args.limit, "path": str(out_path)}))
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


def _graphrag_query_json_subprocess(args: argparse.Namespace, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    request = {
        "question": args.question,
        "root": str(args.working_dir),
        "method": args.method,
        "data": str(args.data) if args.data else None,
        "community_level": args.community_level,
        "dynamic_community_selection": args.dynamic_community_selection,
        "response_type": args.response_type,
    }
    code = r"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from graphrag.cli.query import run_basic_search, run_drift_search, run_global_search, run_local_search
from graphrag.utils.api import reformat_context_data

request = json.loads(__import__("sys").argv[1])
data_dir = Path(request["data"]) if request.get("data") else None
root_dir = Path(request["root"])
method = request["method"]
captured = io.StringIO()

with redirect_stdout(captured):
    if method == "local":
        response, context_data = run_local_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=int(request["community_level"]),
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    elif method == "global":
        response, context_data = run_global_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=int(request["community_level"]),
            dynamic_community_selection=bool(request["dynamic_community_selection"]),
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    elif method == "drift":
        response, context_data = run_drift_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=int(request["community_level"]),
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    elif method == "basic":
        response, context_data = run_basic_search(
            data_dir=data_dir,
            root_dir=root_dir,
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    else:
        raise ValueError(f"unknown GraphRAG query method: {method}")

try:
    formatted_context = reformat_context_data(context_data if isinstance(context_data, dict) else {"context": context_data})
except Exception:
    formatted_context = {"context": context_data}

print(json.dumps({"response": response, "context_data": formatted_context, "captured_stdout": captured.getvalue()}, ensure_ascii=False, default=str))
"""
    return subprocess.run(
        [args.python, "-c", code, json.dumps(request)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )


def _query_payload_from_stdout(args: argparse.Namespace, stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    contexts = _contexts_from_graphrag_data(payload.get("context_data"), top_k=args.top_k, method=args.method)
    result = payload.get("response") or payload.get("captured_stdout") or stdout
    response: dict[str, Any] = {
        "question": args.question,
        "method": args.method,
        "top_k": args.top_k,
        "response_type": args.response_type,
        "result": result,
        "contexts": contexts,
    }
    if args.community_level is not None:
        response["community_level"] = args.community_level
    if args.dynamic_community_selection:
        response["dynamic_community_selection"] = True
    return response


def _contexts_from_graphrag_data(context_data: Any, *, top_k: int, method: str) -> list[dict[str, Any]]:
    if not isinstance(context_data, dict):
        return []
    if top_k <= 0:
        return []
    contexts: list[dict[str, Any]] = []
    for group in ["sources", "reports", "entities", "relationships", "claims", "context"]:
        records = context_data.get(group)
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            context = _context_from_graphrag_record(group, record, method, len(contexts) + 1)
            if context:
                contexts.append(context)
            if len(contexts) >= top_k:
                return contexts
    return contexts


def _context_from_graphrag_record(group: str, record: dict[str, Any], method: str, idx: int) -> dict[str, Any] | None:
    text = _graphrag_record_text(group, record)
    if not text.strip():
        return None
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"text", "content", "full_content", "summary", "description", "all_context"}
    }
    metadata["graph_group"] = group
    return {
        "name": str(record.get("id") or record.get("human_readable_id") or record.get("title") or f"graphrag:{method}:{group}:{idx}"),
        "kind": "chunk" if group == "sources" else "tool_trace",
        "text": text,
        "score": _graphrag_record_score(record),
        "metadata": metadata,
    }


def _graphrag_record_text(group: str, record: dict[str, Any]) -> str:
    if group == "relationships":
        endpoints = " - ".join(str(record.get(key)) for key in ["source", "target"] if record.get(key))
        description = str(record.get("description") or record.get("text") or record.get("content") or "")
        return f"{endpoints}: {description}".strip(": ")
    title = str(record.get("title") or record.get("name") or "").strip()
    body = str(
        record.get("text")
        or record.get("content")
        or record.get("full_content")
        or record.get("summary")
        or record.get("description")
        or record.get("all_context")
        or ""
    ).strip()
    if title and body and title not in body:
        return f"{title}\n\n{body}"
    return body or title


def _graphrag_record_score(record: dict[str, Any]) -> float:
    for key in ["score", "rank", "weight", "occurrence weight"]:
        try:
            if record.get(key) is not None:
                return float(record[key])
        except (TypeError, ValueError):
            continue
    return 1.0


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


def _index_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prepared_input": file_identity(args.working_dir / "input" / "mutcd_chunks.txt"),
        "settings": file_identity(args.working_dir / "settings.yaml"),
        "index_prompts": {
            name: file_identity(args.working_dir / "prompts" / name)
            for name in INDEX_PROMPT_NAMES
        },
        "limit": getattr(args, "limit", None),
    }


def _index_ready(args: argparse.Namespace) -> bool:
    index_files = _index_files(args.working_dir)
    sentinel_path = args.working_dir / INDEX_SENTINEL
    sentinel = read_completion_marker(sentinel_path)
    return bool(
        (args.working_dir / "settings.yaml").is_file()
        and index_files
        and completion_marker_matches(sentinel_path, _index_identity(args))
        and _sentinel_files_present(sentinel, index_files)
    )


def _sentinel_files_present(sentinel: dict[str, Any] | None, index_files: list[str]) -> bool:
    recorded = sentinel.get("index_files") if sentinel else None
    return bool(recorded and set(recorded).issubset(index_files))


def _default_python() -> str:
    if os.getenv("GRAPHRAG_PYTHON"):
        return os.environ["GRAPHRAG_PYTHON"]
    if DEFAULT_ENV_PYTHON.exists():
        return str(DEFAULT_ENV_PYTHON)
    return sys.executable


if __name__ == "__main__":
    raise SystemExit(main())
