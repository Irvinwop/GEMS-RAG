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
DEFAULT_COMMUNITY_REPORT_TOKEN_FLOOR = 4096
DEFAULT_DRIFT_PRIMER_FOLDS = 2
DEFAULT_DRIFT_K_FOLLOWUPS = 3
DEFAULT_DRIFT_DEPTH = 1
COMMUNITY_PROMPT_NAMES = (
    "community_report_graph.txt",
    "community_report_text.txt",
)
EXTRACTION_PROMPT_NAMES = (
    "extract_graph.txt",
    "extract_claims.txt",
)
EXTRACTION_FORMAT_EXAMPLES = {
    "extract_graph.txt": """
######################
-MUTCD Format Example-
######################
Entity_types: STANDARD,CONCEPT
Text:
The MUTCD establishes uniform national criteria for traffic control devices.
######################
Output:
("entity"<|>MUTCD<|>STANDARD<|>The MUTCD establishes uniform national criteria for traffic control devices.)
##
("entity"<|>UNIFORM NATIONAL CRITERIA<|>CONCEPT<|>Uniform national criteria make traffic control devices consistent for road users.)
##
("relationship"<|>MUTCD<|>UNIFORM NATIONAL CRITERIA<|>The MUTCD establishes uniform national criteria for traffic control devices.<|>9)
<|COMPLETE|>
""".strip(),
    "extract_claims.txt": """
-MUTCD Format Example-
Entity specification: standard
Claim description: national legal status
Text: The MUTCD shall be recognized as the national standard for traffic control devices.
Output:

(MUTCD<|>NONE<|>NATIONAL STANDARD STATUS<|>TRUE<|>NONE<|>NONE<|>The MUTCD is recognized as the national standard for traffic control devices.<|>The MUTCD shall be recognized as the national standard for traffic control devices.)
<|COMPLETE|>
""".strip(),
}
EXTRACTION_OUTPUT_CONSTRAINTS = {
    "extract_graph.txt": """
-Output Constraints-
- Extract only entities explicitly named or defined in the real input. Source and chunk identifiers are not entities.
- Emit each entity once and each source-target relationship once. Never repeat or renumber a record.
- Return at most 40 total entity and relationship records, prioritizing the most important if necessary.
- Immediately output <|COMPLETE|> after the final unique record and stop generating.
""".strip(),
    "extract_claims.txt": """
-Output Constraints-
- Emit each distinct claim once. Never repeat or renumber a claim.
- Return at most 40 claims, prioritizing the most important if necessary.
- Immediately output <|COMPLETE|> after the final unique claim and stop generating.
""".strip(),
}
COMMUNITY_FINDINGS_PREFIX = "- DETAILED FINDINGS:"
COMMUNITY_FINDINGS_MIN = 2
COMMUNITY_FINDINGS_MAX = 4
COMMUNITY_FINDINGS_CACHE_MIN = 1
COMMUNITY_FINDINGS_INSTRUCTION = (
    f"- DETAILED FINDINGS: A list of {COMMUNITY_FINDINGS_MIN}-{COMMUNITY_FINDINGS_MAX} "
    "distinct key insights about the community. "
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
            sentinel = read_completion_marker(args.working_dir / INDEX_SENTINEL)
            print(
                json.dumps(
                    {
                        "error": "graphrag_index_not_ready",
                        "working_dir": str(args.working_dir),
                        "limit": args.limit,
                        "requested_community_level": args.community_level,
                        "indexed_community_levels": _indexed_community_levels(sentinel),
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
    parser.add_argument(
        "--query-llm-model",
        default=os.getenv("GRAPHRAG_QUERY_LLM_MODEL"),
        help="Optional completion-model override applied in memory during queries.",
    )
    parser.add_argument(
        "--query-embedding-model",
        default=os.getenv("GRAPHRAG_QUERY_EMBEDDING_MODEL"),
        help="Optional embedding-model override applied in memory during queries.",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Hard ceiling for each GraphRAG completion model call.",
    )
    parser.add_argument(
        "--community-report-token-floor",
        type=int,
        default=DEFAULT_COMMUNITY_REPORT_TOKEN_FLOOR,
        help=(
            "Provider-side output floor for local community-report calls. This "
            "preserves the configured cache keys while allowing deterministic "
            "structured-output retries to finish."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check whether GraphRAG imports from the cloned source tree.")
    check.add_argument("--limit", type=int, help="Expected smoke-index input limit.")
    check.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="Community-report level required by the query profile being checked.",
    )

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
        "--entity-extraction-max-tokens",
        type=int,
        default=4096,
        help="Hard completion-token ceiling for standard entity and claim extraction.",
    )
    init.add_argument(
        "--entity-extraction-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for deterministic entity and claim extraction.",
    )
    init.add_argument(
        "--entity-extraction-frequency-penalty",
        type=float,
        default=0.2,
        help="Portable repetition penalty for entity and claim extraction.",
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
        "--keep-index-prompt-examples",
        "--keep-community-prompt-examples",
        dest="keep_index_prompt_examples",
        action="store_true",
        help="Retain GraphRAG's upstream few-shot extraction and community examples.",
    )

    index = sub.add_parser("index", help="Run GraphRAG indexing.")
    index.add_argument("--method", default="standard", choices=["standard", "fast"])
    index.add_argument("--limit", type=int, help="Input limit used by the prepared smoke index.")
    index.add_argument(
        "--community-levels",
        type=_parse_community_levels,
        default=(2,),
        metavar="LEVELS|all",
        help=(
            "Comma-separated community levels to summarize, or 'all'. The default "
            "builds level 2, which is used by every default GraphRAG query profile."
        ),
    )

    query = sub.add_parser("query", help="Query an indexed GraphRAG workspace.")
    query.add_argument("--question", required=True)
    query.add_argument("--method", default="local", choices=["local", "global", "drift", "basic"])
    query.add_argument(
        "--context-only",
        action="store_true",
        help=(
            "Build and return local-search context without generating GraphRAG's "
            "answer. Intended for injected-context evaluations."
        ),
    )
    query.add_argument("--top-k", type=int, default=6, help="Maximum number of structured context records to emit in JSON mode.")
    query.add_argument("--community-level", type=int, default=2)
    query.add_argument("--dynamic-community-selection", action="store_true")
    query.add_argument("--response-type", default="Multiple Paragraphs")
    query.add_argument("--data", type=Path)
    query.add_argument("--json", action="store_true")
    query.add_argument("--limit", type=int, help="Expected smoke-index input limit.")
    query.add_argument(
        "--drift-primer-folds",
        type=int,
        default=DEFAULT_DRIFT_PRIMER_FOLDS,
        help="Number of community-report folds used to prime DRIFT search.",
    )
    query.add_argument(
        "--drift-k-followups",
        type=int,
        default=DEFAULT_DRIFT_K_FOLLOWUPS,
        help="Maximum DRIFT follow-up actions evaluated at each depth.",
    )
    query.add_argument(
        "--drift-depth",
        type=int,
        default=DEFAULT_DRIFT_DEPTH,
        help="Number of dependent DRIFT exploration steps.",
    )
    args = parser.parse_args()
    if args.llm_max_tokens is not None and args.llm_max_tokens <= 0:
        parser.error("--llm-max-tokens must be positive")
    if args.community_report_token_floor <= 0:
        parser.error("--community-report-token-floor must be positive")
    if getattr(args, "max_gleanings", None) is not None and args.max_gleanings < 0:
        parser.error("--max-gleanings must be non-negative")
    if (
        getattr(args, "community_report_max_length", None) is not None
        and args.community_report_max_length <= 0
    ):
        parser.error("--community-report-max-length must be positive")
    if (
        getattr(args, "entity_extraction_max_tokens", None) is not None
        and args.entity_extraction_max_tokens <= 0
    ):
        parser.error("--entity-extraction-max-tokens must be positive")
    if (
        getattr(args, "entity_extraction_temperature", None) is not None
        and not 0 <= args.entity_extraction_temperature <= 2
    ):
        parser.error("--entity-extraction-temperature must be between 0 and 2")
    if (
        getattr(args, "entity_extraction_frequency_penalty", None) is not None
        and not -2 <= args.entity_extraction_frequency_penalty <= 2
    ):
        parser.error("--entity-extraction-frequency-penalty must be between -2 and 2")
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
    if getattr(args, "context_only", False) and args.method != "local":
        parser.error("--context-only is supported only with --method local")
    for name in ("drift_primer_folds", "drift_k_followups", "drift_depth"):
        if getattr(args, name, 1) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpaths = [str(repo / "packages" / "graphrag"), str(ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpaths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    return env


def _parse_community_levels(value: str) -> tuple[int, ...] | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    try:
        levels = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "community levels must be comma-separated non-negative integers or 'all'"
        ) from exc
    if not levels or levels[0] < 0:
        raise argparse.ArgumentTypeError(
            "community levels must be comma-separated non-negative integers or 'all'"
        )
    return levels


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
    community_level_available = _query_community_level_available(args, sentinel)
    environment_ready = args.repo.exists() and cli_runnable
    index_ready = bool(
        settings_found
        and index_files
        and sentinel_matches_input
        and sentinel_files_present
        and community_level_available
    )
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
        "indexed_community_levels": _indexed_community_levels(sentinel),
        "required_community_level": args.community_level,
        "community_level_available": community_level_available,
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
    removed_truncations = _remove_truncated_index_cache_entries(args.working_dir)
    if removed_truncations:
        print(
            json.dumps(
                {
                    "removed_truncated_index_cache_entries": len(removed_truncations),
                    "sample": removed_truncations[:20],
                }
            )
        )
    removed_invalid_reports = _remove_invalid_community_report_cache_entries(
        args.working_dir
    )
    if removed_invalid_reports:
        print(
            json.dumps(
                {
                    "removed_invalid_community_report_cache_entries": len(
                        removed_invalid_reports
                    ),
                    "sample": removed_invalid_reports[:20],
                }
            )
        )
    community_levels = getattr(args, "community_levels", (2,))
    print(
        json.dumps(
            {
                "community_report_levels": (
                    "all" if community_levels is None else list(community_levels)
                )
            }
        )
    )
    code = _run_graphrag(
        args,
        env,
        ["index", "--root", str(args.working_dir), "--method", args.method],
        community_levels=community_levels,
    )
    if code != 0:
        return code
    truncated = _truncated_index_cache_entries(args.working_dir)
    if truncated:
        print(
            json.dumps(
                {
                    "error": "graphrag_index_completion_truncated",
                    "count": len(truncated),
                    "sample": truncated[:20],
                }
            ),
            file=sys.stderr,
        )
        return 2
    invalid_reports = _invalid_community_report_cache_entries(args.working_dir)
    if invalid_reports:
        print(
            json.dumps(
                {
                    "error": "graphrag_index_invalid_community_reports",
                    "count": len(invalid_reports),
                    "sample": invalid_reports[:20],
                }
            ),
            file=sys.stderr,
        )
        return 2
    index_files = _index_files(args.working_dir)
    if not index_files:
        print(json.dumps({"error": "graphrag_index_produced_no_artifacts"}), file=sys.stderr)
        return 2
    publish_completion_marker(
        sentinel_path,
        _index_identity(args),
        method=args.method,
        index_files=index_files,
        community_levels=(
            "all" if community_levels is None else list(community_levels)
        ),
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
    entity_extraction_max_tokens = getattr(args, "entity_extraction_max_tokens", None)
    entity_extraction_temperature = getattr(args, "entity_extraction_temperature", None)
    entity_extraction_frequency_penalty = getattr(
        args, "entity_extraction_frequency_penalty", None
    )
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
        or entity_extraction_max_tokens is not None
        or entity_extraction_temperature is not None
        or entity_extraction_frequency_penalty is not None
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
            entity_extraction_max_tokens=entity_extraction_max_tokens,
            entity_extraction_temperature=entity_extraction_temperature,
            entity_extraction_frequency_penalty=entity_extraction_frequency_penalty,
            community_report_max_length=community_report_max_length,
            community_report_max_tokens=community_report_max_tokens,
            community_report_temperature=community_report_temperature,
        )
        if code != 0:
            return code
    if not getattr(args, "keep_index_prompt_examples", False):
        return _sanitize_index_prompts(args.working_dir)
    return 0


def _sanitize_index_prompts(working_dir: Path) -> int:
    code = _sanitize_extraction_prompts(working_dir)
    if code != 0:
        return code
    return _sanitize_community_prompts(working_dir)


def _sanitize_extraction_prompts(working_dir: Path) -> int:
    sanitized: list[str] = []
    try:
        for name in EXTRACTION_PROMPT_NAMES:
            path = working_dir / "prompts" / name
            original = path.read_text(encoding="utf-8")
            without_examples = _remove_prompt_section(
                original,
                example_marker="-Examples-",
                real_data_marker="-Real Data-",
            )
            replacement = (
                f"{EXTRACTION_OUTPUT_CONSTRAINTS[name]}\n\n"
                f"{EXTRACTION_FORMAT_EXAMPLES[name]}"
            )
            with_grounded_example = _insert_before_marker(
                without_examples,
                marker="-Real Data-",
                insertion=replacement,
            )
            _atomic_write_text(path, with_grounded_example)
            sanitized.append(str(path))
    except Exception as exc:
        print(json.dumps({"error": "sanitize_graphrag_prompts_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"extraction_prompt_examples_replaced": True, "prompts": sanitized}))
    return 0


def _sanitize_community_prompts(working_dir: Path) -> int:
    sanitized: list[str] = []
    try:
        for name in COMMUNITY_PROMPT_NAMES:
            path = working_dir / "prompts" / name
            original = path.read_text(encoding="utf-8")
            without_example = _remove_few_shot_example(original)
            compact = _compact_community_prompt(without_example)
            _atomic_write_text(path, compact)
            sanitized.append(str(path))
    except Exception as exc:
        print(json.dumps({"error": "sanitize_graphrag_prompts_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"community_prompt_examples_removed": True, "prompts": sanitized}))
    return 0


def _remove_few_shot_example(prompt: str) -> str:
    return _remove_prompt_section(
        prompt,
        example_marker="# Example Input",
        real_data_marker="# Real Data",
    )


def _remove_prompt_section(
    prompt: str,
    *,
    example_marker: str,
    real_data_marker: str,
) -> str:
    example_start = prompt.find(example_marker)
    real_data_start = prompt.find(real_data_marker, example_start + len(example_marker))
    if example_start < 0 or real_data_start < 0:
        raise ValueError(
            f"prompt does not contain expected {example_marker!r} and {real_data_marker!r} markers"
        )
    example_start = _include_preceding_divider(prompt, example_start)
    real_data_start = _include_preceding_divider(prompt, real_data_start)
    return f"{prompt[:example_start].rstrip()}\n\n{prompt[real_data_start:].lstrip()}"


def _include_preceding_divider(prompt: str, marker_start: int) -> int:
    line_start = prompt.rfind("\n", 0, marker_start) + 1
    previous_end = max(0, line_start - 1)
    previous_start = prompt.rfind("\n", 0, previous_end) + 1
    previous_line = prompt[previous_start:previous_end].strip()
    if previous_line and set(previous_line) == {"#"}:
        return previous_start
    return line_start


def _insert_before_marker(prompt: str, *, marker: str, insertion: str) -> str:
    marker_start = prompt.find(marker)
    if marker_start < 0:
        raise ValueError(f"prompt does not contain expected {marker!r} marker")
    marker_start = _include_preceding_divider(prompt, marker_start)
    return (
        f"{prompt[:marker_start].rstrip()}\n\n"
        f"{insertion.strip()}\n\n"
        f"{prompt[marker_start:].lstrip()}"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _truncated_index_cache_entries(working_dir: Path) -> list[str]:
    truncated: list[str] = []
    for partition in _index_completion_cache_partitions(working_dir):
        cache_dir = working_dir / "cache" / partition
        if not cache_dir.is_dir():
            continue
        for path in cache_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            result = payload.get("result") if isinstance(payload, dict) else None
            response = result.get("response") if isinstance(result, dict) else None
            choices = response.get("choices", []) if isinstance(response, dict) else []
            if any(
                isinstance(choice, dict) and choice.get("finish_reason") == "length"
                for choice in choices
            ):
                truncated.append(str(path.relative_to(working_dir)))
    return sorted(truncated)


def _remove_truncated_index_cache_entries(working_dir: Path) -> list[str]:
    truncated = _truncated_index_cache_entries(working_dir)
    for relative in truncated:
        (working_dir / relative).unlink(missing_ok=True)
    return truncated


def _invalid_community_report_cache_entries(working_dir: Path) -> list[str]:
    partition = _index_completion_cache_partition_map(working_dir).get(
        "community_reports"
    )
    if partition is None:
        return []
    cache_dir = working_dir / "cache" / partition
    if not cache_dir.is_dir():
        return []
    invalid: list[str] = []
    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(str(path.relative_to(working_dir)))
            continue
        if not _valid_community_report_cache_payload(payload):
            invalid.append(str(path.relative_to(working_dir)))
    return sorted(invalid)


def _valid_community_report_cache_payload(payload: Any) -> bool:
    result = payload.get("result") if isinstance(payload, dict) else None
    response = result.get("response") if isinstance(result, dict) else None
    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, str):
        return False
    try:
        report = json.loads(content)
    except json.JSONDecodeError:
        return False
    findings = report.get("findings") if isinstance(report, dict) else None
    if (
        not isinstance(findings, list)
        or len(findings) < COMMUNITY_FINDINGS_CACHE_MIN
    ):
        return False
    return all(
        isinstance(finding, dict)
        and isinstance(finding.get("summary"), str)
        and bool(finding["summary"].strip())
        and isinstance(finding.get("explanation"), str)
        and bool(finding["explanation"].strip())
        for finding in findings
    )


def _remove_invalid_community_report_cache_entries(working_dir: Path) -> list[str]:
    invalid = _invalid_community_report_cache_entries(working_dir)
    for relative in invalid:
        (working_dir / relative).unlink(missing_ok=True)
    return invalid


def _index_completion_cache_partitions(working_dir: Path) -> list[str]:
    return list(_index_completion_cache_partition_map(working_dir).values())


def _index_completion_cache_partition_map(working_dir: Path) -> dict[str, str]:
    settings_path = working_dir / "settings.yaml"
    if not settings_path.is_file():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    partitions: dict[str, str] = {}
    for section_name in (
        "extract_graph",
        "summarize_descriptions",
        "extract_claims",
        "community_reports",
    ):
        section = payload.get(section_name) if isinstance(payload, dict) else None
        partition = section.get("model_instance_name") if isinstance(section, dict) else None
        if (
            isinstance(partition, str)
            and partition
            and Path(partition).name == partition
            and partition not in partitions.values()
        ):
            partitions[section_name] = partition
    return partitions


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
    entity_extraction_max_tokens: int | None = None,
    entity_extraction_temperature: float | None = None,
    entity_extraction_frequency_penalty: float | None = None,
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
            entity_extraction_max_tokens is not None
            or entity_extraction_temperature is not None
            or entity_extraction_frequency_penalty is not None
        ):
            completion_models = payload.get("completion_models")
            if not isinstance(completion_models, dict) or not completion_models:
                raise ValueError(f"missing completion_models in {settings_path}")
            source_model = completion_models.get("default_completion_model") or next(
                iter(completion_models.values())
            )
            if not isinstance(source_model, dict):
                raise ValueError("completion model must be a mapping")
            extraction_model = copy.deepcopy(source_model)
            extraction_call_args = extraction_model.get("call_args") or {}
            if not isinstance(extraction_call_args, dict):
                raise ValueError("entity extraction model call_args must be a mapping")
            if entity_extraction_max_tokens is not None:
                extraction_call_args["max_tokens"] = entity_extraction_max_tokens
            if entity_extraction_temperature is not None:
                extraction_call_args["temperature"] = entity_extraction_temperature
            if entity_extraction_frequency_penalty is not None:
                extraction_call_args["frequency_penalty"] = (
                    entity_extraction_frequency_penalty
                )
            extraction_model["call_args"] = extraction_call_args
            extraction_model_id = "entity_extraction_completion_model"
            completion_models[extraction_model_id] = extraction_model
            for section_name in EXTRACTION_PROMPT_NAMES:
                config_name = section_name.removesuffix(".txt")
                config = payload.get(config_name)
                if isinstance(config, dict):
                    config["completion_model_id"] = extraction_model_id
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
                "entity_extraction_max_tokens": entity_extraction_max_tokens,
                "entity_extraction_temperature": entity_extraction_temperature,
                "entity_extraction_frequency_penalty": entity_extraction_frequency_penalty,
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


def _run_graphrag(
    args: argparse.Namespace,
    env: dict[str, str],
    command: list[str],
    *,
    community_levels: tuple[int, ...] | None = None,
) -> int:
    completed = _graphrag_subprocess(
        args,
        env,
        command,
        community_levels=community_levels,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


def _graphrag_subprocess(
    args: argparse.Namespace,
    env: dict[str, str],
    command: list[str],
    *,
    community_levels: tuple[int, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    bootstrap = "from graphrag.cli.main import app; app()"
    bootstrap_args: list[str] = []
    if community_levels is not None:
        bootstrap = """
import json
import sys

from gems_rag.graphrag_indexing import (
    install_community_report_level_filter,
    install_community_report_token_floor,
)

request = json.loads(sys.argv.pop(1))
if request["community_report_token_floor"] is not None:
    install_community_report_token_floor(request["community_report_token_floor"])
install_community_report_level_filter(request["community_levels"])
from graphrag.cli.main import app
app()
""".strip()
        bootstrap_args.append(
            json.dumps(
                {
                    "community_levels": list(community_levels),
                    "community_report_token_floor": (
                        getattr(
                            args,
                            "community_report_token_floor",
                            DEFAULT_COMMUNITY_REPORT_TOKEN_FLOOR,
                        )
                        if getattr(args, "allow_missing_api_key", False)
                        else None
                    ),
                }
            )
        )
    return subprocess.run(
        [args.python, "-c", bootstrap, *bootstrap_args, *command],
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
        "base_url": getattr(args, "base_url", None),
        "embedding_base_url": (
            getattr(args, "embedding_base_url", None)
            or getattr(args, "base_url", None)
        ),
        "llm_model": getattr(args, "query_llm_model", None),
        "embedding_model": getattr(args, "query_embedding_model", None),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "llm_max_tokens": getattr(args, "llm_max_tokens", None),
        "context_only": bool(getattr(args, "context_only", False)),
        "drift_budget": {
            "primer_folds": getattr(
                args,
                "drift_primer_folds",
                DEFAULT_DRIFT_PRIMER_FOLDS,
            ),
            "k_followups": getattr(
                args,
                "drift_k_followups",
                DEFAULT_DRIFT_K_FOLLOWUPS,
            ),
            "n_depth": getattr(args, "drift_depth", DEFAULT_DRIFT_DEPTH),
        },
    }
    code = r"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from graphrag.cli import query as query_cli
from graphrag.utils.api import reformat_context_data

request = json.loads(__import__("sys").argv[1])
data_dir = Path(request["data"]) if request.get("data") else None
root_dir = Path(request["root"])
method = request["method"]
captured = io.StringIO()
upstream_load_config = query_cli.load_config

def load_config_with_runtime_backend(*args, **kwargs):
    config = upstream_load_config(*args, **kwargs)
    for model in config.completion_models.values():
        if request.get("base_url"):
            model.api_base = request["base_url"]
        if request.get("llm_model"):
            model.model = request["llm_model"]
        if request.get("reasoning_effort") or request.get("llm_max_tokens"):
            call_args = dict(model.call_args)
            if request.get("reasoning_effort"):
                call_args["reasoning_effort"] = request["reasoning_effort"]
            if request.get("llm_max_tokens"):
                call_args["max_tokens"] = int(request["llm_max_tokens"])
            model.call_args = call_args
    for model in config.embedding_models.values():
        if request.get("embedding_base_url"):
            model.api_base = request["embedding_base_url"]
        if request.get("embedding_model"):
            model.model = request["embedding_model"]
    if method == "drift":
        budget = request["drift_budget"]
        config.drift_search.primer_folds = int(budget["primer_folds"])
        config.drift_search.drift_k_followups = int(budget["k_followups"])
        config.drift_search.n_depth = int(budget["n_depth"])
    return config

query_cli.load_config = load_config_with_runtime_backend

if request.get("context_only"):
    from graphrag.query.structured_search.local_search.search import LocalSearch

    async def context_only_stream_search(self, query, conversation_history=None):
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )
        for callback in self.callbacks:
            callback.on_context(context_result.context_records)
        if False:
            yield ""

    LocalSearch.stream_search = context_only_stream_search

with redirect_stdout(captured):
    if method == "local":
        response, context_data = query_cli.run_local_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=int(request["community_level"]),
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    elif method == "global":
        response, context_data = query_cli.run_global_search(
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
        response, context_data = query_cli.run_drift_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=int(request["community_level"]),
            response_type=request["response_type"],
            streaming=False,
            query=request["question"],
            verbose=False,
        )
    elif method == "basic":
        response, context_data = query_cli.run_basic_search(
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
    if getattr(args, "context_only", False):
        response["context_only"] = True
    if args.community_level is not None:
        response["community_level"] = args.community_level
    if args.dynamic_community_selection:
        response["dynamic_community_selection"] = True
    if args.method == "drift":
        response["drift_budget"] = {
            "primer_folds": getattr(
                args,
                "drift_primer_folds",
                DEFAULT_DRIFT_PRIMER_FOLDS,
            ),
            "k_followups": getattr(
                args,
                "drift_k_followups",
                DEFAULT_DRIFT_K_FOLLOWUPS,
            ),
            "n_depth": getattr(args, "drift_depth", DEFAULT_DRIFT_DEPTH),
        }
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
        "community_report_token_floor": (
            getattr(
                args,
                "community_report_token_floor",
                DEFAULT_COMMUNITY_REPORT_TOKEN_FLOOR,
            )
            if getattr(args, "allow_missing_api_key", False)
            else None
        ),
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
        and _query_community_level_available(args, sentinel)
    )


def _indexed_community_levels(sentinel: dict[str, Any] | None) -> list[int] | str | None:
    if not sentinel:
        return None
    levels = sentinel.get("community_levels")
    if levels is None:
        return "all"
    if levels == "all":
        return "all"
    if not isinstance(levels, list):
        return None
    try:
        return sorted({int(level) for level in levels})
    except (TypeError, ValueError):
        return None


def _query_community_level_available(
    args: argparse.Namespace,
    sentinel: dict[str, Any] | None,
) -> bool:
    command = getattr(args, "command", None)
    if command not in {"check", "query"}:
        return True
    if command == "query" and getattr(args, "method", None) == "basic":
        return True
    indexed = _indexed_community_levels(sentinel)
    if indexed in (None, "all"):
        return True
    requested = getattr(args, "community_level", None)
    return requested is None or int(requested) in indexed


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
