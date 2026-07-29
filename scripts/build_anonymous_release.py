#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "anonymous_release"
DEFAULT_OUTPUT = ROOT / "data" / "working" / "mutcd-rag-anonymous-release"
DEFAULT_GEMS_RAG_ASSETS = (
    ROOT
    / "data"
    / "extracted"
    / "MRAG-20260715T174043Z-1"
    / "MRAG"
)
GEMS_RAG_SOURCE = ROOT / "external" / "MRAG_stp2"
GRAPHRAG_SOURCE = (
    ROOT / "external" / "rag-implementations" / "graphrag"
)
PAPERQA_SOURCE = (
    ROOT / "external" / "rag-implementations" / "paper-qa"
)
GRAPHRAG_INDEX = ROOT / "data" / "working" / "graphrag_index"
PAPERQA_INDEX = ROOT / "data" / "working" / "paperqa_index"
CORPUS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"
GOLD = ROOT / "data" / "benchmark" / "mutcd_benchmark_gold_v1.jsonl"
GRADER_SPEC = ROOT / "docs" / "MUTCD_RAG_EVALUATION_SPECIFICATION.md"
QUESTIONS = (
    GEMS_RAG_SOURCE
    / "benchmarks"
    / "mutcd150"
    / "v1"
    / "mutcd_benchmark_questions_v1.jsonl"
)
SUPPORT_MODULES = (
    "data.py",
    "endpoint.py",
    "graphrag_indexing.py",
    "index_completion.py",
    "mrag_reference_modes.py",
    "mrag_reference_server.py",
    "retrieval_metrics.py",
    "types.py",
)
FORBIDDEN_BYTES = {
    b"/Users/": "local macOS home path",
    b"\\\\Users\\\\": "local Windows home path",
    b"Irvinwop": "local account identity",
    b"hannanazad": "source repository identity",
    b"MRAG_stp2": "source repository name",
    b"sk-ant-api": "Anthropic API key",
    b"sk-proj-": "OpenAI API key",
    b"anthropic_test_key": "private credential variable",
    b"anthropic_prod_key": "private credential variable",
    b"graphrag_local": "historical method ID",
    b"graphrag-local": "historical method ID",
    b"paperqa2_chunks": "historical method ID",
    b"gems_full": "historical method ID",
}
PUBLIC_LOCAL_EXECUTION_BYTES = {
    b"ollama": "local execution reference",
    b"nomic": "local execution reference",
    b"qwen2.5:": "local-style Qwen tag",
    b"127.0.0.1": "loopback endpoint",
    b"localhost": "loopback endpoint",
    b"local model": "local-model prose",
    b"model weights": "downloaded-weight prose",
    b"local openai-compatible": "local endpoint prose",
    b"huggingface/": "direct local provider route",
}
COMPACT_GEMS_RAG_COLLECTIONS = {
    "mutcd_chunks",
    "mutcd_figures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the anonymous MUTCD comparison source and built indexes "
            "into one portable folder."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gems-rag-assets",
        type=Path,
        default=DEFAULT_GEMS_RAG_ASSETS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output and abandoned staging folder.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if size >= 100 * 1024 * 1024:
        log(f"  copying {destination.name} ({format_size(size)})")
    shutil.copy2(source, destination)
    return destination


def source_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if (
            name in {
                ".git",
                ".github",
                ".vscode",
                "__pycache__",
                "tests",
                "test",
                ".pytest_cache",
                "notebooks",
                "example_notebooks",
                "examples",
            }
            or name.endswith(".egg-info")
            or name.endswith(".pyc")
            or name.endswith(".ipynb")
            or name == ".DS_Store"
        ):
            ignored.add(name)
    return ignored


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=source_ignore,
        copy_function=lambda src, dst: str(copy_file(Path(src), Path(dst))),
        dirs_exist_ok=True,
    )


def copy_template(stage: Path) -> None:
    log("Copying release templates")
    copy_tree(TEMPLATE, stage)


def copy_selected_roots(
    source: Path,
    destination: Path,
    *,
    files: tuple[str, ...],
    directories: tuple[str, ...],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = source / name
        if path.is_file():
            copy_file(path, destination / name)
    for name in directories:
        path = source / name
        if path.is_dir():
            copy_tree(path, destination / name)


def copy_third_party(stage: Path) -> dict[str, str]:
    log("Copying the GraphRAG and PaperQA source required by the adapters")
    copy_selected_roots(
        GRAPHRAG_SOURCE,
        stage / "third_party" / "graphrag",
        files=(
            "CHANGELOG.md",
            "LICENSE",
            "README.md",
            "pyproject.toml",
        ),
        directories=("packages",),
    )
    copy_selected_roots(
        PAPERQA_SOURCE,
        stage / "third_party" / "paperqa",
        files=(
            "CITATION.cff",
            "LICENSE",
            "pyproject.toml",
        ),
        directories=("src",),
    )
    paperqa_version_path = (
        stage / "third_party" / "paperqa" / "src" / "paperqa" / "version.py"
    )
    version_match = re.search(
        r"^__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)",
        paperqa_version_path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if version_match is None:
        raise ValueError("could not determine the packaged PaperQA version")
    paperqa_version = version_match.group(1)
    pyproject = stage / "third_party" / "paperqa" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    marker = "[tool.setuptools_scm]\n"
    if marker not in text:
        raise ValueError("PaperQA pyproject is missing [tool.setuptools_scm]")
    text = text.replace(
        marker,
        f'{marker}fallback_version = "{paperqa_version}"\n',
        1,
    )
    pyproject.write_text(text, encoding="utf-8")
    return {
        "graphrag": git_revision(GRAPHRAG_SOURCE),
        "paperqa": git_revision(PAPERQA_SOURCE),
        "paperqa_version": paperqa_version,
    }


def copy_gems_rag_source(stage: Path) -> None:
    log("Copying the latest curated GEMS-RAG source")
    source_root = stage / "gems-rag"
    copy_tree(GEMS_RAG_SOURCE / "mrag", source_root / "mrag")
    for obsolete in (
        source_root / "mrag" / "figures_v1_deprecated.py",
        source_root / "mrag" / "kg_v1_deprecated.py",
    ):
        obsolete.unlink(missing_ok=True)

    for name in (
        "__init__.py",
        "extract_figures.py",
        "ingest_v3.py",
        "ingest_v4.py",
        "verify_mutcd_benchmark_assets.sh",
    ):
        source = GEMS_RAG_SOURCE / "scripts" / name
        if source.is_file():
            copy_file(source, source_root / "scripts" / name)
    for name in ("architecture.md", "REBUILD_V4.md", "kg_fig2B-1.png"):
        source = GEMS_RAG_SOURCE / "docs" / name
        if source.is_file():
            copy_file(source, source_root / "docs" / name)
    copy_file(
        GEMS_RAG_SOURCE / "requirements.txt",
        source_root / "requirements.txt",
    )

    colab_setup = source_root / "mrag" / "colab_setup.py"
    text = colab_setup.read_text(encoding="utf-8")
    text = text.replace(
        "!pip install -q git+https://github.com/hannanazad/MRAG.git",
        "!pip install -q -e .",
    )
    colab_setup.write_text(text, encoding="utf-8")

    config = source_root / "mrag" / "config.py"
    text = config.read_text(encoding="utf-8")
    old = '''def _default_hf_home(env: str, base: Path) -> Path:
    if env == "colab":
        # Drive HF cache survives session restarts.
        return base / "hf_cache"
    if env == "hprc":
        return Path(os.environ["SCRATCH"]) / "hf_cache"
    return base / "hf_cache"
'''
    new = '''def _default_hf_home(env: str, base: Path) -> Path:
    configured = os.environ.get("MRAG_HF_HOME") or os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser()
    if env == "colab":
        # Drive HF cache survives session restarts.
        return base / "hf_cache"
    if env == "hprc":
        return Path(os.environ["SCRATCH"]) / "hf_cache"
    return base / "hf_cache"
'''
    if old not in text:
        raise ValueError(f"expected Hugging Face cache function is missing in {config}")
    config.write_text(text.replace(old, new, 1), encoding="utf-8")


def transform_script(
    source: Path,
    destination: Path,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise ValueError(f"expected source text is missing in {source}: {old!r}")
        text = text.replace(old, new)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def copy_pipelines_and_support(stage: Path) -> None:
    log("Extracting the comparison adapters and shared support code")
    support = stage / "src" / "comparison_support"
    support.mkdir(parents=True, exist_ok=True)
    (support / "__init__.py").write_text(
        '"""Shared support code for the MUTCD comparison adapters."""\n',
        encoding="utf-8",
    )
    for name in SUPPORT_MODULES:
        source = ROOT / "src" / "gems_rag" / name
        text = source.read_text(encoding="utf-8")
        text = text.replace("gems_rag", "comparison_support")
        if "mrag_reference" in name:
            text = text.replace("mrag_reference", "gems_rag_reference")
            text = text.replace("MragReference", "GemsRagReference")
            text = text.replace("mrag_dir", "gems_rag_dir")
            text = text.replace("MRAG ", "GEMS-RAG ")
            text = text.replace(
                "ColQwen/ColPali query encoder",
                "visual query encoder",
            )
        destination_name = name.replace("mrag_reference", "gems_rag_reference")
        (support / destination_name).write_text(text, encoding="utf-8")

    transform_script(
        ROOT / "scripts" / "query_graphrag_index.py",
        stage / "pipelines" / "query_graphrag.py",
        (
            ("from gems_rag.", "from comparison_support."),
            (
                'DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "graphrag"',
                'DEFAULT_REPO = ROOT / "third_party" / "graphrag"',
            ),
            (
                'DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"',
                'DEFAULT_CHUNKS = ROOT / "indexes" / "corpus" / "chunks.jsonl"',
            ),
            (
                'DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "graphrag_index"',
                'DEFAULT_WORKING_DIR = ROOT / "indexes" / "graphrag"',
            ),
            (
                'DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "graphrag" / "bin" / "python"',
                'DEFAULT_ENV_PYTHON = ROOT / ".venv-graphrag" / "bin" / "python"',
            ),
            (
                'parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")',
                'parser.add_argument("--allow-missing-api-key", action="store_true", help=argparse.SUPPRESS)',
            ),
            (
                'parser.add_argument("--base-url", default=os.getenv("GRAPHRAG_API_BASE") or os.getenv("OPENAI_BASE_URL"))',
                'parser.add_argument("--base-url", default=os.getenv("GRAPHRAG_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")',
            ),
            (
                'help="Optional separate OpenAI-compatible embedding endpoint; defaults to --base-url.",',
                'help="Optional separate OpenAI API embedding endpoint; defaults to --base-url.",',
            ),
            (
                'help="Optional follow-up extraction passes per chunk; zero avoids prompt-example leakage with small local models.",',
                'help="Optional follow-up extraction passes per chunk; zero avoids prompt-example leakage in constrained extraction settings.",',
            ),
            ('api_key = "local"', 'api_key = "placeholder"'),
            (
                'api_key=api_key or ("local" if args.allow_missing_api_key else None)',
                'api_key=api_key or ("placeholder" if args.allow_missing_api_key else None)',
            ),
            (
                'INDEX_SENTINEL = ".gems_rag_graphrag_index.json"',
                'INDEX_SENTINEL = ".graphrag_index.json"',
            ),
            (
                '        "root": str(args.working_dir),\n        "method": args.method,',
                '        "root": str(args.working_dir),\n        "reporting_dir": str(ROOT / "runs" / "graphrag"),\n        "method": args.method,',
            ),
            (
                "    config = upstream_load_config(*args, **kwargs)\n    for model in config.completion_models.values():",
                '    config = upstream_load_config(*args, **kwargs)\n    reporting_dir = Path(request["reporting_dir"])\n    reporting_dir.mkdir(parents=True, exist_ok=True)\n    config.reporting.base_dir = str(reporting_dir)\n    for model in config.completion_models.values():',
            ),
            (
                'print(json.dumps({"question": args.question, "method": args.method, "top_k": args.top_k, "result": stdout}, ensure_ascii=False))',
                'print(json.dumps({"question": args.question, "method": "graphrag", "query_algorithm": args.method, "top_k": args.top_k, "result": stdout}, ensure_ascii=False))',
            ),
            (
                '        "method": args.method,\n        "top_k": args.top_k,\n        "response_type": args.response_type,',
                '        "method": "graphrag",\n        "query_algorithm": args.method,\n        "top_k": args.top_k,\n        "response_type": args.response_type,',
            ),
            (
                'or f"graphrag:{method}:{group}:{idx}")',
                'or f"graphrag:{group}:{idx}")',
            ),
        ),
    )
    transform_script(
        ROOT / "scripts" / "query_paperqa_index.py",
        stage / "pipelines" / "query_paperqa.py",
        (
            ("from gems_rag.", "from comparison_support."),
            (
                'DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "paper-qa"',
                'DEFAULT_REPO = ROOT / "third_party" / "paperqa"',
            ),
            (
                'DEFAULT_CHUNKS = ROOT / "data" / "working" / "mrag_corpus" / "chunks.jsonl"',
                'DEFAULT_CHUNKS = ROOT / "indexes" / "corpus" / "chunks.jsonl"',
            ),
            (
                'DEFAULT_INDEX = ROOT / "data" / "working" / "paperqa_index" / "docs.pkl"',
                'DEFAULT_INDEX = ROOT / "indexes" / "paperqa" / "docs.pkl"',
            ),
            (
                'DEFAULT_NATIVE_INDEX = ROOT / "data" / "working" / "paperqa_index" / "docs-native-pdf.pkl"',
                'DEFAULT_NATIVE_INDEX = ROOT / "indexes" / "paperqa" / "docs-native-pdf.pkl"',
            ),
            (
                'DEFAULT_PDF = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG" / "mutcd11theditionr1hl.pdf"',
                'DEFAULT_PDF = ROOT / "indexes" / "gems-rag" / "mutcd11theditionr1hl.pdf"',
            ),
            (
                'parser.add_argument("--allow-missing-api-key", action="store_true", help="Use a dummy local key when targeting a local OpenAI-compatible server.")',
                'parser.add_argument("--allow-missing-api-key", action="store_true", help=argparse.SUPPRESS)',
            ),
            (
                'parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="Optional OpenAI-compatible base URL, exported as OPENAI_BASE_URL for PaperQA providers.")',
                'parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI API base URL.")',
            ),
            ('api_key = "local"', 'api_key = "placeholder"'),
            (
                'api_key=api_key or ("local" if args.allow_missing_api_key else None)',
                'api_key=api_key or ("placeholder" if args.allow_missing_api_key else None)',
            ),
            (
                'if model.startswith(("openai/", "azure/", "ollama/", "huggingface/")):',
                'if model.startswith(("openai/", "azure/")):',
            ),
            (
                'INDEX_SENTINEL_SUFFIX = ".gems_rag_ready.json"',
                'INDEX_SENTINEL_SUFFIX = ".ready.json"',
            ),
            ("PaperQA2", "PaperQA"),
            ("paperqa2_context", "paperqa_context"),
            (
                '                    "question": args.question,\n                    "top_k": args.top_k,',
                '                    "method": "paperqa",\n                    "question": args.question,\n                    "top_k": args.top_k,',
            ),
        ),
    )
    transform_script(
        ROOT / "scripts" / "query_mrag_reference.py",
        stage / "pipelines" / "query_gems_rag.py",
        (
            ("from gems_rag.", "from comparison_support."),
            (
                'DEFAULT_REPO = ROOT / "external" / "MRAG_stp2"',
                'DEFAULT_REPO = ROOT / "gems-rag"',
            ),
            (
                'DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"',
                'DEFAULT_MRAG_DIR = ROOT / "indexes" / "gems-rag"',
            ),
            (
                'DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "mrag-reference" / "bin" / "python"',
                'DEFAULT_ENV_PYTHON = ROOT / ".venv-gems-rag" / "bin" / "python"',
            ),
            (
                'DEFAULT_SERVER_DIR = ROOT / "data" / "working" / "mrag-reference-server"',
                'DEFAULT_SERVER_DIR = ROOT / "runs" / "gems-rag-server"',
            ),
            (
                'description="Query the cloned hannanazad/MRAG_stp2 retrieval stack."',
                'description="Query the GEMS-RAG retrieval stack."',
            ),
            (
                'parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)',
                'parser.add_argument("--gems-rag-dir", dest="mrag_dir", type=Path, default=DEFAULT_MRAG_DIR)',
            ),
            (
                '    os.environ["MRAG_BASE_DIR"] = str(args.mrag_dir)\n    if str(args.repo) not in sys.path:',
                '    os.environ["MRAG_BASE_DIR"] = str(args.mrag_dir)\n    os.environ.setdefault(\n        "MRAG_HF_HOME", os.environ.get("HF_HOME", str(ROOT / "runs" / "cache"))\n    )\n    if str(args.repo) not in sys.path:',
            ),
            (
                "        print(json.dumps(result, ensure_ascii=False))",
                '        print(json.dumps({"method": "gems-rag", **result}, ensure_ascii=False))',
            ),
            (
                '            {\n                "question": args.question,\n                "chunks": result["chunks"],',
                '            {\n                "method": "gems-rag",\n                "question": args.question,\n                "chunks": result["chunks"],',
            ),
            (
                "Install external/MRAG_stp2/requirements.txt",
                "Install gems-rag/requirements.txt",
            ),
            (
                'ROOT / "src" / "gems_rag" / "mrag_reference_modes.py"',
                'ROOT / "src" / "comparison_support" / "mrag_reference_modes.py"',
            ),
            (
                'ROOT / "src" / "gems_rag" / "mrag_reference_server.py"',
                'ROOT / "src" / "comparison_support" / "mrag_reference_server.py"',
            ),
            ("_gems_rag_dtype_compat", "_comparison_dtype_compat"),
            ("mrag_reference", "gems_rag_reference"),
            ("MragReference", "GemsRagReference"),
            ("MRAG_REFERENCE_PYTHON", "GEMS_RAG_PYTHON"),
            ("DEFAULT_MRAG_DIR", "DEFAULT_GEMS_RAG_DIR"),
            ("mrag_dir", "gems_rag_dir"),
            ("--mrag-dir", "--gems-rag-dir"),
            ("data/working/venvs/mrag-reference", ".venv-gems-rag"),
            ("mrag.sock", "gems-rag.sock"),
            ("MRAG ", "GEMS-RAG "),
            ("an GEMS-RAG", "a GEMS-RAG"),
            (
                "Reuse an auto-started local GEMS-RAG worker instead of reloading model weights per query.",
                "Reuse an auto-started GEMS-RAG worker instead of reinitializing retrieval state per query.",
            ),
        ),
    )

    for path in (
        stage / "pipelines" / "query_bm25.py",
        stage / "pipelines" / "query_graphrag.py",
        stage / "pipelines" / "query_paperqa.py",
        stage / "pipelines" / "query_gems_rag.py",
        stage / "pipelines" / "build_indexes_openai.py",
        stage / "pipelines" / "run_comparison.py",
        stage / "pipelines" / "score_retrieval.py",
        stage / "scripts" / "package_upload.py",
        stage / "scripts" / "setup_environments.sh",
    ):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_benchmark(stage: Path) -> tuple[int, int]:
    log("Copying the benchmark, gold annotations, and grader specification")
    benchmark = stage / "benchmark"
    copy_file(QUESTIONS, benchmark / "questions.jsonl")
    copy_file(GOLD, benchmark / "gold.jsonl")
    copy_file(
        GRADER_SPEC,
        benchmark / "MUTCD_RAG_EVALUATION_SPECIFICATION.md",
    )
    questions = list(read_jsonl(QUESTIONS))
    gold = list(read_jsonl(GOLD))
    question_ids = {
        str(row.get("question_id") or row.get("qa_id") or "")
        for row in questions
    }
    gold_ids = {
        str(row.get("question_id") or row.get("qa_id") or "")
        for row in gold
    }
    if len(questions) != 150 or len(gold) != 150 or question_ids != gold_ids:
        raise ValueError(
            "benchmark/gold identity mismatch: "
            f"{len(questions)} questions, {len(gold)} gold rows"
        )
    return len(questions), sum(row.get("answerable") is False for row in gold)


def copy_corpus(stage: Path) -> int:
    log("Copying the shared canonical corpus")
    destination = stage / "indexes" / "corpus" / "chunks.jsonl"
    copy_file(CORPUS, destination)
    count = sum(1 for _ in read_jsonl(destination))
    write_json(
        destination.parent / "manifest.json",
        {
            "schema_version": 1,
            "documents": count,
            "format": "one canonical MUTCD chunk per JSONL row",
            "chunks": file_identity(destination),
        },
    )
    return count


def copy_graphrag_index(stage: Path) -> None:
    log("Copying the query-time GraphRAG index")
    destination = stage / "indexes" / "graphrag"
    for name in ("input", "output", "prompts"):
        copy_tree(GRAPHRAG_INDEX / name, destination / name)
    copy_file(GRAPHRAG_INDEX / "settings.yaml", destination / "settings.yaml")
    copy_file(
        GRAPHRAG_INDEX / ".gems_rag_graphrag_index.json",
        destination / ".graphrag_index.json",
    )


def copy_paperqa_index(stage: Path) -> None:
    log("Copying the query-time PaperQA index")
    destination = stage / "indexes" / "paperqa"
    copy_file(PAPERQA_INDEX / "docs.pkl", destination / "docs.pkl")
    copy_file(
        PAPERQA_INDEX / "docs.pkl.gems_rag_ready.json",
        destination / "docs.pkl.ready.json",
    )


def relative_media_path(value: str, *, default_subdir: str = "figures") -> str:
    path = Path(str(value))
    parts = {part.lower() for part in path.parts}
    subdir = (
        "page_images"
        if "page_images" in parts or path.name.lower().startswith("page_")
        else default_subdir
    )
    return f"{subdir}/{path.name}"


def sanitize_figures_jsonl(path: Path) -> int:
    rows = list(read_jsonl(path))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_path = str(row.get("image_path") or "")
            if image_path:
                row["image_path"] = relative_media_path(image_path)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return len(rows)


def sanitize_graph(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    for _node, data in graph.nodes(data=True):
        image_path = data.get("image_path")
        if image_path:
            data["image_path"] = relative_media_path(str(image_path))
        image_paths = data.get("image_paths")
        if image_paths:
            converted = [
                relative_media_path(str(value))
                for value in image_paths
                if str(value)
            ]
            data["image_paths"] = (
                tuple(converted)
                if isinstance(image_paths, tuple)
                else converted
            )
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
    }


def rebuild_qdrant(
    source_path: Path,
    destination_path: Path,
    *,
    include_collections: set[str] | None = None,
) -> dict[str, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required to rebuild the portable GEMS-RAG index; "
            "run this builder with the project virtual environment"
        ) from exc

    temporary_path = destination_path.with_name(
        f".{destination_path.name}.rebuilding"
    )
    shutil.rmtree(temporary_path, ignore_errors=True)
    shutil.rmtree(destination_path, ignore_errors=True)
    temporary_path.parent.mkdir(parents=True, exist_ok=True)

    source_client = QdrantClient(path=str(source_path))
    destination_client = QdrantClient(path=str(temporary_path))
    collection_counts: dict[str, int] = {}
    rewritten = 0
    try:
        for collection in source_client.get_collections().collections:
            name = collection.name
            if (
                include_collections is not None
                and name not in include_collections
            ):
                continue
            info = source_client.get_collection(name)
            params = info.config.params
            destination_client.create_collection(
                collection_name=name,
                vectors_config=params.vectors,
                sparse_vectors_config=params.sparse_vectors,
                shard_number=params.shard_number,
                sharding_method=params.sharding_method,
                replication_factor=params.replication_factor,
                write_consistency_factor=params.write_consistency_factor,
                on_disk_payload=params.on_disk_payload,
            )
            batch_size = 4 if name in {"mutcd_pages", "mutcd_figures_visual"} else 64
            offset = None
            copied = 0
            while True:
                points, next_offset = source_client.scroll(
                    collection_name=name,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                )
                upserts = []
                for point in points:
                    payload = dict(point.payload or {})
                    image_path = payload.get("image_path")
                    if image_path:
                        payload["image_path"] = relative_media_path(
                            str(image_path)
                        )
                        rewritten += 1
                    upserts.append(
                        models.PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload=payload,
                        )
                    )
                if upserts:
                    destination_client.upsert(
                        collection_name=name,
                        points=upserts,
                        wait=True,
                    )
                    copied += len(upserts)
                    if copied % 100 < batch_size:
                        log(f"  rebuilt {name}: {copied} points")
                if next_offset is None:
                    break
                offset = next_offset
            collection_counts[name] = copied
    finally:
        source_client.close()
        destination_client.close()
    if include_collections is not None:
        missing = sorted(include_collections - collection_counts.keys())
        if missing:
            raise ValueError(
                f"source Qdrant index is missing compact collections: {missing}"
            )
    (temporary_path / ".lock").unlink(missing_ok=True)

    verification = QdrantClient(path=str(temporary_path))
    try:
        for name, expected in collection_counts.items():
            actual = int(
                verification.count(collection_name=name, exact=True).count
            )
            if actual != expected:
                raise ValueError(
                    f"rebuilt Qdrant count mismatch for {name}: "
                    f"expected {expected}, found {actual}"
                )
    finally:
        verification.close()
    (temporary_path / ".lock").unlink(missing_ok=True)
    os.replace(temporary_path, destination_path)
    return {
        "collections": collection_counts,
        "media_payloads_rewritten": rewritten,
    }


def copy_gems_rag_index(stage: Path, source: Path) -> dict[str, Any]:
    log("Copying the compact query-time GEMS-RAG index")
    destination = stage / "indexes" / "gems-rag"
    pdfs = sorted(source.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF found under {source}")
    copy_file(pdfs[0], destination / "mutcd11theditionr1hl.pdf")

    cache_source = source / "mmrag_cache_v3"
    cache_destination = destination / "mmrag_cache_v3"
    for name in (
        "chunks.jsonl",
        "chunks_v4.stamp",
        "figure_coverage_report.json",
        "figures.jsonl",
        "graph.gpickle",
        "sign_codes.json",
    ):
        path = cache_source / name
        if path.is_file():
            copy_file(path, cache_destination / name)
    for noise in destination.rglob(".DS_Store"):
        noise.unlink()

    log("Rewriting GEMS-RAG media metadata to portable relative paths")
    figure_count = sanitize_figures_jsonl(
        cache_destination / "figures.jsonl"
    )
    graph = sanitize_graph(cache_destination / "graph.gpickle")
    qdrant = rebuild_qdrant(
        source / "qdrant_db",
        destination / "qdrant_db",
        include_collections=COMPACT_GEMS_RAG_COLLECTIONS,
    )
    return {
        "profile": "compact-text-graph",
        "default_query_mode": "no_visual",
        "source_pdf_included": True,
        "derived_media_included": False,
        "omitted_collections": [
            "mutcd_figures_visual",
            "mutcd_pages",
        ],
        "figures": figure_count,
        "graph": graph,
        "qdrant": qdrant,
    }


def replace_readme_placeholders(
    stage: Path,
    *,
    versions: dict[str, str],
    question_count: int,
    corpus_count: int,
) -> None:
    readme = stage / "README.md"
    text = readme.read_text(encoding="utf-8")
    release_bytes = sum(
        path.stat().st_size for path in stage.rglob("*") if path.is_file()
    )
    replacements = {
        "{{RELEASE_DATE}}": date.today().isoformat(),
        "{{QUESTION_COUNT}}": str(question_count),
        "{{CORPUS_COUNT}}": str(corpus_count),
        "{{GRAPHRAG_REVISION}}": versions["graphrag"],
        "{{PAPERQA_REVISION}}": versions["paperqa"],
        "{{RELEASE_SIZE}}": format_size(release_bytes),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    unresolved = sorted(set(re.findall(r"\{\{[^}]+\}\}", text)))
    if unresolved:
        raise ValueError(f"unresolved README placeholders: {unresolved}")
    readme.write_text(text, encoding="utf-8")


def scan_file(path: Path, patterns: dict[bytes, str]) -> list[str]:
    matches: list[str] = []
    maximum = max(len(pattern) for pattern in patterns)
    overlap = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            window = overlap + block
            for pattern, label in patterns.items():
                if pattern.lower() in window.lower():
                    matches.append(label)
            overlap = window[-maximum:]
    return sorted(set(matches))


def validate_anonymity(stage: Path) -> None:
    log("Scanning the assembled folder for identities, secrets, and stale names")
    problems: list[str] = []
    for path in sorted(stage.rglob("*")):
        relative = path.relative_to(stage).as_posix()
        if path.is_symlink():
            problems.append(f"symlink is not portable: {relative}")
            continue
        if path.is_dir():
            if path.name == ".git":
                problems.append(f"Git metadata included: {relative}")
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            problems.append(f"credential file included: {relative}")
        for label in scan_file(path, FORBIDDEN_BYTES):
            problems.append(f"{relative}: {label}")
    asset_root = stage / "indexes" / "gems-rag"
    stale_asset_path = b"/content/drive/MyDrive/MRAG"
    for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
        if scan_file(path, {stale_asset_path: "stale absolute media path"}):
            problems.append(
                f"{path.relative_to(stage).as_posix()}: stale absolute media path"
            )
    if problems:
        sample = "\n".join(f"- {problem}" for problem in problems[:100])
        raise ValueError(f"anonymous-release scan failed:\n{sample}")


def validate_public_api_language(stage: Path) -> None:
    log("Scanning release-authored surfaces for local execution references")
    roots = (
        stage / "README.md",
        stage / "RELEASE_MANIFEST.json",
        stage / ".env.example",
        stage / "THIRD_PARTY_NOTICES.md",
        stage / "configs",
        stage / "pipelines",
        stage / "scripts",
        stage / "src",
    )
    problems: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            for label in scan_file(path, PUBLIC_LOCAL_EXECUTION_BYTES):
                problems.append(f"{path.relative_to(stage).as_posix()}: {label}")
    if problems:
        sample = "\n".join(f"- {problem}" for problem in problems[:100])
        raise ValueError(f"public API-language scan failed:\n{sample}")


def compile_python(stage: Path) -> None:
    log("Compiling packaged Python sources")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            str(stage / "gems-rag" / "mrag"),
            str(stage / "pipelines"),
            str(stage / "scripts"),
            str(stage / "src"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    for cache in stage.rglob("__pycache__"):
        shutil.rmtree(cache)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)


def manifest_files(stage: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative in {"RELEASE_MANIFEST.json", "CHECKSUMS.sha256"}:
            continue
        identity = file_identity(path)
        files.append({"path": relative, **identity})
    return files


def write_manifest(
    stage: Path,
    *,
    versions: dict[str, str],
    question_count: int,
    unanswerable_count: int,
    corpus_count: int,
    gems_rag_index: dict[str, Any],
) -> None:
    log("Hashing release files and writing the integrity manifest")
    files = manifest_files(stage)
    manifest = {
        "schema_version": 1,
        "release_name": "mutcd-rag-anonymous-release",
        "release_date": date.today().isoformat(),
        "status": "complete",
        "method_ids": ["bm25", "graphrag", "paperqa", "gems-rag"],
        "default_comparison_methods": ["bm25", "graphrag", "paperqa"],
        "benchmark": {
            "id": "MUTCD-150-v1.0",
            "questions": question_count,
            "unanswerable_questions": unanswerable_count,
            "gold_sha256": sha256(stage / "benchmark" / "gold.jsonl"),
        },
        "corpus": {
            "canonical_chunks": corpus_count,
            "sha256": sha256(stage / "indexes" / "corpus" / "chunks.jsonl"),
        },
        "external_source_revisions": {
            "graphrag": versions["graphrag"],
            "paperqa": versions["paperqa"],
            "paperqa_version": versions["paperqa_version"],
        },
        "gems_rag_source": {
            "revision_disclosed": False,
            "parser_part_hierarchy_fix_applied": True,
            "license_file_present_in_source_snapshot": False,
        },
        "gems_rag_index": gems_rag_index,
        "packaging_adjustments": [
            "removed Git histories, remotes, notebooks, backups, runs, credentials, and runtime caches",
            "renamed public method and asset paths to gems-rag",
            "rewrote GEMS-RAG media payloads to portable relative paths",
            "retained the MUTCD PDF and compact GEMS-RAG text/graph index while omitting reproducible visual derivatives",
            "added a PaperQA setuptools-scm fallback matching the copied source version",
        ],
        "file_count": len(files),
        "total_file_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(stage / "RELEASE_MANIFEST.json", manifest)

    checksum_paths = [
        path
        for path in sorted(item for item in stage.rglob("*") if item.is_file())
        if path.name != "CHECKSUMS.sha256"
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(stage).as_posix()}"
        for path in checksum_paths
    ]
    (stage / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def validate_sources(args: argparse.Namespace) -> None:
    required = (
        TEMPLATE / "README.md",
        GEMS_RAG_SOURCE / "mrag" / "parsing.py",
        GRAPHRAG_SOURCE / "LICENSE",
        PAPERQA_SOURCE / "LICENSE",
        GRAPHRAG_INDEX / ".gems_rag_graphrag_index.json",
        PAPERQA_INDEX / "docs.pkl.gems_rag_ready.json",
        CORPUS,
        GOLD,
        QUESTIONS,
        GRADER_SPEC,
        args.gems_rag_assets / "qdrant_db" / "meta.json",
        args.gems_rag_assets / "mmrag_cache_v3" / "graph.gpickle",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing release inputs: {missing}")


def main() -> int:
    args = parse_args()
    validate_sources(args)
    output = args.output.expanduser().absolute()
    stage = output.parent / f".{output.name}.building"
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if stage.exists() and not args.force:
        raise FileExistsError(
            f"{stage} is an abandoned staging folder; pass --force to replace it"
        )
    if args.force:
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    try:
        copy_template(stage)
        versions = copy_third_party(stage)
        copy_gems_rag_source(stage)
        copy_pipelines_and_support(stage)
        question_count, unanswerable_count = copy_benchmark(stage)
        corpus_count = copy_corpus(stage)
        copy_graphrag_index(stage)
        copy_paperqa_index(stage)
        gems_rag_index = copy_gems_rag_index(
            stage,
            args.gems_rag_assets,
        )
        replace_readme_placeholders(
            stage,
            versions=versions,
            question_count=question_count,
            corpus_count=corpus_count,
        )
        compile_python(stage)
        validate_anonymity(stage)
        validate_public_api_language(stage)
        write_manifest(
            stage,
            versions=versions,
            question_count=question_count,
            unanswerable_count=unanswerable_count,
            corpus_count=corpus_count,
            gems_rag_index=gems_rag_index,
        )
        validate_anonymity(stage)
        validate_public_api_language(stage)
        os.replace(stage, output)
    except Exception:
        log(f"Build failed; staging folder retained for inspection: {stage}")
        raise

    log(f"Anonymous release ready: {output}")
    log(f"Size: {format_size(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
