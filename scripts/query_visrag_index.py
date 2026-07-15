#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_chunks
from gems_rag.visrag_server import (
    VisragServerError,
    VisragServerState,
    request_visrag_socket,
    serve_visrag_socket,
)

DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "visrag"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"
DEFAULT_WORKING_DIR = ROOT / "data" / "working" / "visrag_index"
DEFAULT_MANIFEST = DEFAULT_WORKING_DIR / "visual_manifest.jsonl"
DEFAULT_EMBEDDINGS = DEFAULT_WORKING_DIR / "embeddings.npy"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "visrag" / "bin" / "python"
DEFAULT_SERVER_DIR = ROOT / "data" / "working" / "visrag_server"
DEFAULT_MODEL = "openbmb/VisRAG-Ret"
DEFAULT_MODEL_REVISION = "95ef596df871b606167cb7e4b7215caf1bfdf761"
INDEX_SCHEMA_VERSION = 1
SERVER_SCHEMA_VERSION = 1
INSTRUCTION = "Represent this query for retrieving relevant documents: "
REQUIRED_MODULES = ["torch", "transformers", "PIL", "numpy"]
EVIDENCE_KINDS = {"page", "figure"}


def main() -> int:
    args = _parse_args()
    if args.command in {"check", "index", "query", "serve"}:
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
    if args.command == "serve":
        return _serve(args)
    if args.command == "stop":
        return _stop_server(args)
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
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Pinned Hugging Face revision used for both image and query embeddings.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or any torch device string.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model weights from Hugging Face.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER_DIR)
    parser.add_argument("--server-startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-query-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-idle-timeout-s", type=float, default=1800.0)
    parser.add_argument("--server-max-cache-entries", type=int, default=2048)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Report whether the VisRAG adapter has dependencies, manifest, and embeddings.")

    prepare = sub.add_parser("prepare", help="Build a local visual manifest from extracted MRAG images.")
    prepare.add_argument("--scope", choices=["pages", "figures", "both"], default="pages")
    prepare.add_argument("--limit", type=int)

    index = sub.add_parser("index", help="Encode manifest images with VisRAG-Ret and save embeddings.")
    index.add_argument("--batch-size", type=int, default=4)
    index.add_argument("--limit", type=int, help="Encode only the first N manifest rows.")
    index.add_argument("--force", action="store_true", help="Discard an incompatible partial index and rebuild it.")

    query = sub.add_parser("query", help="Query the saved VisRAG-Ret embedding index.")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=6)
    query.add_argument("--json", action=argparse.BooleanOptionalAction, default=True)
    query.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an auto-started local worker instead of reloading VisRAG-Ret for every question.",
    )
    sub.add_parser("serve", help="Run the local persistent VisRAG retrieval worker.")
    sub.add_parser("stop", help="Stop this workspace's persistent VisRAG retrieval worker.")
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
    embedding_info = _embedding_info(args.embeddings) if args.embeddings.exists() and not import_errors.get("numpy") else None
    embedding_rows = embedding_info["shape"][0] if embedding_info else None
    repo_found = args.repo.exists()
    source_found = (args.repo / "src").exists()
    readiness = _index_readiness(args, manifest_rows, embedding_info)
    index_ready = readiness["ready"]
    server_health = _server_health(args)
    expected_server_fingerprint = _server_fingerprint(args) if not import_errors else None
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
        "embedding_shape": embedding_info["shape"] if embedding_info else None,
        "embedding_dtype": embedding_info["dtype"] if embedding_info else None,
        "ready_marker": str(readiness["ready_marker"]),
        "ready_marker_found": readiness["ready_marker_found"],
        "progress_marker": str(readiness["progress_marker"]),
        "progress_marker_found": readiness["progress_marker_found"],
        "completed_rows": readiness["completed_rows"],
        "index_state_reasons": readiness["reasons"],
        "index_ready": index_ready,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": getattr(args, "model_revision", DEFAULT_MODEL_REVISION),
        "persistent_server_ready": _server_matches(server_health, args, expected_server_fingerprint),
        "persistent_server": server_health,
        "missing_or_failed_imports": import_errors,
        "notes": "VisRAG-Ret indexing follows the upstream AutoModel/AutoTokenizer weighted-mean-pooling recipe. Every completed batch is checkpointed, and query readiness requires a checksum-matched ready marker rather than only a row-count match.",
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
    signature = _index_signature(args, len(records))
    artifacts = _index_artifact_paths(args.embeddings)
    if args.force:
        for path in (artifacts["partial"], artifacts["progress"], artifacts["ready"]):
            path.unlink(missing_ok=True)
    ready = _read_json_object(artifacts["ready"])
    embedding_info = _embedding_info(args.embeddings) if args.embeddings.exists() else None
    if _complete_index_matches(ready, signature, embedding_info):
        print(
            json.dumps(
                {
                    "indexed": True,
                    "already_ready": True,
                    "records": len(records),
                    "embeddings": str(args.embeddings),
                    "ready_marker": str(artifacts["ready"]),
                },
                indent=2,
            )
        )
        return 0
    progress = _read_json_object(artifacts["progress"])
    try:
        recovered = _recover_completed_index(
            embeddings=args.embeddings,
            artifacts=artifacts,
            signature=signature,
            progress=progress,
        )
        if recovered is not None:
            print(json.dumps(recovered, indent=2))
            return 0
        _validate_partial_state(artifacts, signature, progress)
        model, tokenizer, torch, np = _load_model(args)

        def encode_batch(batch: list[dict[str, Any]]) -> Any:
            images = [_open_image(record["image_path"]) for record in batch]
            return _encode(model, tokenizer, torch, np, images)

        result = _write_resumable_index(
            records=records,
            signature=signature,
            embeddings=args.embeddings,
            artifacts=artifacts,
            batch_size=max(args.batch_size, 1),
            encode_batch=encode_batch,
            np=np,
        )
    except Exception as exc:
        print(json.dumps({"error": "visrag_index_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def _query(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if report["missing_or_failed_imports"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not report["index_ready"]:
        print(json.dumps({"error": "index_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    if args.persistent:
        try:
            payload = _query_persistent(args)
        except Exception as exc:
            print(json.dumps({"error": "visrag_server_query_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
            return 2
        _print_query_payload(payload, as_json=args.json)
        return 0
    records = _read_jsonl(args.manifest)
    try:
        model, tokenizer, torch, np = _load_model(args)
        embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
        payload = _query_loaded(
            question=args.question,
            top_k=args.top_k,
            records=records,
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            np=np,
            model_name_or_path=args.model_name_or_path,
            model_revision=args.model_revision,
        )
    except Exception as exc:
        print(json.dumps({"error": "visrag_query_failed", "detail": repr(exc)}, indent=2), file=sys.stderr)
        return 2
    payload["debug"] = {"persistent_server": False, "persistent_cache_hit": False}
    _print_query_payload(payload, as_json=args.json)
    return 0


def _query_loaded(
    *,
    question: str,
    top_k: int,
    records: list[dict[str, Any]],
    embeddings: Any,
    model: Any,
    tokenizer: Any,
    torch: Any,
    np: Any,
    model_name_or_path: str,
    model_revision: str,
) -> dict[str, Any]:
    query_embedding = _encode(model, tokenizer, torch, np, [INSTRUCTION + question])[0]
    scores = np.asarray(embeddings @ query_embedding)
    if scores.ndim != 1 or scores.shape[0] != len(records) or not np.isfinite(scores).all():
        raise RuntimeError(f"VisRAG returned invalid query scores with shape {scores.shape!r}.")
    order = np.argsort(-scores)[: max(top_k, 1)]
    contexts = [_context_from_record(records[int(idx)], float(scores[int(idx)])) for idx in order]
    return {
        "question": question,
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "contexts": contexts,
    }


def _print_query_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    for context in payload.get("contexts", []):
        print(f"{context['score']:.4f}\t{context['name']}\t{context['image_path']}")


def _query_persistent(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_server(args)
    response = request_visrag_socket(
        _server_socket(args),
        {"action": "query", "question": args.question, "top_k": max(args.top_k, 1)},
        timeout_s=args.server_query_timeout_s,
    )
    if not response.get("ok") or not isinstance(response.get("result"), dict):
        raise VisragServerError(str(response.get("detail") or response.get("error") or response))
    payload = dict(response["result"])
    payload["debug"] = {
        "persistent_server": True,
        "persistent_cache_hit": bool(response.get("cache_hit")),
    }
    return payload


def _serve(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "index_not_ready", **report}, indent=2), file=sys.stderr)
        return 2
    existing = report.get("persistent_server")
    if isinstance(existing, dict) and existing.get("status") == "ready":
        if report["persistent_server_ready"]:
            print(json.dumps({"serving": True, "already_running": True, "pid": existing.get("pid")}))
            return 0
        print(json.dumps({"error": "different_visrag_server_already_running", "server": existing}), file=sys.stderr)
        return 2
    args.server_dir.mkdir(parents=True, exist_ok=True)
    _write_pid(_server_pid(args), os.getpid())
    try:
        records = _read_jsonl(args.manifest)
        model, tokenizer, torch, np = _load_model(args)
        embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)

        def query_func(question: str, top_k: int) -> dict[str, Any]:
            return _query_loaded(
                question=question,
                top_k=top_k,
                records=records,
                embeddings=embeddings,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                np=np,
                model_name_or_path=args.model_name_or_path,
                model_revision=args.model_revision,
            )

        state = VisragServerState(
            query_func=query_func,
            fingerprint=_server_fingerprint(args),
            manifest=args.manifest,
            embeddings=args.embeddings,
            model_name_or_path=args.model_name_or_path,
            model_revision=args.model_revision,
            max_cache_entries=args.server_max_cache_entries,
        )
        serve_visrag_socket(
            _server_socket(args),
            state,
            idle_timeout_s=args.server_idle_timeout_s,
        )
    except Exception as exc:
        print(json.dumps({"error": "visrag_server_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2
    finally:
        _remove_owned_pid(_server_pid(args), os.getpid())
        _server_socket(args).unlink(missing_ok=True)
    return 0


def _stop_server(args: argparse.Namespace) -> int:
    health = _server_health(args)
    if health is None:
        _server_socket(args).unlink(missing_ok=True)
        _server_pid(args).unlink(missing_ok=True)
        print(json.dumps({"stopped": True, "server_dir": str(args.server_dir), "worker_was_running": False}))
        return 0
    confirmed_pid = int(health["pid"]) if health.get("pid") is not None else None
    try:
        request_visrag_socket(
            _server_socket(args),
            {"action": "stop"},
            timeout_s=min(5.0, args.server_query_timeout_s),
        )
    except VisragServerError:
        pass
    try:
        _wait_for_server_exit(args, timeout_s=10.0)
    except VisragServerError:
        if confirmed_pid and _pid_alive(confirmed_pid):
            os.kill(confirmed_pid, signal.SIGTERM)
            _wait_for_server_exit(args, timeout_s=10.0)
    _server_socket(args).unlink(missing_ok=True)
    _server_pid(args).unlink(missing_ok=True)
    print(json.dumps({"stopped": True, "server_dir": str(args.server_dir)}))
    return 0


def _ensure_server(args: argparse.Namespace) -> dict[str, Any]:
    expected = _server_fingerprint(args)
    health = _server_health(args)
    if _server_matches(health, args, expected):
        return health
    args.server_dir.mkdir(parents=True, exist_ok=True)
    with _server_start_lock(args.server_dir / "start.lock"):
        health = _server_health(args)
        if _server_matches(health, args, expected):
            return health
        if health is not None:
            request_visrag_socket(
                _server_socket(args),
                {"action": "stop"},
                timeout_s=min(5.0, args.server_query_timeout_s),
            )
            _wait_for_server_exit(args, timeout_s=15.0)
        existing_pid = _read_pid(_server_pid(args))
        if existing_pid and _pid_alive(existing_pid):
            return _wait_for_server(args, expected, process=None)
        _server_pid(args).unlink(missing_ok=True)
        _server_socket(args).unlink(missing_ok=True)
        log_path = args.server_dir / "server.log"
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                _server_command(args),
                cwd=ROOT,
                env={**os.environ, "HF_HUB_DISABLE_XET": os.getenv("HF_HUB_DISABLE_XET", "1")},
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return _wait_for_server(args, expected, process=process)


def _wait_for_server(
    args: argparse.Namespace,
    expected_fingerprint: str,
    *,
    process: subprocess.Popen | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.server_startup_timeout_s
    while time.monotonic() < deadline:
        health = _server_health(args)
        if _server_matches(health, args, expected_fingerprint):
            return health
        if health is not None and health.get("status") == "ready":
            raise VisragServerError("VisRAG server started with a different index or runtime fingerprint")
        if process is not None and process.poll() is not None:
            raise VisragServerError(
                f"VisRAG server exited with code {process.returncode}: {_server_log_tail(args)}"
            )
        time.sleep(0.25)
    raise VisragServerError(f"VisRAG server did not become ready within {args.server_startup_timeout_s:g}s")


def _wait_for_server_exit(args: argparse.Namespace, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pid = _read_pid(_server_pid(args))
        if _server_health(args) is None and not (pid and _pid_alive(pid)):
            return
        time.sleep(0.1)
    raise VisragServerError("VisRAG server did not stop")


def _server_health(args: argparse.Namespace) -> dict[str, Any] | None:
    socket_path = _server_socket(args)
    if not socket_path.exists():
        return None
    try:
        response = request_visrag_socket(socket_path, {"action": "health"}, timeout_s=1.0)
    except VisragServerError:
        return None
    return response if response.get("ok") else None


def _server_matches(
    health: dict[str, Any] | None,
    args: argparse.Namespace,
    expected_fingerprint: str | None,
) -> bool:
    return bool(
        health
        and health.get("status") == "ready"
        and health.get("fingerprint") == expected_fingerprint
        and health.get("manifest") == str(args.manifest.resolve())
        and health.get("embeddings") == str(args.embeddings.resolve())
    )


def _server_fingerprint(args: argparse.Namespace) -> str:
    artifacts = _index_artifact_paths(args.embeddings)
    effective_device, effective_dtype = _effective_runtime(args)
    payload = {
        "schema_version": SERVER_SCHEMA_VERSION,
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(getattr(args, "model_revision", DEFAULT_MODEL_REVISION)),
        "device": str(getattr(args, "device", "auto")),
        "dtype": str(getattr(args, "dtype", "auto")),
        "effective_device": effective_device,
        "effective_dtype": effective_dtype,
        "trust_remote_code": bool(getattr(args, "trust_remote_code", True)),
        "local_files_only": bool(getattr(args, "local_files_only", False)),
        "manifest": str(args.manifest.resolve()),
        "embeddings": str(args.embeddings.resolve()),
        "ready_sha256": _sha256_file(artifacts["ready"]) if artifacts["ready"].exists() else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for path in (Path(__file__).resolve(), ROOT / "src" / "gems_rag" / "visrag_server.py"):
        if path.exists():
            digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _server_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo",
        str(args.repo),
        "--mrag-dir",
        str(args.mrag_dir),
        "--working-dir",
        str(args.working_dir),
        "--manifest",
        str(args.manifest),
        "--embeddings",
        str(args.embeddings),
        "--python",
        sys.executable,
        "--model-name-or-path",
        str(args.model_name_or_path),
        "--model-revision",
        str(args.model_revision),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--server-dir",
        str(args.server_dir),
        "--server-startup-timeout-s",
        str(args.server_startup_timeout_s),
        "--server-query-timeout-s",
        str(args.server_query_timeout_s),
        "--server-idle-timeout-s",
        str(args.server_idle_timeout_s),
        "--server-max-cache-entries",
        str(args.server_max_cache_entries),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    command.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
    command.append("serve")
    return command


def _server_socket(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "visrag.sock"


def _server_pid(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "server.pid"


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="ascii")


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _remove_owned_pid(path: Path, pid: int) -> None:
    if _read_pid(path) == pid:
        path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _server_log_tail(args: argparse.Namespace, limit: int = 4000) -> str:
    path = Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "server.log"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "server log unavailable"


@contextmanager
def _server_start_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo / "src"))
    sys.path.insert(0, str(args.repo / "timm_modified"))
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    device = _device(torch, args.device)
    dtype = _torch_dtype(torch, args.dtype, device)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(args.model_name_or_path, **model_kwargs)
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


def _index_signature(args: argparse.Namespace, record_count: int) -> dict[str, Any]:
    effective_device, effective_dtype = _effective_runtime(args)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "manifest_sha256": _sha256_file(args.manifest) if args.manifest.exists() else None,
        "record_count": record_count,
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(getattr(args, "model_revision", DEFAULT_MODEL_REVISION)),
        "trust_remote_code": bool(getattr(args, "trust_remote_code", True)),
        "device": str(getattr(args, "device", "auto")),
        "dtype": str(getattr(args, "dtype", "auto")),
        "effective_device": effective_device,
        "effective_dtype": effective_dtype,
    }


def _index_artifact_paths(embeddings: Path) -> dict[str, Path]:
    base = embeddings.with_suffix("") if embeddings.suffix == ".npy" else embeddings
    return {
        "partial": Path(f"{base}.partial.npy"),
        "progress": Path(f"{base}.progress.json"),
        "ready": Path(f"{base}.ready.json"),
    }


def _semantic_signature(signature: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "manifest_sha256",
        "record_count",
        "model_name_or_path",
        "model_revision",
        "trust_remote_code",
        "effective_device",
        "effective_dtype",
    )
    return {key: signature.get(key) for key in keys}


def _complete_index_matches(
    ready: dict[str, Any] | None,
    signature: dict[str, Any],
    embedding_info: dict[str, Any] | None,
) -> bool:
    if not ready or not embedding_info or ready.get("status") != "ready":
        return False
    if _semantic_signature(dict(ready.get("signature") or {})) != _semantic_signature(signature):
        return False
    shape = ready.get("shape")
    return bool(
        shape == embedding_info["shape"]
        and len(shape) == 2
        and shape[0] == signature["record_count"]
        and ready.get("completed_rows") == signature["record_count"]
        and ready.get("embedding_dtype") == embedding_info["dtype"]
    )


def _index_readiness(
    args: argparse.Namespace,
    manifest_rows: int,
    embedding_info: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts = _index_artifact_paths(args.embeddings)
    ready = _read_json_object(artifacts["ready"])
    progress = _read_json_object(artifacts["progress"])
    signature = _index_signature(args, manifest_rows)
    reasons: list[str] = []
    if manifest_rows <= 0:
        reasons.append("manifest_empty_or_missing")
    if not args.embeddings.exists():
        reasons.append("embeddings_missing")
    elif embedding_info is None:
        reasons.append("embeddings_unreadable")
    if ready is None:
        reasons.append("ready_marker_missing_or_invalid")
    elif ready.get("status") != "ready":
        reasons.append("ready_marker_status_invalid")
    else:
        ready_signature = dict(ready.get("signature") or {})
        for key, expected in _semantic_signature(signature).items():
            if ready_signature.get(key) != expected:
                reasons.append(f"ready_{key}_mismatch")
        if embedding_info is not None:
            if ready.get("shape") != embedding_info["shape"]:
                reasons.append("ready_embedding_shape_mismatch")
            if ready.get("embedding_dtype") != embedding_info["dtype"]:
                reasons.append("ready_embedding_dtype_mismatch")
        if ready.get("completed_rows") != manifest_rows:
            reasons.append("ready_completed_rows_mismatch")
    completed_rows = 0
    if progress is not None:
        completed_rows = int(progress.get("completed_rows") or 0)
    elif ready is not None:
        completed_rows = int(ready.get("completed_rows") or 0)
    return {
        "ready": not reasons and _complete_index_matches(ready, signature, embedding_info),
        "reasons": reasons,
        "ready_marker": artifacts["ready"],
        "ready_marker_found": artifacts["ready"].exists(),
        "progress_marker": artifacts["progress"],
        "progress_marker_found": artifacts["progress"].exists(),
        "completed_rows": completed_rows,
    }


def _validate_partial_state(
    artifacts: dict[str, Path],
    signature: dict[str, Any],
    progress: dict[str, Any] | None,
) -> None:
    partial_exists = artifacts["partial"].exists()
    progress_exists = artifacts["progress"].exists()
    if not partial_exists and not progress_exists:
        return
    if not partial_exists or progress is None:
        raise RuntimeError("VisRAG partial index state is incomplete; rerun index with --force to rebuild it.")
    if progress.get("signature") != signature:
        raise RuntimeError(
            "VisRAG partial index was created with a different manifest, model, device, or dtype; "
            "resume with the original options or rerun index with --force."
        )
    info = _embedding_info(artifacts["partial"])
    if info is None or info["shape"] != progress.get("shape") or info["dtype"] != progress.get("embedding_dtype"):
        raise RuntimeError("VisRAG partial embedding matrix does not match its progress marker; use --force.")
    completed = int(progress.get("completed_rows") or 0)
    if completed < 0 or completed > signature["record_count"]:
        raise RuntimeError("VisRAG progress marker has an invalid completed row count; use --force.")


def _recover_completed_index(
    *,
    embeddings: Path,
    artifacts: dict[str, Path],
    signature: dict[str, Any],
    progress: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if progress is None or progress.get("signature") != signature:
        return None
    if int(progress.get("completed_rows") or 0) != signature["record_count"]:
        return None
    candidate = artifacts["partial"] if artifacts["partial"].exists() else embeddings
    info = _embedding_info(candidate) if candidate.exists() else None
    if info is None or info["shape"] != progress.get("shape") or info["dtype"] != progress.get("embedding_dtype"):
        return None
    if candidate == artifacts["partial"]:
        embeddings.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, embeddings)
    ready = _ready_payload(signature, info, embeddings)
    _write_json_atomic(artifacts["ready"], ready)
    artifacts["progress"].unlink(missing_ok=True)
    return {
        "indexed": True,
        "recovered_completed_index": True,
        "records": signature["record_count"],
        "embeddings": str(embeddings),
        "ready_marker": str(artifacts["ready"]),
    }


def _write_resumable_index(
    *,
    records: list[dict[str, Any]],
    signature: dict[str, Any],
    embeddings: Path,
    artifacts: dict[str, Path],
    batch_size: int,
    encode_batch: Any,
    np: Any,
) -> dict[str, Any]:
    embeddings.parent.mkdir(parents=True, exist_ok=True)
    progress = _read_json_object(artifacts["progress"])
    _validate_partial_state(artifacts, signature, progress)
    completed = int(progress.get("completed_rows") or 0) if progress else 0
    matrix = np.load(artifacts["partial"], mmap_mode="r+") if progress else None
    resumed_from = completed

    for start in range(completed, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        values = np.asarray(encode_batch(records[start:stop]), dtype=np.float32)
        expected_dim = int(matrix.shape[1]) if matrix is not None else None
        _validate_embedding_batch(values, stop - start, expected_dim, np)
        if matrix is None:
            shape = (len(records), int(values.shape[1]))
            matrix = np.lib.format.open_memmap(
                artifacts["partial"],
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            progress = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "status": "indexing",
                "signature": signature,
                "shape": list(shape),
                "embedding_dtype": str(matrix.dtype),
                "completed_rows": 0,
                "embeddings": str(embeddings),
                "partial_embeddings": str(artifacts["partial"]),
            }
            _write_json_atomic(artifacts["progress"], progress)
        matrix[start:stop] = values
        _flush_memmap(matrix, artifacts["partial"])
        progress["completed_rows"] = stop
        _write_json_atomic(artifacts["progress"], progress)
        print(
            json.dumps({"event": "visrag_index_progress", "completed_rows": stop, "total_rows": len(records)}),
            file=sys.stderr,
            flush=True,
        )

    if matrix is None or progress is None:
        raise RuntimeError("VisRAG indexing produced no embedding matrix.")
    info = {"shape": list(matrix.shape), "dtype": str(matrix.dtype)}
    _flush_memmap(matrix, artifacts["partial"])
    del matrix
    os.replace(artifacts["partial"], embeddings)
    ready = _ready_payload(signature, info, embeddings)
    _write_json_atomic(artifacts["ready"], ready)
    artifacts["progress"].unlink(missing_ok=True)
    return {
        "indexed": True,
        "records": len(records),
        "resumed_from": resumed_from,
        "embeddings": str(embeddings),
        "embedding_shape": info["shape"],
        "ready_marker": str(artifacts["ready"]),
    }


def _validate_embedding_batch(values: Any, expected_rows: int, expected_dim: int | None, np: Any) -> None:
    if values.ndim != 2 or values.shape[0] != expected_rows or not values.shape[1]:
        raise RuntimeError(f"VisRAG returned an invalid embedding batch shape: {values.shape!r}")
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise RuntimeError(f"VisRAG embedding width changed from {expected_dim} to {values.shape[1]}.")
    if not np.isfinite(values).all():
        raise RuntimeError("VisRAG returned non-finite embeddings.")


def _ready_payload(signature: dict[str, Any], info: dict[str, Any], embeddings: Path) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": "ready",
        "signature": signature,
        "shape": info["shape"],
        "embedding_dtype": info["dtype"],
        "completed_rows": signature["record_count"],
        "embeddings": str(embeddings),
    }


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


def _torch_dtype(torch: Any, requested: str, device: str) -> Any:
    effective = _effective_dtype_name(torch, requested, device)
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[effective]


def _effective_runtime(args: argparse.Namespace) -> tuple[str, str]:
    requested_device = str(getattr(args, "device", "auto"))
    requested_dtype = str(getattr(args, "dtype", "auto"))
    torch = None
    if requested_device == "auto" or (requested_dtype == "auto" and requested_device.startswith("cuda")):
        try:
            import torch as torch_module
        except ModuleNotFoundError:
            return requested_device, requested_dtype

        torch = torch_module
    device = _device(torch, requested_device) if requested_device == "auto" else requested_device
    dtype = requested_dtype if requested_dtype != "auto" else _effective_dtype_name(torch, requested_dtype, device)
    return device, dtype


def _effective_dtype_name(torch: Any, requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if str(device).startswith("cuda"):
        supports_bfloat16 = getattr(getattr(torch, "cuda", None), "is_bf16_supported", lambda: False)
        return "bfloat16" if supports_bfloat16() else "float16"
    if str(device).startswith("mps"):
        return "float16"
    return "float32"


def _open_image(path: str):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _flush_memmap(matrix: Any, path: Path) -> None:
    matrix.flush()
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _embedding_info(path: Path) -> dict[str, Any] | None:
    try:
        import numpy as np

        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2:
            return None
        return {"shape": list(matrix.shape), "dtype": str(matrix.dtype)}
    except Exception:
        return None


def _embedding_rows(path: Path) -> int | None:
    info = _embedding_info(path)
    return int(info["shape"][0]) if info else None


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
