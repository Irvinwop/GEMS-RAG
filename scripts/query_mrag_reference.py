#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_chunks
from gems_rag.mrag_reference_modes import REFERENCE_MODES, retrieve_reference_mode
from gems_rag.mrag_reference_server import (
    CachedReranker,
    MragReferenceServerError,
    MragReferenceServerState,
    request_reference_socket,
    serve_reference_socket,
)

DEFAULT_REPO = ROOT / "external" / "MRAG_stp2"
DEFAULT_MRAG_DIR = ROOT / "data" / "extracted" / "MRAG-20260715T174043Z-1" / "MRAG"
DEFAULT_ENV_PYTHON = ROOT / "data" / "working" / "venvs" / "mrag-reference" / "bin" / "python"
DEFAULT_SERVER_DIR = ROOT / "data" / "working" / "mrag-reference-server"
SERVER_SCHEMA_VERSION = 1
REQUIRED_MODULES = ["qdrant_client", "numpy", "torch"]
GRAPH_MODULES = ["networkx"]
TEXT_RETRIEVAL_MODULES = ["FlagEmbedding", "sentence_transformers"]
RERANK_MODULES = ["mxbai_rerank", "sentence_transformers"]
VISUAL_MODULES = ["colpali_engine"]
RERANK_MODES = {"full", "no_graph", "no_visual", "no_rule", "no_hierarchy"}
VISUAL_MODES = {"multimodal", "full", "no_graph", "no_rule", "no_hierarchy"}
GRAPH_MODES = {"full", "no_visual", "no_rule", "no_hierarchy"}


def main() -> int:
    args = _parse_args()
    reexec_code = _maybe_reexec(args.python)
    if reexec_code is not None:
        return reexec_code
    if args.command == "check":
        report = _dependency_report(args)
        print(json.dumps(report, indent=2))
        return 0 if report["runnable"] else 2
    if args.command == "retrieve":
        return _retrieve(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "stop":
        return _stop_server(args)
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the cloned hannanazad/MRAG_stp2 retrieval stack.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.getenv("MRAG_REFERENCE_PYTHON", str(DEFAULT_ENV_PYTHON))),
        help="Optional isolated Python with MRAG reference dependencies. Defaults to data/working/venvs/mrag-reference/bin/python when present.",
    )
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER_DIR)
    parser.add_argument("--server-startup-timeout-s", type=float, default=240.0)
    parser.add_argument("--server-query-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-idle-timeout-s", type=float, default=1800.0)
    parser.add_argument("--server-max-cache-entries", type=int, default=2048)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Report whether the local environment can run an MRAG retrieval mode.")
    check.add_argument("--mode", choices=REFERENCE_MODES, default="full")

    retrieve = sub.add_parser("retrieve", help="Run MRAG retrieval and print JSON evidence.")
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--top-k", type=int, default=6)
    retrieve.add_argument("--mode", choices=REFERENCE_MODES, default="full")
    retrieve.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an auto-started local MRAG worker instead of reloading model weights per query.",
    )
    retrieve.add_argument("--with-image", action="store_true", help="Deprecated compatibility flag; visual modes load the encoder automatically.")
    sub.add_parser("serve", help="Run the local persistent MRAG retrieval worker.")
    sub.add_parser("stop", help="Stop this workspace's persistent MRAG retrieval worker.")
    return parser.parse_args()


def _dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    mode = getattr(args, "mode", "full")
    missing_required = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    text_ok = any(importlib.util.find_spec(name) is not None for name in TEXT_RETRIEVAL_MODULES)
    rerank_ok = any(importlib.util.find_spec(name) is not None for name in RERANK_MODULES)
    visual_ok = all(importlib.util.find_spec(name) is not None for name in VISUAL_MODULES)
    graph_ok = all(importlib.util.find_spec(name) is not None for name in GRAPH_MODULES)
    missing_groups = []
    if not text_ok:
        missing_groups.append({"group": "text_embedding", "one_of": TEXT_RETRIEVAL_MODULES})
    if mode in RERANK_MODES and not rerank_ok:
        missing_groups.append({"group": "reranking", "one_of": RERANK_MODULES})
    if mode in VISUAL_MODES and not visual_ok:
        missing_groups.append({"group": "visual_embedding", "all_of": VISUAL_MODULES})
    if mode in GRAPH_MODES and not graph_ok:
        missing_groups.append({"group": "graph", "all_of": GRAPH_MODULES})
    return {
        "runnable": (
            not missing_required
            and text_ok
            and (mode not in RERANK_MODES or rerank_ok)
            and (mode not in VISUAL_MODES or visual_ok)
            and (mode not in GRAPH_MODES or graph_ok)
        ),
        "mode": mode,
        "components": {
            "reranker_required": mode in RERANK_MODES,
            "visual_required": mode in VISUAL_MODES,
            "graph_required": mode in GRAPH_MODES,
        },
        "adapter_python": str(args.python),
        "adapter_python_found": args.python.exists(),
        "current_python": sys.executable,
        "missing_required_modules": missing_required,
        "missing_alternative_groups": missing_groups,
        "notes": "Install external/MRAG_stp2/requirements.txt into an isolated environment for the selected reference mode.",
    }


def _retrieve(args: argparse.Namespace) -> int:
    report = _dependency_report(args)
    if not report["runnable"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    if not args.repo.exists():
        print(json.dumps({"error": "repo_not_found", "repo": str(args.repo)}), file=sys.stderr)
        return 2
    if not args.mrag_dir.exists():
        print(json.dumps({"error": "mrag_dir_not_found", "mrag_dir": str(args.mrag_dir)}), file=sys.stderr)
        return 2

    if args.persistent and _dependency_report(SimpleNamespace(**{**vars(args), "mode": "full"}))["runnable"]:
        try:
            result = _retrieve_persistent(args)
        except Exception as exc:
            print(json.dumps({"error": "persistent_retrieve_failed", "detail": repr(exc)}), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0

    return _retrieve_direct(args)


def _retrieve_direct(args: argparse.Namespace) -> int:
    try:
        config = _load_upstream_config(args)
    except Exception as exc:
        print(json.dumps({"error": "import_failed", "detail": repr(exc)}), file=sys.stderr)
        return 2

    try:
        pipeline = _init_mode_pipeline(config, args.mode)
        result = retrieve_reference_mode(
            pipeline,
            args.question,
            mode=args.mode,
            top_k=args.top_k,
            chunks=load_chunks(args.mrag_dir),
        )
    except Exception as exc:
        print(json.dumps({"error": "retrieve_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "question": args.question,
                "chunks": result["chunks"],
                "figures": result["figures"],
                "pages": result["pages"],
                "debug": result["debug"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _retrieve_persistent(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_server(args)
    response = request_reference_socket(
        _server_socket(args),
        {
            "action": "retrieve",
            "mode": args.mode,
            "question": args.question,
            "top_k": args.top_k,
        },
        timeout_s=args.server_query_timeout_s,
    )
    if not response.get("ok") or not isinstance(response.get("result"), dict):
        raise MragReferenceServerError(str(response.get("detail") or response.get("error") or response))
    result = dict(response["result"])
    debug = dict(result.get("debug") or {})
    debug["persistent_server"] = True
    debug["persistent_cache_hit"] = bool(response.get("cache_hit"))
    result["debug"] = debug
    return result


def _serve(args: argparse.Namespace) -> int:
    report = _dependency_report(SimpleNamespace(**{**vars(args), "mode": "full"}))
    if not report["runnable"]:
        print(json.dumps({"error": "missing_dependencies", **report}, indent=2), file=sys.stderr)
        return 2
    args.server_dir.mkdir(parents=True, exist_ok=True)
    _write_pid(_server_pid(args), os.getpid())
    try:
        config = _load_upstream_config(args)
        pipeline = _init_mode_pipeline(config, "full")
        if pipeline.rerank is not None:
            pipeline.rerank = CachedReranker(pipeline.rerank)
        state = MragReferenceServerState(
            pipeline=pipeline,
            chunks=load_chunks(args.mrag_dir),
            mrag_dir=args.mrag_dir,
            fingerprint=_server_fingerprint(args),
            max_cache_entries=args.server_max_cache_entries,
        )
        serve_reference_socket(
            _server_socket(args),
            state,
            idle_timeout_s=args.server_idle_timeout_s,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"error": "server_failed", "detail": repr(exc)}), file=sys.stderr)
        return 1
    finally:
        _remove_owned_pid(_server_pid(args), os.getpid())
        _server_socket(args).unlink(missing_ok=True)
    return 0


def _stop_server(args: argparse.Namespace) -> int:
    health = _server_health(args)
    if health is not None:
        try:
            request_reference_socket(
                _server_socket(args),
                {"action": "stop"},
                timeout_s=min(5.0, args.server_query_timeout_s),
            )
        except (OSError, MragReferenceServerError):
            pass
    else:
        pid = _read_pid(_server_pid(args))
        if pid and _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        pid = _read_pid(_server_pid(args))
        if _server_health(args) is None and not (pid and _pid_alive(pid)):
            _server_socket(args).unlink(missing_ok=True)
            _server_pid(args).unlink(missing_ok=True)
            print(json.dumps({"stopped": True, "server_dir": str(args.server_dir)}))
            return 0
        time.sleep(0.1)
    print(json.dumps({"stopped": False, "error": "server_stop_timeout"}), file=sys.stderr)
    return 1


def _load_upstream_config(args: argparse.Namespace) -> Any:
    os.environ["MRAG_BASE_DIR"] = str(args.mrag_dir)
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))
    from mrag.config import CFG

    return CFG


def _init_mode_pipeline(config: Any, mode: str) -> SimpleNamespace:
    _install_flagembedding_transformers_compat()
    from mrag.embeddings import ImageEmbedder, Reranker, TextEmbedder
    from mrag.vector_store import VectorStore

    pipeline = SimpleNamespace()
    pipeline.store = VectorStore(config.qdrant_dir)
    pipeline.mrag_dir = config.base_dir
    pipeline.text = TextEmbedder(config.bge_m3_model).load()
    pipeline.image = None
    pipeline.kg = None
    pipeline.rerank = None
    if mode in GRAPH_MODES:
        from mrag.kg import KG, read as kg_read

        pipeline.kg = KG(kg_read(config.graph_pickle))
    if mode in RERANK_MODES:
        pipeline.rerank = Reranker(config.reranker_model).load()
    if mode in VISUAL_MODES:
        pipeline.image = ImageEmbedder(config.colqwen_model).load()
    return pipeline


def _install_flagembedding_transformers_compat() -> None:
    """Translate FlagEmbedding 1.4's new ``dtype`` kwarg for Transformers 4.54."""
    from transformers import AutoModel

    if getattr(AutoModel, "_gems_rag_dtype_compat", False):
        return
    original = AutoModel.from_pretrained

    def from_pretrained(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original(*args, **kwargs)

    AutoModel.from_pretrained = classmethod(from_pretrained)
    AutoModel._gems_rag_dtype_compat = True


def _ensure_server(args: argparse.Namespace) -> dict[str, Any]:
    expected = _server_fingerprint(args)
    health = _server_health(args)
    if _server_matches(health, args, expected):
        return health

    args.server_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.server_dir / "start.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        health = _server_health(args)
        if _server_matches(health, args, expected):
            return health
        if health is not None:
            request_reference_socket(
                _server_socket(args),
                {"action": "stop"},
                timeout_s=min(5.0, args.server_query_timeout_s),
            )
            _wait_for_server_exit(args, timeout_s=10)

        existing_pid = _read_pid(_server_pid(args))
        if existing_pid and _pid_alive(existing_pid):
            return _wait_for_server(args, expected, process=None)
        _server_pid(args).unlink(missing_ok=True)
        _server_socket(args).unlink(missing_ok=True)

        log_path = args.server_dir / "server.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                _server_command(args),
                cwd=ROOT,
                stdout=log,
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
        if health is not None:
            raise MragReferenceServerError(
                "MRAG server started with a different corpus or adapter fingerprint"
            )
        if process is not None and process.poll() is not None:
            raise MragReferenceServerError(
                f"MRAG server exited with code {process.returncode}: {_server_log_tail(args)}"
            )
        time.sleep(0.25)
    raise MragReferenceServerError(
        f"MRAG server did not become ready within {args.server_startup_timeout_s:g}s"
    )


def _wait_for_server_exit(args: argparse.Namespace, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pid = _read_pid(_server_pid(args))
        if _server_health(args) is None and not (pid and _pid_alive(pid)):
            return
        time.sleep(0.1)
    raise MragReferenceServerError("stale MRAG server did not stop")


def _server_health(args: argparse.Namespace) -> dict[str, Any] | None:
    socket_path = _server_socket(args)
    if not socket_path.exists():
        return None
    try:
        response = request_reference_socket(
            socket_path,
            {"action": "health"},
            timeout_s=0.5,
        )
    except (OSError, MragReferenceServerError, TimeoutError):
        return None
    return response if response.get("ok") else None


def _server_matches(
    health: dict[str, Any] | None,
    args: argparse.Namespace,
    expected_fingerprint: str,
) -> bool:
    return bool(
        health
        and health.get("status") == "ready"
        and health.get("fingerprint") == expected_fingerprint
        and health.get("mrag_dir") == str(args.mrag_dir.resolve())
    )


def _server_fingerprint(args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema:{SERVER_SCHEMA_VERSION}\n".encode())
    for path in [
        Path(__file__).resolve(),
        ROOT / "src" / "gems_rag" / "mrag_reference_modes.py",
        ROOT / "src" / "gems_rag" / "mrag_reference_server.py",
    ]:
        digest.update(path.read_bytes())
    digest.update(str(args.mrag_dir.resolve()).encode())
    cache_paths = [
        args.mrag_dir / "mmrag_cache_v3" / "chunks.jsonl",
        args.mrag_dir / "mmrag_cache_v3" / "graph.gpickle",
    ]
    qdrant_dir = args.mrag_dir / "qdrant_db"
    if qdrant_dir.exists():
        cache_paths.extend(path for path in sorted(qdrant_dir.rglob("*")) if path.is_file())
    for path in cache_paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            digest.update(f"missing:{path}\n".encode())
            continue
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo",
        str(args.repo),
        "--mrag-dir",
        str(args.mrag_dir),
        "--python",
        sys.executable,
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
        "serve",
    ]


def _server_socket(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "mrag.sock"


def _server_pid(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "server.pid"


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{pid}\n", encoding="utf-8")
    temporary.replace(path)


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _remove_owned_pid(path: Path, pid: int) -> None:
    if _read_pid(path) == pid:
        path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _server_log_tail(args: argparse.Namespace, limit: int = 4000) -> str:
    path = Path(getattr(args, "server_dir", DEFAULT_SERVER_DIR)) / "server.log"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return "server log unavailable"


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
