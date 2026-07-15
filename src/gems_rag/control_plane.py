from __future__ import annotations

import json
import base64
import os
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import DatasetConfig, ExperimentConfig, GraderConfig, RetrieverConfig, write_experiment_config
from .credentials import clear_credential, credential_status, load_local_env, set_credential
from .manual import DEFAULT_MRAG_DIR, manual_status
from .model_catalog import catalog_entries_to_models_payload, load_model_catalog
from .planning import plan_experiment
from .retriever_catalog import catalog_entries_to_retrievers_payload, load_retriever_catalog
from .run_bundles import export_run_bundle, import_pro_grades

ROOT = Path(__file__).resolve().parents[2]
GUI_WORKING_DIR = ROOT / "data" / "working" / "gui"
MODEL_CATALOG = ROOT / "configs" / "model-catalog.example.json"
RETRIEVER_CATALOG = ROOT / "configs" / "retriever-catalog.example.json"
STATIC_DIR = Path(__file__).resolve().parent / "web"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ControlPlane:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root.resolve()
        self.env_path = self.root / ".env"
        load_local_env(self.env_path)
        self._manual = manual_status(root=self.root)
        self.jobs = JobManager(self.root)

    def state(self) -> dict[str, Any]:
        models = load_model_catalog(self.root / "configs" / "model-catalog.example.json")
        retrievers = load_retriever_catalog(self.root / "configs" / "retriever-catalog.example.json")
        return {
            "project": {"name": "GEMS-RAG", "root": str(self.root)},
            "manual": self._manual,
            "credentials": credential_status(self.env_path),
            "catalogs": {
                "models": catalog_entries_to_models_payload(models)["models"],
                "retrievers": catalog_entries_to_retrievers_payload(retrievers)["retrievers"],
            },
            "context_modes": [
                {"name": "injected", "label": "Injected context"},
                {"name": "tool_explore", "label": "Explore selected hits"},
                {"name": "tool_search", "label": "Search then explore"},
                {"name": "tool_native", "label": "Native tool calls"},
            ],
            "runs": self.list_runs(),
            "jobs": self.jobs.list(),
        }

    def materialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _experiment_name(payload.get("name"))
        ingestion_mode = str(payload.get("ingestion_mode") or "shared_corpus")
        if ingestion_mode not in {"shared_corpus", "native_pdf"}:
            raise ValueError("ingestion_mode must be shared_corpus or native_pdf")
        top_k = _bounded_int(payload.get("top_k", 6), 1, 100, "top_k")
        limit_value = payload.get("limit")
        limit = None if limit_value in {None, ""} else _bounded_int(limit_value, 1, 100000, "limit")
        max_evidence = _bounded_int(payload.get("max_evidence_chars", 1600), 100, 100000, "max_evidence_chars")

        retriever_entries = load_retriever_catalog(self.root / "configs" / "retriever-catalog.example.json")
        selected_retrievers = set(_string_list(payload.get("retrievers")))
        retrievers = [
            _retriever_for_ingestion(replace(entry.config, top_k=top_k), entry.family, ingestion_mode)
            for entry in retriever_entries
            if entry.config.name in selected_retrievers
        ]
        missing_retrievers = selected_retrievers - {entry.config.name for entry in retriever_entries}
        if missing_retrievers:
            raise ValueError(f"unknown retrievers: {', '.join(sorted(missing_retrievers))}")
        if not retrievers:
            raise ValueError("select at least one retriever")

        model_entries = load_model_catalog(self.root / "configs" / "model-catalog.example.json")
        selected_models = set(_string_list(payload.get("models")))
        answer_entries = [entry for entry in model_entries if "answer" in entry.roles]
        models = [entry.config for entry in answer_entries if _model_id(entry.config.provider, entry.config.model) in selected_models]
        known_models = {_model_id(entry.config.provider, entry.config.model) for entry in answer_entries}
        missing_models = selected_models - known_models
        if missing_models:
            raise ValueError(f"unknown models: {', '.join(sorted(missing_models))}")
        if not models:
            raise ValueError("select at least one answer model")

        context_modes = _string_list(payload.get("context_modes"))
        known_contexts = {"injected", "tool_explore", "tool_search", "tool_native"}
        if not context_modes or set(context_modes) - known_contexts:
            raise ValueError("select one or more valid context modes")

        grader_mode = str(payload.get("grader_mode") or "heuristic")
        grader = GraderConfig()
        if grader_mode == "api":
            grader_id = str(payload.get("grader") or "")
            graders = [entry for entry in model_entries if "grader" in entry.roles]
            match = next((entry for entry in graders if _model_id(entry.config.provider, entry.config.model) == grader_id), None)
            if match is None:
                raise ValueError("select a valid API grader")
            grader = GraderConfig(provider=match.config.provider, model=match.config.model, options=match.config.options)
        elif grader_mode not in {"heuristic", "gpt_pro"}:
            raise ValueError("grader_mode must be heuristic, api, or gpt_pro")

        config = ExperimentConfig(
            name=name,
            dataset=DatasetConfig(
                qa_path=DEFAULT_MRAG_DIR / "eval" / "gold_qa.jsonl",
                mrag_dir=DEFAULT_MRAG_DIR,
                limit=limit,
            ),
            retrievers=retrievers,
            context_modes=context_modes,
            models=models,
            grader=grader,
            output_dir=Path("runs"),
            max_evidence_chars=max_evidence,
            dry_run=bool(payload.get("dry_run", False)),
        )
        config_path = self.root / "data" / "working" / "gui" / "configs" / f"{name}.json"
        write_experiment_config(config, config_path)
        request_path = config_path.with_suffix(".request.json")
        request_path.write_text(json.dumps({**payload, "ingestion_mode": ingestion_mode, "grader_mode": grader_mode}, indent=2) + "\n", encoding="utf-8")
        plan = plan_experiment(config)
        return {
            "status": "ready",
            "config_path": str(config_path),
            "request_path": str(request_path),
            "grader_mode": grader_mode,
            "ingestion_mode": ingestion_mode,
            "plan": plan,
        }

    def set_credential(self, payload: dict[str, Any]) -> dict[str, Any]:
        return set_credential(str(payload.get("name") or ""), str(payload.get("value") or ""), self.env_path)

    def clear_credential(self, payload: dict[str, Any]) -> dict[str, Any]:
        return clear_credential(str(payload.get("name") or ""), self.env_path)

    def create_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        runs = self._root_path(payload.get("runs"), must_exist=True)
        mode = str(payload.get("mode") or "gpt_pro")
        name = runs.parent.name if runs.is_file() else runs.name
        output = self.root / "data" / "working" / "bundles" / f"{name}-{mode}.zip"
        qa_path = self.root / DEFAULT_MRAG_DIR / "eval" / "gold_qa.jsonl"
        return export_run_bundle(runs, output_path=output, qa_path=qa_path, mode=mode)

    def import_grades(self, payload: dict[str, Any]) -> dict[str, Any]:
        runs = self._root_path(payload.get("runs"), must_exist=True)
        encoded = payload.get("grades_base64")
        if encoded:
            filename = str(payload.get("grades_filename") or "grades.jsonl")
            suffix = ".zip" if filename.lower().endswith(".zip") else ".jsonl"
            try:
                content = base64.b64decode(str(encoded), validate=True)
            except ValueError as exc:
                raise ValueError("invalid grades file encoding") from exc
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("grades file must be 20 MB or smaller")
            grades = self.root / "data" / "working" / "gui" / "imports" / f"{uuid.uuid4().hex}{suffix}"
            grades.parent.mkdir(parents=True, exist_ok=True)
            grades.write_bytes(content)
        else:
            grades = self._root_path(payload.get("grades"), must_exist=True)
        run_file = runs / "runs.jsonl" if runs.is_dir() else runs
        output = run_file.parent / "gpt-pro-graded-runs.jsonl"
        return import_pro_grades(run_file, grades, output_path=output)

    def list_runs(self) -> list[dict[str, Any]]:
        run_root = self.root / "runs"
        rows = []
        if not run_root.is_dir():
            return rows
        for runs_path in sorted(run_root.glob("*/runs.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
            rows.append(
                {
                    "name": runs_path.parent.name,
                    "path": str(runs_path),
                    "rows": _line_count(runs_path),
                    "bytes": runs_path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(runs_path.stat().st_mtime, UTC).isoformat(),
                    "has_config": (runs_path.parent / "materialized_config.json").is_file(),
                    "has_gpt_pro_grades": (runs_path.parent / "gpt-pro-graded-runs.jsonl").is_file(),
                }
            )
        return rows

    def _root_path(self, value: Any, *, must_exist: bool) -> Path:
        if not value:
            raise ValueError("path is required")
        path = Path(str(value))
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("path must stay inside the project")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "")
        if action not in {"preflight", "run", "external_indexes"}:
            raise ValueError("unsupported job action")
        config = self._config_path(payload.get("config_path"))
        command = [sys.executable, "-m", "gems_rag.cli"]
        if action == "preflight":
            command.extend(["preflight", str(config)])
            if not bool(payload.get("external_checks", False)):
                command.append("--no-external-checks")
        elif action == "run":
            command.extend(["run", str(config)])
            run_mode = str(payload.get("run_mode") or "resume")
            if run_mode not in {"overwrite", "resume", "retry_errors"}:
                raise ValueError("invalid run_mode")
            command.append(f"--{run_mode.replace('_', '-')}")
        else:
            command.extend(["external-indexes", "--config", str(config)])
            ingestion = str(payload.get("ingestion_mode") or "shared_corpus")
            if ingestion not in {"shared_corpus", "native_pdf"}:
                raise ValueError("invalid ingestion_mode")
            command.extend(["--ingestion-mode", ingestion])
            if bool(payload.get("dry_run", False)):
                command.append("--dry-run")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "action": action,
            "status": "queued",
            "command": command[2:],
            "config_path": str(config),
            "created_at": datetime.now(UTC).isoformat(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "logs": [],
        }
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id, command), daemon=True).start()
        return dict(job)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job, logs=list(job["logs"])) for job in reversed(self._jobs.values())]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job, logs=list(job["logs"])) if job else None

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            process = self._processes.get(job_id)
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("unknown job")
        if process is not None and process.poll() is None:
            process.terminate()
        return self.get(job_id) or job

    def _run(self, job_id: str, command: list[str]) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.root / "src")
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = datetime.now(UTC).isoformat()
        try:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._processes[job_id] = process
            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    logs = self._jobs[job_id]["logs"]
                    logs.append(_redact_log(line.rstrip()))
                    del logs[:-500]
            returncode = process.wait()
            with self._lock:
                job = self._jobs[job_id]
                job["returncode"] = returncode
                job["status"] = "complete" if returncode == 0 else "failed"
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["returncode"] = 127
                job["logs"].append(f"job failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._jobs[job_id]["finished_at"] = datetime.now(UTC).isoformat()
                self._processes.pop(job_id, None)

    def _config_path(self, value: Any) -> Path:
        if not value:
            raise ValueError("config_path is required")
        path = Path(str(value))
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.suffix != ".json":
            raise ValueError("config_path must be an existing project JSON file")
        return path


class ControlPlaneHandler(BaseHTTPRequestHandler):
    server: "ControlPlaneServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            return self._json(self.server.control.state())
        if parsed.path == "/api/jobs":
            return self._json({"jobs": self.server.control.jobs.list()})
        if parsed.path.startswith("/api/jobs/"):
            job = self.server.control.jobs.get(parsed.path.rsplit("/", 1)[-1])
            return self._json(job or {"error": "not_found"}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
        if parsed.path == "/api/manual/page/1":
            manual = self.server.control._manual["manual"]
            image = Path(manual["path"]).parent / "page_images" / "page_0001.png"
            return self._file(image, "image/png")
        if parsed.path == "/api/download":
            query = parse_qs(parsed.query)
            try:
                path = self.server.control._root_path((query.get("path") or [None])[0], must_exist=True)
                if path.suffix.lower() != ".zip":
                    raise ValueError("only ZIP downloads are allowed")
                return self._file(path, "application/zip", attachment=True)
            except (ValueError, FileNotFoundError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self._static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        if not self._trusted_origin():
            return self._json({"error": "untrusted_origin"}, HTTPStatus.FORBIDDEN)
        try:
            payload = self._body_json()
            if self.path == "/api/configs":
                return self._json(self.server.control.materialize(payload), HTTPStatus.CREATED)
            if self.path == "/api/credentials":
                return self._json(self.server.control.set_credential(payload))
            if self.path == "/api/credentials/clear":
                return self._json(self.server.control.clear_credential(payload))
            if self.path == "/api/jobs":
                return self._json(self.server.control.jobs.start(payload), HTTPStatus.ACCEPTED)
            if self.path.endswith("/cancel") and self.path.startswith("/api/jobs/"):
                return self._json(self.server.control.jobs.cancel(self.path.split("/")[-2]))
            if self.path == "/api/bundles":
                return self._json(self.server.control.create_bundle(payload), HTTPStatus.CREATED)
            if self.path == "/api/import-grades":
                return self._json(self.server.control.import_grades(payload))
            return self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _trusted_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.hostname in LOOPBACK_HOSTS

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 28_000_000:
            raise ValueError("request body too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request JSON must be an object")
        return value

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str, *, attachment: bool = False) -> None:
        if not path.is_file():
            return self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        path = (STATIC_DIR / relative).resolve()
        if not path.is_relative_to(STATIC_DIR.resolve()) or not path.is_file():
            return self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        content_types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)


class ControlPlaneServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], control: ControlPlane) -> None:
        self.control = control
        super().__init__(address, ControlPlaneHandler)


def serve_gui(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("the credential-bearing GUI can only bind to a loopback host")
    control = ControlPlane()
    server = ControlPlaneServer((host, port), control)
    url = f"http://{host}:{server.server_port}/"
    print(f"GEMS-RAG model picker: {url}", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _retriever_for_ingestion(config: RetrieverConfig, family: str, ingestion_mode: str) -> RetrieverConfig:
    if ingestion_mode != "native_pdf" or family not in {"paperqa2", "raganything"}:
        return config
    options = dict(config.options)
    for key in ["command", "check_command"]:
        command = options.get(key)
        if isinstance(command, list) and "--ingestion-mode" not in command:
            options[key] = [*command, "--ingestion-mode", "native_pdf"]
    return replace(config, options=options)


def _experiment_name(value: Any) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", str(value or "gui-ablation").strip().lower()).strip("-")
    if not name:
        raise ValueError("experiment name is required")
    return name[:80]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _model_id(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _bounded_int(value: Any, minimum: int, maximum: int, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return number


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _redact_log(line: str) -> str:
    return re.sub(r"(?i)(api[_-]?key|authorization|password)([\"'\s:=]+)[^\s,}\]]+", r"\1\2[REDACTED]", line)
