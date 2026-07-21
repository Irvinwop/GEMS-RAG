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

from .comparison_study import (
    BENCHMARK_ID,
    BENCHMARK_QUESTION_COUNT,
    BENCHMARK_SHA256,
    COMPARISON_CONTEXT_MODES,
    COMPARISON_MAX_EVIDENCE_CHARS,
    COMPARISON_RETRIEVERS,
    COMPARISON_TOP_K,
    bundle_comparison,
    comparison_contract,
    validate_comparison_run,
)
from .config import DatasetConfig, ExperimentConfig, GraderConfig, RetrieverConfig, incompatible_context_modes, load_experiment_config, rag_backend_to_dict, write_experiment_config
from .credentials import clear_credential, credential_status, load_local_env, set_credential
from .datasets import DEFAULT_DATASET_ID, dataset_catalog, get_dataset_spec
from .manual import manual_status
from .model_catalog import catalog_entries_to_models_payload, load_model_catalog
from .planning import plan_experiment
from .rag_backends import configure_retriever_backend, rag_backend_from_payload, rag_backend_presets_payload
from .retriever_catalog import catalog_entries_to_retrievers_payload, load_retriever_catalog
from .retrieval_snapshots import (
    default_retrieval_snapshot_path,
    retrieval_snapshot_status,
)
from .run_bundles import export_run_bundle, import_pro_grades

ROOT = Path(__file__).resolve().parents[2]
GUI_WORKING_DIR = ROOT / "data" / "working" / "gui"
MODEL_CATALOG = ROOT / "configs" / "model-catalog.example.json"
RETRIEVER_CATALOG = ROOT / "configs" / "retriever-catalog.example.json"
DEFAULT_GRADER_SPEC = ROOT / "docs" / "MUTCD_RAG_EVALUATION_SPECIFICATION.md"
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
        datasets = dataset_catalog(self.root)
        default_dataset = next(row for row in datasets if row["id"] == DEFAULT_DATASET_ID)
        return {
            "project": {"name": "GEMS-RAG", "root": str(self.root)},
            "manual": self._manual,
            "dataset": default_dataset,
            "datasets": datasets,
            "default_dataset": DEFAULT_DATASET_ID,
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
            "rag_backend_presets": rag_backend_presets_payload(),
            "comparison_study": {
                "benchmark_id": BENCHMARK_ID,
                "question_count": BENCHMARK_QUESTION_COUNT,
                "question_sha256": BENCHMARK_SHA256,
                "retrievers": list(COMPARISON_RETRIEVERS),
                "context_modes": list(COMPARISON_CONTEXT_MODES),
                "top_k": COMPARISON_TOP_K,
                "max_evidence_chars": COMPARISON_MAX_EVIDENCE_CHARS,
            },
            "grader_specification": {
                "path": str(DEFAULT_GRADER_SPEC),
                "available": DEFAULT_GRADER_SPEC.is_file(),
            },
            "runs": self.list_runs(),
            "jobs": self.jobs.list(),
        }

    def materialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _experiment_name(payload.get("name"))
        output_dir = self._output_dir(payload.get("output_dir"))
        zip_name = _zip_filename(payload.get("zip_name"), experiment_name=name)
        ingestion_mode = str(payload.get("ingestion_mode") or "shared_corpus")
        if ingestion_mode not in {"shared_corpus", "native_pdf"}:
            raise ValueError("ingestion_mode must be shared_corpus or native_pdf")
        top_k = _bounded_int(payload.get("top_k", 6), 1, 100, "top_k")
        limit_value = payload.get("limit")
        limit = None if limit_value in {None, ""} else _bounded_int(limit_value, 1, 100000, "limit")
        max_evidence = _bounded_int(payload.get("max_evidence_chars", 1600), 100, 100000, "max_evidence_chars")
        dataset_id = str(payload.get("dataset") or DEFAULT_DATASET_ID)
        dataset_spec = get_dataset_spec(dataset_id)
        qa_path = dataset_spec.qa_path if dataset_spec.qa_path.is_absolute() else self.root / dataset_spec.qa_path
        if not qa_path.is_file():
            raise FileNotFoundError(f"dataset is unavailable: {qa_path}")

        retriever_entries = load_retriever_catalog(self.root / "configs" / "retriever-catalog.example.json")
        selected_retrievers = set(_string_list(payload.get("retrievers")))
        selected_entries = [entry for entry in retriever_entries if entry.config.name in selected_retrievers]
        rag_backend = rag_backend_from_payload(payload.get("rag_backend"))
        retrievers = [
            configure_retriever_backend(
                _retriever_for_ingestion(replace(entry.config, top_k=top_k), entry.family, ingestion_mode),
                entry.family,
                rag_backend,
            )
            for entry in selected_entries
        ]
        missing_retrievers = selected_retrievers - {entry.config.name for entry in retriever_entries}
        if missing_retrievers:
            raise ValueError(f"unknown retrievers: {', '.join(sorted(missing_retrievers))}")
        if not retrievers:
            raise ValueError("select at least one retriever")
        gold_only = [entry.config.name for entry in selected_entries if entry.interaction == "gold_reference"]
        if gold_only and not dataset_spec.includes_gold_references:
            raise ValueError(
                f"dataset {dataset_id} has no gold references required by: {', '.join(sorted(gold_only))}"
            )

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
        incompatible = {
            retriever.name: incompatible_context_modes(retriever, context_modes)
            for retriever in retrievers
            if incompatible_context_modes(retriever, context_modes)
        }
        if incompatible:
            details = "; ".join(f"{name}: {', '.join(modes)}" for name, modes in incompatible.items())
            raise ValueError(f"RAG/context combinations are incompatible: {details}")
        retrievers = [
            replace(retriever, context_modes=tuple(context_modes))
            for retriever in retrievers
        ]

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
                qa_path=dataset_spec.qa_path,
                mrag_dir=dataset_spec.mrag_dir,
                limit=limit,
            ),
            retrievers=retrievers,
            context_modes=context_modes,
            models=models,
            grader=grader,
            rag_backend=rag_backend,
            output_dir=output_dir,
            max_evidence_chars=max_evidence,
            dry_run=bool(payload.get("dry_run", False)),
        )
        if context_modes == ["injected"]:
            config = replace(
                config,
                retrieval_snapshot=default_retrieval_snapshot_path(config, root=self.root),
            )
        config_path = self.root / "data" / "working" / "gui" / "configs" / f"{name}.json"
        write_experiment_config(config, config_path)
        request_path = config_path.with_suffix(".request.json")
        request_path.write_text(
            json.dumps(
                {
                    **payload,
                    "name": name,
                    "output_dir": str(output_dir),
                    "zip_name": zip_name,
                    "dataset": dataset_id,
                    "ingestion_mode": ingestion_mode,
                    "grader_mode": grader_mode,
                    "grader_spec": str(DEFAULT_GRADER_SPEC),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        plan = plan_experiment(config)
        run_dir = output_dir / name
        return {
            "status": "ready",
            "config_path": str(config_path),
            "request_path": str(request_path),
            "grader_mode": grader_mode,
            "dataset": dataset_id,
            "ingestion_mode": ingestion_mode,
            "rag_backend": rag_backend_to_dict(rag_backend),
            "plan": plan,
            "artifacts": {
                "output_dir": str(output_dir),
                "run_dir": str(run_dir),
                "runs_path": str(run_dir / "runs.jsonl"),
                "zip_name": zip_name,
                "zip_path": str(run_dir / zip_name),
                "grader_spec": str(DEFAULT_GRADER_SPEC),
                "retrieval_snapshot": (
                    str(config.retrieval_snapshot)
                    if config.retrieval_snapshot is not None
                    else None
                ),
            },
        }

    def set_credential(self, payload: dict[str, Any]) -> dict[str, Any]:
        return set_credential(str(payload.get("name") or ""), str(payload.get("value") or ""), self.env_path)

    def clear_credential(self, payload: dict[str, Any]) -> dict[str, Any]:
        return clear_credential(str(payload.get("name") or ""), self.env_path)

    def create_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        runs = self._root_path(payload.get("runs"), must_exist=True)
        mode = str(payload.get("mode") or "gpt_pro")
        name = runs.parent.name if runs.is_file() else runs.name
        run_file = runs / "runs.jsonl" if runs.is_dir() else runs
        zip_name = _zip_filename(payload.get("zip_name"), experiment_name=name, mode=mode)
        output = run_file.parent / zip_name
        grader_spec = self._grader_spec_path(payload.get("grader_spec"))
        config_path = run_file.parent / "materialized_config.json"
        if config_path.is_file():
            config = load_experiment_config(config_path)
            if comparison_contract(config, root=self.root)["ok"]:
                return bundle_comparison(
                    config_path,
                    runs_path=run_file,
                    output_path=output,
                    grader_spec_path=grader_spec,
                    root=self.root,
                )
        return export_run_bundle(
            runs,
            output_path=output,
            mode=mode,
            grader_spec_path=grader_spec,
        )

    def run_status(self, config_value: Any, zip_value: Any = None) -> dict[str, Any]:
        config_path = self._root_path(config_value, must_exist=True)
        if config_path.suffix.lower() != ".json":
            raise ValueError("config_path must be a project JSON file")
        config = load_experiment_config(config_path)
        output_dir = config.output_dir if config.output_dir.is_absolute() else self.root / config.output_dir
        run_dir = output_dir.resolve() / config.name
        if not run_dir.is_relative_to(self.root):
            raise ValueError("run output must stay inside the project")
        runs_path = run_dir / "runs.jsonl"
        zip_name = _zip_filename(zip_value, experiment_name=config.name)
        zip_path = run_dir / zip_name
        counts = _jsonl_progress(runs_path)
        expected_rows = int(plan_experiment(config)["estimates"]["rows"])
        comparison = comparison_contract(config, root=self.root)
        snapshot = (
            retrieval_snapshot_status(config, root=self.root)
            if config.retrieval_snapshot is not None
            else None
        )
        operational = None
        if comparison["ok"] and runs_path.is_file():
            operational = validate_comparison_run(config, runs_path=runs_path, root=self.root)
        complete = counts["completed_rows"] >= expected_rows and counts["invalid_rows"] == 0
        if operational is not None:
            complete = operational["ok"]
        resumable = runs_path.is_file() and counts["completed_rows"] < expected_rows
        if snapshot is not None and snapshot["status"] in {
            "ready_to_build",
            "ready_to_resume",
            "incomplete",
        }:
            resumable = True
        if operational is not None and not operational["ok"]:
            resumable = True
        return {
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "runs_path": str(runs_path),
            "zip_path": str(zip_path),
            "zip_exists": zip_path.is_file(),
            "expected_rows": expected_rows,
            **counts,
            "complete": complete,
            "resumable": resumable,
            "retrieval_snapshot": snapshot,
            "operational_validation": operational,
        }

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

    def _output_dir(self, value: Any) -> Path:
        path = Path(str(value or "runs")).expanduser()
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("output directory must stay inside the project")
        return resolved

    def _grader_spec_path(self, value: Any = None) -> Path:
        return _resolve_grader_spec(self.root, value)


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "")
        if action not in {"preflight", "rag_audit", "run", "external_indexes"}:
            raise ValueError("unsupported job action")
        config = self._config_path(payload.get("config_path"))
        job_id = uuid.uuid4().hex[:12]
        report_path = None
        command = [sys.executable, "-m", "gems_rag.cli"]
        if action == "preflight":
            command.extend(["preflight", str(config)])
            if not bool(payload.get("external_checks", False)):
                command.append("--no-external-checks")
        elif action == "rag_audit":
            report_path = self.root / "data" / "working" / "gui" / "audits" / f"{job_id}.json"
            timeout_s = _bounded_int(payload.get("timeout_s", 30), 1, 300, "timeout_s")
            command.extend(
                ["rag-audit", str(config), "--timeout-s", str(timeout_s), "--output", str(report_path)]
            )
            if not bool(payload.get("external_checks", True)):
                command.append("--no-external-checks")
        elif action == "run":
            command.extend(["run", str(config)])
            run_mode = str(payload.get("run_mode") or "resume")
            if run_mode not in {"overwrite", "resume", "retry_errors"}:
                raise ValueError("invalid run_mode")
            command.append(f"--{run_mode.replace('_', '-')}")
            experiment = load_experiment_config(config)
            output_dir = experiment.output_dir if experiment.output_dir.is_absolute() else self.root / experiment.output_dir
            run_dir = (output_dir / experiment.name).resolve()
            if not run_dir.is_relative_to(self.root):
                raise ValueError("run output must stay inside the project")
            zip_name = _zip_filename(payload.get("zip_name"), experiment_name=experiment.name)
        else:
            command.extend(["external-indexes", "--config", str(config)])
            ingestion = str(payload.get("ingestion_mode") or "shared_corpus")
            if ingestion not in {"shared_corpus", "native_pdf"}:
                raise ValueError("invalid ingestion_mode")
            command.extend(["--ingestion-mode", ingestion])
            if bool(payload.get("dry_run", False)):
                command.append("--dry-run")
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
        if report_path is not None:
            job["report_path"] = str(report_path)
            job["report"] = None
            job["report_error"] = None
        if action == "run":
            job["runs_path"] = str(run_dir / "runs.jsonl")
            job["zip_path"] = str(run_dir / zip_name)
            job["bundle"] = None
            job["bundle_error"] = None
            job["grader_spec_path"] = str(
                _resolve_grader_spec(self.root, payload.get("grader_spec"))
            )
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
            bundle = None
            bundle_error = None
            report = None
            report_error = None
            with self._lock:
                report_path = self._jobs[job_id].get("report_path")
            if returncode == 0 and report_path:
                try:
                    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
                except Exception as exc:
                    report_error = f"{type(exc).__name__}: {exc}"
            if returncode == 0:
                with self._lock:
                    runs_path = self._jobs[job_id].get("runs_path")
                    zip_path = self._jobs[job_id].get("zip_path")
                    grader_spec_path = self._jobs[job_id].get("grader_spec_path")
                    config_path = self._jobs[job_id].get("config_path")
                if runs_path and zip_path:
                    try:
                        config = load_experiment_config(Path(config_path))
                        if comparison_contract(config, root=self.root)["ok"]:
                            bundle = bundle_comparison(
                                Path(config_path),
                                runs_path=Path(runs_path),
                                output_path=Path(zip_path),
                                grader_spec_path=Path(grader_spec_path),
                                root=self.root,
                            )
                        else:
                            bundle = export_run_bundle(
                                Path(runs_path),
                                output_path=Path(zip_path),
                                mode="gpt_pro",
                                grader_spec_path=(
                                    Path(grader_spec_path) if grader_spec_path else None
                                ),
                            )
                    except Exception as exc:
                        bundle_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                job = self._jobs[job_id]
                job["bundle"] = bundle
                job["bundle_error"] = bundle_error
                if report_path:
                    job["report"] = report
                    job["report_error"] = report_error
                job["returncode"] = 2 if bundle_error or report_error else returncode
                job["status"] = "complete" if returncode == 0 and not bundle_error and not report_error else "failed"
                if bundle:
                    job["logs"].append(f"ZIP written: {bundle['output']}")
                if bundle_error:
                    job["logs"].append(f"ZIP export failed: {bundle_error}")
                if report_error:
                    job["logs"].append(f"RAG audit report failed: {report_error}")
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
        if parsed.path == "/api/run-status":
            query = parse_qs(parsed.query)
            try:
                return self._json(
                    self.server.control.run_status(
                        (query.get("config_path") or [None])[0],
                        (query.get("zip_name") or [None])[0],
                    )
                )
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
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
    name = re.sub(
        r"[^a-z0-9-]+",
        "-",
        str(value or "mutcd-rag-comparison").strip().lower(),
    ).strip("-")
    if not name:
        raise ValueError("comparison name is required")
    return name[:80]


def _resolve_grader_spec(root: Path, value: Any = None) -> Path:
    path = Path(str(value or root / "docs" / DEFAULT_GRADER_SPEC.name)).expanduser()
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("grader specification must stay inside the project")
    if resolved.suffix.lower() != ".md" or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


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


def _zip_filename(value: Any, *, experiment_name: str, mode: str = "gpt_pro") -> str:
    mode_slug = re.sub(r"[^a-z0-9-]+", "-", mode.lower().replace("_", "-")).strip("-") or "bundle"
    filename = str(value or f"{experiment_name}-{mode_slug}.zip").strip()
    if not filename.lower().endswith(".zip"):
        filename += ".zip"
    if filename in {".zip", ".", ".."} or len(filename) > 180:
        raise ValueError("ZIP name must be a non-empty filename no longer than 180 characters")
    if Path(filename).name != filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise ValueError("ZIP name must be a filename, not a path")
    return filename


def _jsonl_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "rows_on_disk": 0,
            "completed_rows": 0,
            "invalid_rows": 0,
            "bytes": 0,
            "modified_at": None,
        }

    completed: set[tuple[str, str, str, str, str]] = set()
    rows_on_disk = 0
    invalid_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_on_disk += 1
            try:
                row = json.loads(line)
                config = row.get("config") or {}
                key = (
                    str(row.get("qa_id") or ""),
                    str(config.get("retriever") or ""),
                    str(config.get("context_mode") or ""),
                    str(config.get("model_provider") or ""),
                    str(config.get("model") or ""),
                )
                if not all(key):
                    raise ValueError("missing run identity")
                completed.add(key)
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid_rows += 1
    stat = path.stat()
    return {
        "rows_on_disk": rows_on_disk,
        "completed_rows": len(completed),
        "invalid_rows": invalid_rows,
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _redact_log(line: str) -> str:
    return re.sub(r"(?i)(api[_-]?key|authorization|password)([\"'\s:=]+)[^\s,}\]]+", r"\1\2[REDACTED]", line)
