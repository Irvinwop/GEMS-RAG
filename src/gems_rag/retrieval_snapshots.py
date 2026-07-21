from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from .config import ExperimentConfig, RetrieverConfig, rag_backend_to_dict
from .data import load_qa_items
from .retrieval import Retriever, build_retriever
from .run_bundles import redact_secrets
from .types import Evidence, QAItem, RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalSnapshotIndex:
    path: Path
    snapshot_id: str
    snapshot_sha256: str
    rows: dict[tuple[str, str], dict[str, Any]]
    retriever_fingerprints: dict[str, str]

    def retriever(self, config: RetrieverConfig) -> "SnapshotRetriever":
        fingerprint = self.retriever_fingerprints.get(config.name)
        if fingerprint is None:
            raise ValueError(f"retrieval snapshot has no fingerprint for {config.name}")
        return SnapshotRetriever(config.name, fingerprint, self)


class SnapshotRetriever(Retriever):
    def __init__(
        self,
        name: str,
        fingerprint: str,
        snapshot: RetrievalSnapshotIndex,
    ) -> None:
        self.name = name
        self.fingerprint = fingerprint
        self.snapshot = snapshot

    def retrieve(self, item: QAItem) -> RetrievalResult:
        row = self.snapshot.rows.get((item.qa_id, self.name))
        if row is None:
            raise ValueError(
                f"retrieval snapshot is missing qa_id={item.qa_id!r}, retriever={self.name!r}"
            )
        expected_question = question_fingerprint(item.qa_id, item.question)
        if row.get("question_fingerprint") != expected_question:
            raise ValueError(
                f"retrieval snapshot question mismatch for qa_id={item.qa_id!r}, retriever={self.name!r}"
            )
        if row.get("retriever_fingerprint") != self.fingerprint:
            raise ValueError(
                f"retrieval snapshot retriever mismatch for qa_id={item.qa_id!r}, retriever={self.name!r}"
            )
        evidence = [_evidence_from_record(record) for record in row.get("evidence", [])]
        source_debug = row.get("debug") if isinstance(row.get("debug"), dict) else {}
        return RetrievalResult(
            adapter=self.name,
            query=str(row.get("query") or item.question),
            evidence=evidence,
            debug={
                **source_debug,
                "snapshot_reused": True,
                "snapshot_id": self.snapshot.snapshot_id,
                "snapshot_sha256": self.snapshot.snapshot_sha256,
                "snapshot_path": str(self.snapshot.path),
                "source_retriever_fingerprint": self.fingerprint,
            },
            error=row.get("error"),
        )


def default_retrieval_snapshot_path(
    config: ExperimentConfig,
    *,
    root: Path = PROJECT_ROOT,
) -> Path:
    spec = _snapshot_spec(config, root=root)
    retriever_digest = _digest(
        [
            {"name": row["name"], "fingerprint": row["fingerprint"]}
            for row in spec["retrievers"]
        ]
    )[:16]
    return Path("data/working/retrieval-snapshots") / (
        f"{spec['dataset']['fingerprint'][:16]}-{retriever_digest}.jsonl"
    )


def build_retrieval_snapshot(
    config: ExperimentConfig,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
    retry_errors: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if overwrite and retry_errors:
        raise ValueError("overwrite and retry_errors are mutually exclusive")
    path = resolve_retrieval_snapshot_path(config, output_path=output_path, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = retrieval_snapshot_manifest_path(path)
    spec = _snapshot_spec(config, root=root)

    with _snapshot_lock(path):
        if overwrite:
            path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        repaired_tail = _repair_truncated_tail(path)
        rows, invalid_lines = _read_jsonl_lenient(path)
        if invalid_lines:
            raise ValueError(
                f"retrieval snapshot contains invalid JSON lines that are not a truncated tail: {invalid_lines[:5]}"
            )
        manifest = _load_manifest(manifest_path)
        if manifest and manifest.get("snapshot_sha256") and path.is_file():
            actual_sha256 = _sha256(path)
            if manifest["snapshot_sha256"] != actual_sha256:
                raise ValueError(
                    "retrieval snapshot SHA-256 does not match its completed manifest; "
                    "use a different path or --overwrite"
                )
        merged_retrievers = _guard_and_merge_manifest(manifest, spec, path)
        snapshot_id = spec["dataset"]["fingerprint"][:24]

        retry_archive = None
        if retry_errors:
            retry_keys = {
                _row_key(row)
                for row in rows
                if row.get("error") and _row_key(row)[1] in {item["name"] for item in spec["retrievers"]}
            }
            if retry_keys:
                retry_rows = [row for row in rows if _row_key(row) in retry_keys]
                retry_archive = _archive_retry_rows(path, retry_rows)
                rows = [row for row in rows if _row_key(row) not in retry_keys]
                _write_jsonl_atomic(path, rows)

        _write_json_atomic(
            manifest_path,
            _manifest_payload(
                path,
                spec,
                merged_retrievers,
                snapshot_id=snapshot_id,
                status="building",
                repaired_tail=repaired_tail,
                retry_archive=retry_archive,
            ),
        )

        completed = {_row_key(row) for row in rows}
        items = load_qa_items(
            _resolve(root, config.dataset.qa_path),
            limit=config.dataset.limit,
            qa_ids=config.dataset.qa_ids,
        )
        retrievers: list[tuple[RetrieverConfig, Retriever | None, str | None]] = []
        for retriever_config in config.retrievers:
            if all((item.qa_id, retriever_config.name) in completed for item in items):
                retrievers.append((retriever_config, None, None))
                continue
            try:
                retriever = build_retriever(
                    retriever_config,
                    _resolve(root, config.dataset.mrag_dir),
                )
                retrievers.append((retriever_config, retriever, None))
            except Exception as exc:
                retrievers.append(
                    (
                        retriever_config,
                        None,
                        f"retriever_build_failed: {type(exc).__name__}: {exc}",
                    )
                )

        written = 0
        skipped = 0
        with path.open("a", encoding="utf-8") as handle:
            for retriever_config, retriever, build_error in retrievers:
                fingerprint = retriever_fingerprint(retriever_config, root=root)
                for item in items:
                    key = (item.qa_id, retriever_config.name)
                    if key in completed:
                        skipped += 1
                        continue
                    started = time.monotonic()
                    result = _retrieve_live(retriever_config, retriever, item, build_error)
                    row = {
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
                        "snapshot_id": snapshot_id,
                        "qa_id": item.qa_id,
                        "question": item.question,
                        "question_fingerprint": question_fingerprint(item.qa_id, item.question),
                        "retriever": retriever_config.name,
                        "retriever_kind": retriever_config.kind,
                        "retriever_fingerprint": fingerprint,
                        "query": result.query,
                        "evidence": [_evidence_record(evidence) for evidence in result.evidence],
                        "debug": _json_safe(result.debug),
                        "error": result.error,
                        "latency_s": round(time.monotonic() - started, 3),
                    }
                    row["row_sha256"] = _row_sha256(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(key)
                    written += 1

        report = retrieval_snapshot_status(config, output_path=path, root=root)
        final_status = "complete" if report["ok"] else "failed"
        snapshot_sha256 = _sha256(path) if path.is_file() else None
        _write_json_atomic(
            manifest_path,
            _manifest_payload(
                path,
                spec,
                merged_retrievers,
                snapshot_id=snapshot_id,
                status=final_status,
                report=report,
                snapshot_sha256=snapshot_sha256,
                repaired_tail=repaired_tail,
                retry_archive=retry_archive,
            ),
        )

    final = retrieval_snapshot_status(config, output_path=path, root=root)
    return {
        **final,
        "rows_written": written,
        "rows_skipped": skipped,
        "truncated_tail_repaired": repaired_tail,
        "retry_archive": str(retry_archive) if retry_archive else None,
    }


def retrieval_snapshot_status(
    config: ExperimentConfig,
    *,
    output_path: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    try:
        path = resolve_retrieval_snapshot_path(config, output_path=output_path, root=root)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "not_configured",
            "path": None,
            "problems": [str(exc)],
        }
    manifest_path = retrieval_snapshot_manifest_path(path)
    spec = _snapshot_spec(config, root=root)
    expected = {
        (item.qa_id, retriever.name): (
            question_fingerprint(item.qa_id, item.question),
            retriever_fingerprint(retriever, root=root),
        )
        for retriever in config.retrievers
        for item in load_qa_items(
            _resolve(root, config.dataset.qa_path),
            limit=config.dataset.limit,
            qa_ids=config.dataset.qa_ids,
        )
    }
    problems: list[str] = []
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        problems.append(f"retrieval snapshot manifest is missing: {manifest_path}")
    elif manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        problems.append("retrieval snapshot manifest schema version is unsupported")
    elif manifest.get("dataset", {}).get("fingerprint") != spec["dataset"]["fingerprint"]:
        problems.append("retrieval snapshot dataset fingerprint does not match the experiment")

    rows, invalid_lines = _read_jsonl_lenient(path)
    if not path.is_file():
        problems.append(f"retrieval snapshot is missing: {path}")
    if invalid_lines:
        problems.append(f"invalid retrieval snapshot JSON lines: {len(invalid_lines)}")

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(_row_key(row), []).append(row)
    duplicate_keys = sorted(key for key, values in by_key.items() if len(values) > 1)
    missing_keys = sorted(set(expected) - set(by_key))
    mismatched_keys = []
    error_keys = []
    for key, identities in expected.items():
        candidates = by_key.get(key, [])
        if len(candidates) != 1:
            continue
        row = candidates[0]
        if (
            row.get("question_fingerprint") != identities[0]
            or row.get("retriever_fingerprint") != identities[1]
            or row.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
            or row.get("snapshot_id") != spec["dataset"]["fingerprint"][:24]
            or row.get("row_sha256") != _row_sha256(row)
        ):
            mismatched_keys.append(key)
        if row.get("error"):
            error_keys.append(key)
    if duplicate_keys:
        problems.append(f"duplicate retrieval snapshot rows: {len(duplicate_keys)}")
    if missing_keys:
        problems.append(f"missing retrieval snapshot rows: {len(missing_keys)}")
    if mismatched_keys:
        problems.append(f"mismatched retrieval snapshot rows: {len(mismatched_keys)}")
    if error_keys:
        problems.append(f"retrieval snapshot rows with errors: {len(error_keys)}")

    expected_retrievers = {
        row["name"]: row["fingerprint"] for row in spec["retrievers"]
    }
    manifest_retrievers = {
        str(row.get("name")): str(row.get("fingerprint"))
        for row in (manifest or {}).get("retrievers", [])
        if isinstance(row, dict)
    }
    conflicts = sorted(
        name
        for name, fingerprint in expected_retrievers.items()
        if name in manifest_retrievers and manifest_retrievers[name] != fingerprint
    )
    if conflicts:
        problems.append("retrieval snapshot retriever fingerprint conflicts: " + ", ".join(conflicts))

    recorded_sha256 = (manifest or {}).get("snapshot_sha256")
    actual_sha256 = _sha256(path) if path.is_file() else None
    if recorded_sha256 and recorded_sha256 != actual_sha256:
        problems.append("retrieval snapshot SHA-256 does not match its completed manifest")

    ok = not problems
    if ok:
        status = "ready"
    elif not path.is_file() or manifest is None:
        status = "ready_to_build"
    elif (
        manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or conflicts
        or invalid_lines
        or mismatched_keys
        or duplicate_keys
    ):
        status = "blocked"
    else:
        status = "incomplete"
    matching_rows = sum(key in expected for key in by_key)
    return {
        "ok": ok,
        "status": status,
        "path": str(path),
        "manifest_path": str(manifest_path),
        "snapshot_id": (manifest or {}).get("snapshot_id") or spec["dataset"]["fingerprint"][:24],
        "snapshot_sha256": actual_sha256,
        "expected_rows": len(expected),
        "matching_rows": matching_rows,
        "rows_on_disk": len(rows),
        "missing_rows": len(missing_keys),
        "duplicate_rows": len(duplicate_keys),
        "mismatched_rows": len(mismatched_keys),
        "error_rows": len(error_keys),
        "invalid_json_lines": len(invalid_lines),
        "missing_sample": [_key_record(key) for key in missing_keys[:20]],
        "error_sample": [_key_record(key) for key in error_keys[:20]],
        "problems": problems,
    }


def load_retrieval_snapshot(
    config: ExperimentConfig,
    *,
    output_path: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> RetrievalSnapshotIndex:
    status = retrieval_snapshot_status(config, output_path=output_path, root=root)
    if not status["ok"]:
        raise ValueError("retrieval snapshot is not ready: " + "; ".join(status["problems"]))
    path = Path(status["path"])
    rows, invalid_lines = _read_jsonl_lenient(path)
    if invalid_lines:
        raise ValueError(f"retrieval snapshot contains invalid JSON lines: {invalid_lines[:5]}")
    requested_names = {retriever.name for retriever in config.retrievers}
    selected = {
        _row_key(row): row
        for row in rows
        if str(row.get("retriever")) in requested_names
    }
    return RetrievalSnapshotIndex(
        path=path,
        snapshot_id=str(status["snapshot_id"]),
        snapshot_sha256=str(status["snapshot_sha256"]),
        rows=selected,
        retriever_fingerprints={
            retriever.name: retriever_fingerprint(retriever, root=root)
            for retriever in config.retrievers
        },
    )


def resolve_retrieval_snapshot_path(
    config: ExperimentConfig,
    *,
    output_path: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> Path:
    value = output_path or config.retrieval_snapshot
    if value is None:
        raise ValueError("experiment does not configure retrieval_snapshot")
    return _resolve(root, value)


def retrieval_snapshot_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def retriever_fingerprint(
    config: RetrieverConfig,
    *,
    root: Path = PROJECT_ROOT,
) -> str:
    return _digest(
        {
            "name": config.name,
            "kind": config.kind,
            "top_k": config.top_k,
            "options": redact_secrets(config.options),
            "context_modes": list(config.context_modes),
            "interaction": config.interaction,
            "implementation_files": _retriever_implementation_files(config, root=root),
        }
    )


def question_fingerprint(qa_id: str, question: str) -> str:
    return hashlib.sha256(f"{qa_id}\0{question}".encode("utf-8")).hexdigest()


def _snapshot_spec(config: ExperimentConfig, *, root: Path) -> dict[str, Any]:
    qa_path = _resolve(root, config.dataset.qa_path)
    items = load_qa_items(
        qa_path,
        limit=config.dataset.limit,
        qa_ids=config.dataset.qa_ids,
    )
    question_records = [
        {
            "qa_id": item.qa_id,
            "question_fingerprint": question_fingerprint(item.qa_id, item.question),
        }
        for item in items
    ]
    mrag_dir = _resolve(root, config.dataset.mrag_dir)
    dataset_payload = {
        "qa_path": str(qa_path),
        "qa_sha256": _sha256(qa_path),
        "mrag_dir": str(mrag_dir),
        "source_files": [
            _file_identity(path)
            for path in [
                mrag_dir / "mmrag_cache_v3" / "chunks.jsonl",
                mrag_dir / "mmrag_cache_v3" / "figures.jsonl",
                mrag_dir / "mmrag_cache_v3" / "graph.gpickle",
                mrag_dir / "mutcd11theditionr1hl.pdf",
            ]
        ],
        "limit": config.dataset.limit,
        "qa_ids": config.dataset.qa_ids,
        "questions": question_records,
    }
    dataset_payload["fingerprint"] = _digest(dataset_payload)
    return {
        "dataset": dataset_payload,
        "rag_backend": redact_secrets(rag_backend_to_dict(config.rag_backend)),
        "retrievers": [
            {
                "name": retriever.name,
                "kind": retriever.kind,
                "top_k": retriever.top_k,
                "fingerprint": retriever_fingerprint(retriever, root=root),
                "config": redact_secrets(
                    {
                        "name": retriever.name,
                        "kind": retriever.kind,
                        "top_k": retriever.top_k,
                        "options": retriever.options,
                        "context_modes": list(retriever.context_modes),
                        "interaction": retriever.interaction,
                    }
                ),
            }
            for retriever in config.retrievers
        ],
    }


def _guard_and_merge_manifest(
    manifest: dict[str, Any] | None,
    spec: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    if manifest is None:
        return list(spec["retrievers"])
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported retrieval snapshot manifest at {path}")
    if manifest.get("dataset", {}).get("fingerprint") != spec["dataset"]["fingerprint"]:
        raise ValueError(
            "retrieval snapshot dataset fingerprint does not match; use a different path or --overwrite"
        )
    merged = {
        str(row.get("name")): row
        for row in manifest.get("retrievers", [])
        if isinstance(row, dict) and row.get("name")
    }
    for descriptor in spec["retrievers"]:
        previous = merged.get(descriptor["name"])
        if previous and previous.get("fingerprint") != descriptor["fingerprint"]:
            raise ValueError(
                f"retrieval snapshot already contains a different configuration for {descriptor['name']}; "
                "use a different path or --overwrite"
            )
        merged[descriptor["name"]] = descriptor
    return [merged[name] for name in sorted(merged)]


def _manifest_payload(
    path: Path,
    spec: dict[str, Any],
    retrievers: list[dict[str, Any]],
    *,
    snapshot_id: str,
    status: str,
    report: dict[str, Any] | None = None,
    snapshot_sha256: str | None = None,
    repaired_tail: bool = False,
    retry_archive: Path | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "status": status,
        "updated_at": now,
        "path": str(path),
        "snapshot_sha256": snapshot_sha256,
        "dataset": spec["dataset"],
        "rag_backend": spec["rag_backend"],
        "retrievers": retrievers,
        "expected_rows": len(spec["dataset"]["questions"]) * len(retrievers),
        "validation": report,
        "truncated_tail_repaired": repaired_tail,
        "latest_retry_archive": str(retry_archive) if retry_archive else None,
    }


def _retrieve_live(
    config: RetrieverConfig,
    retriever: Retriever | None,
    item: QAItem,
    build_error: str | None,
) -> RetrievalResult:
    if build_error or retriever is None:
        error = build_error or "retriever_build_failed"
        return RetrievalResult(
            adapter=config.name,
            query=item.question,
            evidence=[],
            debug={"retriever_build_error": error},
            error=error,
        )
    try:
        return retriever.retrieve(item)
    except Exception as exc:
        error = f"retriever_failed: {type(exc).__name__}: {exc}"
        return RetrievalResult(
            adapter=config.name,
            query=item.question,
            evidence=[],
            debug={"retriever_error": error},
            error=error,
        )


def _evidence_record(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "kind": evidence.kind,
        "score": evidence.score,
        "metadata": _json_safe(evidence.metadata),
        "text": evidence.text,
    }


def _evidence_from_record(record: dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=str(record.get("evidence_id") or "snapshot-evidence"),
        kind=str(record.get("kind") or "tool_trace"),  # type: ignore[arg-type]
        text=str(record.get("text") or ""),
        metadata=dict(record.get("metadata") or {}),
        score=float(record.get("score") or 0.0),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("qa_id") or ""), str(row.get("retriever") or "")


def _row_sha256(row: dict[str, Any]) -> str:
    return _digest({key: value for key, value in row.items() if key != "row_sha256"})


def _key_record(key: tuple[str, str]) -> dict[str, str]:
    return {"qa_id": key[0], "retriever": key[1]}


def _read_jsonl_lenient(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    if not path.is_file():
        return [], []
    rows = []
    invalid = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(line_number)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                invalid.append(line_number)
    return rows, invalid


def _repair_truncated_tail(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
        handle.seek(0)
        payload = handle.read()
        line_start = payload.rfind(b"\n") + 1
        tail = payload[line_start:]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            handle.seek(line_start)
            handle.truncate()
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _archive_retry_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    retry_dir = path.parent / f"{path.stem}.retry_history"
    retry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = retry_dir / f"retrieval-errors-{stamp}.jsonl"
    _write_jsonl_atomic(archive, rows)
    return archive


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"retrieval snapshot manifest must contain an object: {path}")
    return payload


@contextmanager
def _snapshot_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(".lock")
    owner_pid = os.getpid()
    descriptor = _acquire_lock(lock_path, owner_pid)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            owner = {}
        if owner.get("pid") == owner_pid:
            lock_path.unlink(missing_ok=True)


def _acquire_lock(lock_path: Path, owner_pid: int) -> int:
    for _ in range(3):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pid, age_s = _lock_owner(lock_path)
            if pid is None and age_s < 10:
                raise RuntimeError(f"retrieval snapshot is already starting; lock exists at {lock_path}")
            if pid is not None and _process_is_alive(pid):
                raise RuntimeError(
                    f"retrieval snapshot is already active in process {pid}; lock exists at {lock_path}"
                )
            lock_path.unlink(missing_ok=True)
            continue
        payload = json.dumps({"pid": owner_pid, "started_at": datetime.now(UTC).isoformat()}) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
        return descriptor
    raise RuntimeError(f"could not acquire retrieval snapshot lock at {lock_path}")


def _lock_owner(lock_path: Path) -> tuple[int | None, float]:
    try:
        age_s = max(0.0, time.time() - lock_path.stat().st_mtime)
    except FileNotFoundError:
        return None, 0.0
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid"))
        return (pid if pid > 0 else None), age_s
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None, age_s


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _retriever_implementation_files(
    config: RetrieverConfig,
    *,
    root: Path,
) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    command_parts = []
    for key in ("command", "check_command"):
        value = config.options.get(key)
        if isinstance(value, str):
            command_parts.extend(value.split())
        elif isinstance(value, (list, tuple)):
            command_parts.extend(str(part) for part in value)
    for part in command_parts:
        if part.endswith(".py"):
            candidates.append(_resolve(root, part))
    command_text = " ".join(command_parts).lower()
    if "query_graphrag_index.py" in command_text:
        candidates.append(root / "data/working/graphrag_index/.gems_rag_graphrag_index.json")
    if "query_paperqa_index.py" in command_text:
        candidates.append(root / "data/working/paperqa_index/docs.pkl.gems_rag_ready.json")
    unique = {str(path.resolve()): path.resolve() for path in candidates}
    return [_file_identity(unique[key]) for key in sorted(unique)]


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "exists": False, "bytes": None, "sha256": None}
    if not resolved.is_file():
        return {"path": str(resolved), "exists": True, "bytes": None, "sha256": None}
    return {
        "path": str(resolved),
        "exists": True,
        "bytes": stat.st_size,
        "sha256": _sha256_cached(str(resolved), stat.st_size, stat.st_mtime_ns),
    }


@lru_cache(maxsize=256)
def _sha256_cached(path: str, _size: int, _mtime_ns: int) -> str:
    return _sha256(Path(path))


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = path if isinstance(path, Path) else Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root.resolve() / candidate).resolve()


def _digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
