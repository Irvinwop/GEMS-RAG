from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .config import ExperimentConfig, ModelConfig, experiment_config_to_dict
from .data import load_qa_items
from .models import ModelClient
from .prompts import build_injected_prompt
from .retrieval_snapshots import load_retrieval_snapshot, retrieval_snapshot_status
from .runner import _acquire_run_lock, run_experiment
from .types import ModelResult


ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
BATCH_STATE_SCHEMA_VERSION = 1


class BatchTransport(Protocol):
    def create(self, requests: list[dict[str, Any]]) -> dict[str, Any]: ...

    def create_message(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def retrieve(self, batch_id: str) -> dict[str, Any]: ...

    def results(self, batch_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class BatchTask:
    custom_id: str
    key: tuple[str, str, str, str, str]
    prompt: str
    request: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "key": list(self.key),
            "prompt_sha256": _digest_text(self.prompt),
            "prompt": self.prompt,
            "request": self.request,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BatchTask":
        key = record.get("key")
        if not isinstance(key, list) or len(key) != 5:
            raise ValueError("Anthropic batch request manifest has an invalid row key")
        prompt = str(record.get("prompt") or "")
        if record.get("prompt_sha256") != _digest_text(prompt):
            raise ValueError("Anthropic batch request manifest prompt hash mismatch")
        request = record.get("request")
        if not isinstance(request, dict):
            raise ValueError("Anthropic batch request manifest has an invalid request")
        return cls(
            custom_id=str(record.get("custom_id") or ""),
            key=tuple(str(value) for value in key),
            prompt=prompt,
            request=request,
        )


class AnthropicBatchTransport:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = ANTHROPIC_API_BASE,
        timeout_s: float = 180.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def create(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        return self._json_request("POST", "/v1/messages/batches", {"requests": requests})

    def create_message(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._json_request("POST", "/v1/messages", params)

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return self._json_request("GET", f"/v1/messages/batches/{batch_id}")

    def results(self, batch_id: str) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/v1/messages/batches/{batch_id}/results")
        rows = []
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid Anthropic batch result JSON on line {line_number}: {exc}") from exc
        return rows

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self._request(method, path, payload)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Anthropic batch API returned invalid JSON for {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Anthropic batch API returned a non-object response for {path}")
        return value

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
                "user-agent": "gems-rag/0.1",
                "x-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            raise RuntimeError(f"Anthropic batch API HTTP {exc.code} for {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic batch API request failed for {path}: {exc.reason}") from exc


class _BatchReplayModel(ModelClient):
    def __init__(self, config: ModelConfig, responses: dict[str, deque[ModelResult]]) -> None:
        self.config = config
        self.responses = responses

    def generate(self, prompt: str) -> ModelResult:
        prompt_hash = _digest_text(prompt)
        queue = self.responses.get(prompt_hash)
        if not queue:
            raise RuntimeError(f"no Anthropic batch response for prompt {prompt_hash}")
        return queue.popleft()


def run_anthropic_batch(
    config: ExperimentConfig,
    *,
    poll_interval_s: float = 30.0,
    wait: bool = True,
    transport: BatchTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    _validate_batch_config(config)
    output_dir = config.output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with _batch_lock(output_dir):
        return _run_anthropic_batch_locked(
            config,
            output_dir=output_dir,
            poll_interval_s=max(0.0, poll_interval_s),
            wait=wait,
            transport=transport,
            sleep=sleep,
        )


def retry_anthropic_batch_failure(
    config: ExperimentConfig,
    custom_id: str,
    *,
    transport: BatchTransport | None = None,
) -> dict[str, Any]:
    """Retry one failed batch request through the synchronous Messages API."""
    _validate_batch_config(config)
    output_dir = config.output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with _batch_lock(output_dir):
        state_path = output_dir / "anthropic_batch_state.json"
        requests_path = output_dir / "anthropic_batch_requests.jsonl"
        results_path = output_dir / "anthropic_batch_results.jsonl"
        retries_path = output_dir / "anthropic_batch_retries.jsonl"
        state = _load_json(state_path)
        if not state or not state.get("batch_id"):
            raise ValueError("Anthropic batch must be submitted before retrying a failed request")
        if state.get("config_sha256") != _digest_json(experiment_config_to_dict(config)):
            raise ValueError("existing Anthropic batch state belongs to a different experiment config")
        tasks = _load_tasks(requests_path)
        task_by_id = {task.custom_id: task for task in tasks}
        task = task_by_id.get(custom_id)
        if task is None:
            raise ValueError(f"unknown Anthropic batch custom_id: {custom_id}")

        if transport is None:
            api_key_env = _batch_api_key_env(config)
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"missing API key env var: {api_key_env}")
            transport = AnthropicBatchTransport(api_key)

        batch_id = str(state["batch_id"])
        if not results_path.is_file():
            batch = transport.retrieve(batch_id)
            if str(batch.get("processing_status") or "") != "ended":
                raise ValueError("Anthropic batch has not ended; failed requests cannot be retried yet")
            _write_jsonl_atomic(results_path, transport.results(batch_id))
        result_rows = _load_jsonl(results_path)
        indexed = _index_result_rows(tasks, result_rows)
        original = indexed[custom_id]
        retry_reason = _result_retry_reason(original)
        if retry_reason is None:
            raise ValueError(f"Anthropic batch request {custom_id} is operationally valid and must not be retried")

        retry_rows = _load_jsonl(retries_path)
        existing = [row for row in retry_rows if row.get("custom_id") == custom_id]
        if len(existing) > 1:
            raise ValueError(f"multiple synchronous retries are recorded for {custom_id}")
        if existing:
            _prepare_canonical_retry(output_dir, task, existing[0])
            return {
                "status": "already_retried",
                "batch_id": batch_id,
                "custom_id": custom_id,
                "message_id": (existing[0].get("message") or {}).get("id"),
            }

        message = transport.create_message(task.request["params"])
        if message.get("type") != "message" or not message.get("id"):
            raise RuntimeError("Anthropic synchronous retry returned an invalid message response")
        retry_row = {
            "schema_version": 1,
            "custom_id": custom_id,
            "key": list(task.key),
            "batch_id": batch_id,
            "retried_at": datetime.now(UTC).isoformat(),
            "request_sha256": _digest_json(task.request["params"]),
            "retry_reason": retry_reason,
            "original_result": original.get("result"),
            "message": message,
        }
        _write_jsonl_atomic(retries_path, [*retry_rows, retry_row])
        _prepare_canonical_retry(output_dir, task, retry_row)
        state.update(
            {
                "status": "retry_pending",
                "retry_count": len(retry_rows) + 1,
                "last_retry_at": retry_row["retried_at"],
            }
        )
        _write_json_atomic(state_path, state)
        return {
            "status": "retried",
            "batch_id": batch_id,
            "custom_id": custom_id,
            "message_id": message["id"],
        }


def _run_anthropic_batch_locked(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    poll_interval_s: float,
    wait: bool,
    transport: BatchTransport | None,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    runs_path = output_dir / "runs.jsonl"
    state_path = output_dir / "anthropic_batch_state.json"
    requests_path = output_dir / "anthropic_batch_requests.jsonl"
    results_path = output_dir / "anthropic_batch_results.jsonl"
    retries_path = output_dir / "anthropic_batch_retries.jsonl"
    config_sha256 = _digest_json(experiment_config_to_dict(config))
    completed = _completed_keys(runs_path)
    expected = _expected_keys(config)
    pending = expected - completed

    state = _load_json(state_path)
    if not pending:
        report = _summary(config, state, expected_rows=len(expected), completed_rows=len(completed))
        report["status"] = "complete"
        return report

    if state:
        if state.get("config_sha256") != config_sha256:
            raise ValueError("existing Anthropic batch state belongs to a different experiment config")
        tasks = _load_tasks(requests_path)
        unknown_pending = pending - {task.key for task in tasks}
        if unknown_pending:
            raise ValueError(f"existing Anthropic batch does not cover {len(unknown_pending)} pending rows")
    else:
        tasks = _prepare_tasks(config, completed)
        _write_jsonl_atomic(requests_path, [task.to_record() for task in tasks])
        state = {
            "schema_version": BATCH_STATE_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "requests_sha256": _sha256_file(requests_path),
            "request_count": len(tasks),
            "status": "prepared",
            "prepared_at": datetime.now(UTC).isoformat(),
            "batch_id": None,
        }
        _write_json_atomic(state_path, state)

    if state.get("requests_sha256") != _sha256_file(requests_path):
        raise ValueError("Anthropic batch request manifest hash mismatch")

    if transport is None:
        api_key_env = _batch_api_key_env(config)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"missing API key env var: {api_key_env}")
        transport = AnthropicBatchTransport(api_key)

    batch_id = state.get("batch_id")
    if not batch_id:
        batch = transport.create([task.request for task in tasks])
        batch_id = str(batch.get("id") or "")
        if not batch_id:
            raise RuntimeError("Anthropic batch creation response did not include an id")
        state.update(
            {
                "batch_id": batch_id,
                "status": str(batch.get("processing_status") or "submitted"),
                "submitted_at": datetime.now(UTC).isoformat(),
                "request_counts": batch.get("request_counts"),
            }
        )
        _write_json_atomic(state_path, state)
    else:
        batch = transport.retrieve(str(batch_id))

    while str(batch.get("processing_status") or "") != "ended":
        state.update(
            {
                "status": str(batch.get("processing_status") or "in_progress"),
                "last_polled_at": datetime.now(UTC).isoformat(),
                "request_counts": batch.get("request_counts"),
            }
        )
        _write_json_atomic(state_path, state)
        if not wait:
            return _summary(config, state, expected_rows=len(expected), completed_rows=len(completed))
        sleep(poll_interval_s)
        batch = transport.retrieve(str(batch_id))

    state.update(
        {
            "status": "ended",
            "ended_at": _json_value(batch.get("ended_at")) or datetime.now(UTC).isoformat(),
            "last_polled_at": datetime.now(UTC).isoformat(),
            "request_counts": batch.get("request_counts"),
        }
    )
    _write_json_atomic(state_path, state)

    if not results_path.is_file():
        result_rows = transport.results(str(batch_id))
        _write_jsonl_atomic(results_path, result_rows)
    result_rows = _load_jsonl(results_path)
    retry_rows = _load_jsonl(retries_path)
    results = _validated_results(
        tasks,
        result_rows,
        retry_rows=retry_rows,
        batch_id=str(batch_id),
    )
    current_completed = _completed_keys(runs_path)
    replay_tasks = [task for task in tasks if task.key not in current_completed]
    clients = _replay_clients(config, replay_tasks, results, str(batch_id))
    run_experiment(
        config,
        resume=True,
        model_client_factory=lambda model: clients[(model.provider, model.model)],
    )

    completed = _completed_keys(runs_path)
    remaining = expected - completed
    if remaining:
        raise RuntimeError(f"Anthropic batch replay left {len(remaining)} experiment rows incomplete")
    state.update(
        {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "results_sha256": _sha256_file(results_path),
            "retry_count": len(retry_rows),
            "retries_sha256": _sha256_file(retries_path) if retry_rows else None,
        }
    )
    _write_json_atomic(state_path, state)
    report = _summary(config, state, expected_rows=len(expected), completed_rows=len(completed))
    report["usage"] = _batch_usage(list(results.values()))
    return report


def _validate_batch_config(config: ExperimentConfig) -> None:
    if config.dry_run:
        raise ValueError("Anthropic batches cannot run a dry-run config")
    if config.retrieval_snapshot is None:
        raise ValueError("Anthropic batches require a frozen retrieval snapshot")
    if set(config.context_modes) != {"injected"}:
        raise ValueError("Anthropic batches currently support injected context only")
    if not config.models or any(model.provider != "anthropic" for model in config.models):
        raise ValueError("Anthropic batches require one or more Anthropic answer models")
    if any(bool(model.options.get("vision", False)) for model in config.models):
        raise ValueError("Anthropic batch replay currently requires text-only model configs")
    report = retrieval_snapshot_status(config)
    if not report["ok"]:
        raise ValueError("retrieval snapshot is not ready: " + "; ".join(report["problems"]))
    _batch_api_key_env(config)


def _prepare_tasks(
    config: ExperimentConfig,
    completed: set[tuple[str, str, str, str, str]],
) -> list[BatchTask]:
    items = load_qa_items(config.dataset.qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids)
    snapshot = load_retrieval_snapshot(config)
    tasks = []
    sequence = 0
    for item in items:
        for retriever_config in config.retrievers:
            retrieval = snapshot.retriever(retriever_config).retrieve(item)
            if retrieval.error:
                raise ValueError(
                    f"snapshot retrieval failed for {item.qa_id}/{retriever_config.name}: {retrieval.error}"
                )
            for context_mode in config.context_modes:
                prompt = build_injected_prompt(item, retrieval.evidence, config.max_evidence_chars)
                for model in config.models:
                    key = (item.qa_id, retriever_config.name, context_mode, model.provider, model.model)
                    if key in completed:
                        continue
                    custom_id = f"row-{sequence:06d}-{_digest_text('|'.join(key))[:12]}"
                    request = {
                        "custom_id": custom_id,
                        "params": _message_params(model, prompt),
                    }
                    tasks.append(BatchTask(custom_id, key, prompt, request))
                    sequence += 1
    return tasks


def _message_params(model: ModelConfig, prompt: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model.model,
        "max_tokens": int(model.options.get("max_tokens", 900)),
        "messages": [{"role": "user", "content": prompt}],
    }
    if "temperature" in model.options and model.options["temperature"] is not None:
        params["temperature"] = float(model.options["temperature"])
    thinking = model.options.get("thinking")
    if thinking is not None:
        params["thinking"] = (
            {"type": thinking.strip().lower()}
            if isinstance(thinking, str)
            else thinking
        )
    return params


def _validated_results(
    tasks: list[BatchTask],
    rows: list[dict[str, Any]],
    *,
    retry_rows: list[dict[str, Any]] | None = None,
    batch_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    results = _index_result_rows(tasks, rows)
    expected = {task.custom_id for task in tasks}
    task_by_id = {task.custom_id: task for task in tasks}
    failures = {
        custom_id: retry_reason
        for custom_id, row in results.items()
        if custom_id in expected and (retry_reason := _result_retry_reason(row)) is not None
    }
    retries: dict[str, dict[str, Any]] = {}
    retry_duplicates = set()
    for retry in retry_rows or []:
        custom_id = str(retry.get("custom_id") or "")
        if custom_id in retries:
            retry_duplicates.add(custom_id)
        retries[custom_id] = retry
    invalid_retry_ids = set(retries) - set(failures)
    invalid_retries = set()
    for custom_id, retry in retries.items():
        task = task_by_id.get(custom_id)
        if task is None:
            continue
        message = retry.get("message")
        if not isinstance(message, dict) or not message.get("id"):
            invalid_retries.add(custom_id)
            continue
        if retry.get("key") != list(task.key):
            invalid_retries.add(custom_id)
        if retry.get("request_sha256") != _digest_json(task.request["params"]):
            invalid_retries.add(custom_id)
        if retry.get("retry_reason") != failures.get(custom_id):
            invalid_retries.add(custom_id)
        if retry.get("original_result") != results[custom_id].get("result"):
            invalid_retries.add(custom_id)
        if batch_id is not None and retry.get("batch_id") != batch_id:
            invalid_retries.add(custom_id)
        if message.get("model") != task.key[4]:
            invalid_retries.add(custom_id)
        if _message_retry_reason(message) is not None:
            invalid_retries.add(custom_id)
    if retry_duplicates or invalid_retry_ids or invalid_retries:
        raise RuntimeError(
            "Anthropic batch retry results failed validation: "
            f"duplicates={len(retry_duplicates)}, invalid_ids={sorted(invalid_retry_ids)}, "
            f"invalid_messages={sorted(invalid_retries)}"
        )
    unresolved = set(failures) - set(retries)
    if unresolved:
        raise RuntimeError(
            "Anthropic batch results failed validation: "
            f"failures={{{', '.join(f'{key!r}: {failures[key]!r}' for key in sorted(unresolved))}}}"
        )
    for custom_id, retry in retries.items():
        original = results[custom_id]
        results[custom_id] = {
            "custom_id": custom_id,
            "result": {"type": "succeeded", "message": retry["message"]},
            "recovery": {
                "type": "anthropic_sync_retry",
                "retried_at": retry.get("retried_at"),
                "request_sha256": retry.get("request_sha256"),
                "retry_reason": retry.get("retry_reason"),
                "original_result": original.get("result"),
            },
        }
    return results


def _index_result_rows(
    tasks: list[BatchTask],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = {task.custom_id for task in tasks}
    results: dict[str, dict[str, Any]] = {}
    duplicates = set()
    for row in rows:
        custom_id = str(row.get("custom_id") or "")
        if custom_id in results:
            duplicates.add(custom_id)
        results[custom_id] = row
    missing = expected - set(results)
    extra = set(results) - expected
    if duplicates or missing or extra:
        raise RuntimeError(
            "Anthropic batch results failed validation: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, extra={len(extra)}"
        )
    return results


def _result_retry_reason(row: dict[str, Any]) -> str | None:
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    result_type = str(result.get("type") or "missing")
    if result_type != "succeeded":
        return f"result:{result_type}"
    message = result.get("message") if isinstance(result.get("message"), dict) else {}
    return _message_retry_reason(message)


def _message_retry_reason(message: dict[str, Any]) -> str | None:
    blocks = message.get("content") if isinstance(message.get("content"), list) else []
    output = "".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not output.strip():
        return "empty_response"
    stop_reason = str(message.get("stop_reason") or "")
    if stop_reason == "max_tokens":
        return "max_tokens"
    return None


def _prepare_canonical_retry(
    output_dir: Path,
    task: BatchTask,
    retry: dict[str, Any],
) -> None:
    runs_path = output_dir / "runs.jsonl"
    if not runs_path.is_file():
        return
    rows = _load_jsonl(runs_path)
    matching = [row for row in rows if _run_row_key(row) == task.key]
    if len(matching) > 1:
        raise ValueError(f"multiple canonical run rows match retry {task.custom_id}")
    if not matching:
        return
    current = matching[0]
    raw = current.get("model_raw") if isinstance(current.get("model_raw"), dict) else {}
    message = retry.get("message") if isinstance(retry.get("message"), dict) else {}
    if raw.get("api") == "anthropic_sync_retry" and raw.get("id") == message.get("id"):
        return
    if str(current.get("answer") or "").strip():
        raise ValueError(f"refusing to replace non-empty canonical answer for {task.custom_id}")

    archive_path = output_dir / "retry_history" / f"anthropic_batch_{task.custom_id}.jsonl"
    if archive_path.is_file():
        archived = _load_jsonl(archive_path)
        if archived != matching:
            raise ValueError(f"retry archive does not match canonical row for {task.custom_id}")
    else:
        _write_jsonl_atomic(archive_path, matching)
    _write_jsonl_atomic(runs_path, [row for row in rows if _run_row_key(row) != task.key])


def _replay_clients(
    config: ExperimentConfig,
    tasks: list[BatchTask],
    results: dict[str, dict[str, Any]],
    batch_id: str,
) -> dict[tuple[str, str], _BatchReplayModel]:
    queues: dict[tuple[str, str], dict[str, deque[ModelResult]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for task in tasks:
        provider, model = task.key[3], task.key[4]
        queues[(provider, model)][_digest_text(task.prompt)].append(
            _model_result(task, results[task.custom_id], batch_id)
        )
    return {
        (model.provider, model.model): _BatchReplayModel(
            model,
            dict(queues[(model.provider, model.model)]),
        )
        for model in config.models
    }


def _model_result(task: BatchTask, row: dict[str, Any], batch_id: str) -> ModelResult:
    message = row["result"]["message"]
    blocks = message.get("content") if isinstance(message.get("content"), list) else []
    output = "".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    provider_finish_reason = str(message.get("stop_reason") or "")
    finish_reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
    }.get(provider_finish_reason, provider_finish_reason or None)
    recovery = row.get("recovery") if isinstance(row.get("recovery"), dict) else None
    raw = {
        "id": message.get("id"),
        "api": "anthropic_sync_retry" if recovery else "anthropic_batch",
        "batch_id": batch_id,
        "batch_custom_id": task.custom_id,
        "finish_reason": finish_reason,
        "provider_finish_reason": provider_finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if recovery:
        raw["recovery"] = recovery
    return ModelResult(
        provider=task.key[3],
        model=task.key[4],
        output=output,
        raw=raw,
    )


def _expected_keys(config: ExperimentConfig) -> set[tuple[str, str, str, str, str]]:
    items = load_qa_items(config.dataset.qa_path, limit=config.dataset.limit, qa_ids=config.dataset.qa_ids)
    return {
        (item.qa_id, retriever.name, mode, model.provider, model.model)
        for item in items
        for retriever in config.retrievers
        for mode in config.context_modes
        for model in config.models
    }


def _completed_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    return {_run_row_key(row) for row in _load_jsonl(path)}


def _run_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    config = row.get("config") if isinstance(row.get("config"), dict) else {}
    return (
        str(row.get("qa_id") or ""),
        str(config.get("retriever") or ""),
        str(config.get("context_mode") or ""),
        str(config.get("model_provider") or ""),
        str(config.get("model") or ""),
    )


def _batch_api_key_env(config: ExperimentConfig) -> str:
    envs = {str(model.options.get("api_key_env") or "ANTHROPIC_API_KEY") for model in config.models}
    if len(envs) != 1:
        raise ValueError("all models in one Anthropic batch must use the same API key environment variable")
    return next(iter(envs))


def _summary(
    config: ExperimentConfig,
    state: dict[str, Any] | None,
    *,
    expected_rows: int,
    completed_rows: int,
) -> dict[str, Any]:
    state = state or {}
    return {
        "experiment": config.name,
        "status": state.get("status") or "ready",
        "batch_id": state.get("batch_id"),
        "request_count": state.get("request_count", 0),
        "request_counts": state.get("request_counts"),
        "expected_rows": expected_rows,
        "completed_rows": completed_rows,
        "remaining_rows": max(0, expected_rows - completed_rows),
        "retry_count": int(state.get("retry_count") or 0),
    }


def _batch_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        message = result.get("message") if isinstance(result.get("message"), dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _load_tasks(path: Path) -> list[BatchTask]:
    return [BatchTask.from_record(record) for record in _load_jsonl(path)]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object in {path} line {line_number}")
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _write_bytes_atomic(path, payload.encode("utf-8"))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _digest_text(payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


@contextmanager
def _batch_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".anthropic-batch.lock"
    owner_pid = os.getpid()
    descriptor = _acquire_run_lock(lock_path, owner_pid)
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
