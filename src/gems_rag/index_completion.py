from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def file_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def value_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_completion_marker(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def completion_marker_matches(path: Path, identity: dict[str, Any]) -> bool:
    marker = read_completion_marker(path)
    return bool(
        marker
        and marker.get("schema_version") == SCHEMA_VERSION
        and marker.get("complete") is True
        and marker.get("identity") == identity
    )


def publish_completion_marker(
    path: Path,
    identity: dict[str, Any],
    **details: Any,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "identity": identity,
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload
