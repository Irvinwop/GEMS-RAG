#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "PREBUILT_INDEXES.json"
DEFAULT_CACHE = ROOT / ".downloads"
READY_PATH = ROOT / ".prebuilt_indexes_ready.json"
STAGING_PATH = ROOT / ".prebuilt-indexes.staging"
BACKUP_PATH = ROOT / ".prebuilt-indexes.backup"
CHUNK_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumably download, verify, and atomically install the supplied "
            "prebuilt comparison indexes."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--url",
        help="Override the release-asset URL recorded in the manifest.",
    )
    parser.add_argument(
        "--asset",
        type=Path,
        help="Use an already-downloaded index ZIP instead of the network.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even when the current indexes pass validation.",
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the verified ZIP after successful installation.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported prebuilt-index manifest: {path}")
    asset = manifest.get("asset")
    files = manifest.get("files")
    if not isinstance(asset, dict) or not isinstance(files, list) or not files:
        raise ValueError(f"incomplete prebuilt-index manifest: {path}")
    return manifest


def expected_files(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        relative = str(item.get("path") or "")
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "indexes"
        ):
            raise ValueError(f"unsafe index path in manifest: {relative!r}")
        if relative in result:
            raise ValueError(f"duplicate index path in manifest: {relative}")
        result[relative] = item
    return result


def file_matches(path: Path, identity: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(identity["bytes"])
        and sha256(path) == str(identity["sha256"])
    )


def indexes_match(root: Path, manifest: dict[str, Any]) -> bool:
    return all(
        file_matches(root.joinpath(*PurePosixPath(relative).parts), identity)
        for relative, identity in expected_files(manifest).items()
    )


def verified_asset(path: Path, asset: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(asset["bytes"])
        and sha256(path) == str(asset["sha256"])
    )


def download_asset(
    *,
    url: str,
    destination: Path,
    asset: dict[str, Any],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified_asset(destination, asset):
        print(f"Using verified download: {destination}", flush=True)
        return destination
    if destination.exists():
        destination.unlink()

    partial = destination.with_name(f"{destination.name}.part")
    expected_bytes = int(asset["bytes"])
    if partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mutcd-rag-index-initializer/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(
            f"Resuming {asset['name']} at {offset:,} / "
            f"{expected_bytes:,} bytes",
            flush=True,
        )
    else:
        print(f"Downloading {asset['name']} ({expected_bytes:,} bytes)", flush=True)

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset == expected_bytes:
            os.replace(partial, destination)
            if verified_asset(destination, asset):
                return destination
        raise

    status = getattr(response, "status", None)
    append = offset > 0 and status == 206
    if offset and not append:
        offset = 0
    mode = "ab" if append else "wb"
    downloaded = offset
    next_report = downloaded + max(expected_bytes // 20, CHUNK_BYTES)
    with response, partial.open(mode) as handle:
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            handle.write(block)
            downloaded += len(block)
            if downloaded >= next_report or downloaded == expected_bytes:
                print(
                    f"  downloaded {downloaded:,} / {expected_bytes:,} bytes",
                    flush=True,
                )
                next_report = downloaded + max(
                    expected_bytes // 20,
                    CHUNK_BYTES,
                )
        handle.flush()
        os.fsync(handle.fileno())

    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"incomplete download: {partial.stat().st_size:,} of "
            f"{expected_bytes:,} bytes; rerun to resume"
        )
    actual = sha256(partial)
    if actual != asset["sha256"]:
        partial.unlink()
        raise ValueError(
            f"download checksum mismatch: expected {asset['sha256']}, "
            f"found {actual}"
        )
    os.replace(partial, destination)
    return destination


def safe_extract(
    archive: Path,
    staging: Path,
    manifest: dict[str, Any],
) -> None:
    expected = expected_files(manifest)
    expected_archive_entries = set(expected) | {
        "PREBUILT_INDEXES_MANIFEST.json"
    }
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    observed: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as bundle:
            corrupt = bundle.testzip()
            if corrupt:
                raise ValueError(f"corrupt prebuilt-index entry: {corrupt}")
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or stat.S_ISLNK(mode)
                    or member.filename not in expected_archive_entries
                ):
                    raise ValueError(
                        f"unsafe or unexpected index entry: {member.filename}"
                    )
                observed.add(member.filename)
                if member.filename == "PREBUILT_INDEXES_MANIFEST.json":
                    embedded = json.loads(bundle.read(member))
                    if embedded.get("files") != manifest.get("files"):
                        raise ValueError(
                            "embedded prebuilt-index manifest does not match"
                        )
                    continue
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
        if observed != expected_archive_entries:
            missing = sorted(expected_archive_entries - observed)
            extra = sorted(observed - expected_archive_entries)
            raise ValueError(
                "prebuilt-index ZIP inventory mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        if not indexes_match(staging, manifest):
            raise ValueError("extracted prebuilt indexes failed validation")
    except Exception:
        print(
            f"Extraction failed; staging retained for inspection: {staging}",
            file=sys.stderr,
        )
        raise


def ready_record(manifest: dict[str, Any]) -> dict[str, Any]:
    asset = manifest["asset"]
    return {
        "schema_version": 1,
        "asset": {
            "name": asset["name"],
            "bytes": asset["bytes"],
            "sha256": asset["sha256"],
        },
        "files": len(manifest["files"]),
        "ready": True,
    }


def recover_interrupted_install(manifest: dict[str, Any]) -> None:
    target = ROOT / "indexes"
    if not BACKUP_PATH.exists():
        return
    if target.exists() and indexes_match(ROOT, manifest):
        atomic_write_json(READY_PATH, ready_record(manifest))
        shutil.rmtree(BACKUP_PATH)
        return
    if target.exists():
        shutil.rmtree(target)
    os.replace(BACKUP_PATH, target)


def install_indexes(staging: Path, manifest: dict[str, Any]) -> None:
    target = ROOT / "indexes"
    staged_indexes = staging / "indexes"
    if not staged_indexes.is_dir():
        raise FileNotFoundError(staged_indexes)
    shutil.rmtree(BACKUP_PATH, ignore_errors=True)
    if target.exists():
        os.replace(target, BACKUP_PATH)
    try:
        os.replace(staged_indexes, target)
        if not indexes_match(ROOT, manifest):
            raise ValueError("installed prebuilt indexes failed validation")
        atomic_write_json(READY_PATH, ready_record(manifest))
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if BACKUP_PATH.exists():
            os.replace(BACKUP_PATH, target)
        raise
    shutil.rmtree(BACKUP_PATH, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    recover_interrupted_install(manifest)
    if indexes_match(ROOT, manifest) and not args.force:
        atomic_write_json(READY_PATH, ready_record(manifest))
        print("Prebuilt indexes are already initialized.", flush=True)
        return 0

    asset = manifest["asset"]
    if args.asset:
        archive = args.asset.expanduser().resolve()
        if not verified_asset(archive, asset):
            raise ValueError(f"local prebuilt-index asset failed validation: {archive}")
    else:
        url = args.url or str(asset["url"])
        archive = download_asset(
            url=url,
            destination=args.cache.expanduser().resolve() / str(asset["name"]),
            asset=asset,
        )

    safe_extract(archive, STAGING_PATH, manifest)
    install_indexes(STAGING_PATH, manifest)
    if not args.keep_download and not args.asset:
        archive.unlink(missing_ok=True)
    print(
        f"Prebuilt indexes initialized: {len(manifest['files'])} files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
