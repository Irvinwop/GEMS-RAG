from __future__ import annotations

import binascii
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ARCHIVE_RE = re.compile(r"^MRAG-.*\.zip$")
POINTER_RE = re.compile(r"^(?:\.\./)+blobs/(?P<blob>[0-9a-f]{40,64}(?:\.[A-Za-z0-9]+)?)$")


def import_mrag_bundle(
    raw_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
    restore_qdrant: bool = True,
    verify_detached_blobs: bool = True,
    fallback_hf_caches: Sequence[Path] = (),
) -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    archives = sorted(path for path in raw_dir.iterdir() if path.is_file() and ARCHIVE_RE.match(path.name))
    if not archives:
        raise ValueError(f"no MRAG bundle archives found in {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_reports = []
    for archive in archives:
        archive_reports.append(_extract_zip(archive, output_dir, force=force))

    mrag_dir = output_dir / "MRAG"
    if not mrag_dir.is_dir():
        raise ValueError(f"bundle did not contain an MRAG directory: {output_dir}")

    link_report = restore_hf_cache_links(
        mrag_dir / "hf_cache",
        raw_dir,
        verify_detached_blobs=verify_detached_blobs,
        fallback_hf_caches=fallback_hf_caches,
    )
    qdrant_report: dict[str, Any] = {"restored": False, "reason": "disabled"}
    qdrant_tar = mrag_dir / "qdrant_db.tar"
    if restore_qdrant:
        if not qdrant_tar.is_file():
            raise FileNotFoundError(qdrant_tar)
        qdrant_report = _extract_tar(qdrant_tar, mrag_dir, force=force)

    report = {
        "schema_version": 1,
        "status": "complete",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "mrag_dir": str(mrag_dir),
        "archives": archive_reports,
        "hf_cache": link_report,
        "qdrant": qdrant_report,
        "artifacts": _artifact_summary(mrag_dir),
    }
    manifest_path = output_dir / "import_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["manifest"] = str(manifest_path)
    return report


def restore_hf_cache_links(
    hf_cache: Path,
    raw_dir: Path,
    *,
    verify_detached_blobs: bool = True,
    fallback_hf_caches: Sequence[Path] = (),
) -> dict[str, Any]:
    if not hf_cache.is_dir():
        return {
            "restored_links": 0,
            "detached_blobs": [],
            "renamed_cache_blobs": [],
            "fallback_cache_blobs": [],
            "missing_blobs": [],
        }
    raw_dir = raw_dir.resolve()
    detached_by_digest = _detached_blob_sources(raw_dir)
    fallback_by_digest = _cached_blob_sources(fallback_hf_caches)
    verified: set[str] = set()
    restored_links = 0
    detached_used: dict[str, str] = {}
    fallback_used: dict[str, str] = {}
    renamed_cache_blobs: dict[str, str] = {}
    missing: list[str] = []

    pointer_files = sorted(path for path in hf_cache.glob("models--*/snapshots/**/*") if path.is_file())
    for pointer_file in pointer_files:
        pointer = _read_pointer(pointer_file)
        if pointer is None:
            continue
        match = POINTER_RE.fullmatch(pointer)
        if match is None:
            continue
        model_root = next((parent for parent in pointer_file.parents if parent.name.startswith("models--")), None)
        if model_root is None:
            raise ValueError(f"could not identify model cache root for {pointer_file}")
        target = (model_root / "blobs" / match.group("blob")).resolve()
        normalized_pointer = os.path.relpath(target, pointer_file.parent)
        digest = match.group("blob").split(".", 1)[0]
        flattened_target = _is_flattened_pointer(target, digest)
        detached_source = detached_by_digest.get(digest)
        if (
            detached_source is not None
            and target.is_file()
            and not flattened_target
            and target.samefile(detached_source)
        ):
            if verify_detached_blobs and digest not in verified:
                actual = _blob_digest(detached_source, len(digest))
                if actual != digest:
                    raise ValueError(
                        f"recovered blob checksum mismatch: {detached_source}: {actual} != {digest}"
                    )
                verified.add(digest)
            detached_used[digest] = str(detached_source)
        if not target.is_file() or flattened_target:
            source = detached_source
            source_kind = "detached" if source is not None else None
            if source is None:
                renamed = sorted(
                    candidate
                    for candidate in target.parent.glob(f"{digest}.*")
                    if not _is_flattened_pointer(candidate, digest)
                )
                if len(renamed) == 1:
                    source = renamed[0]
                    source_kind = "renamed"
                    renamed_cache_blobs[digest] = str(source)
            if source is None:
                source = fallback_by_digest.get(digest)
                if source is not None:
                    source_kind = "fallback"
                    fallback_used[digest] = str(source)
            if source is None:
                missing.append(str(target))
                continue
            if verify_detached_blobs and digest not in verified:
                actual = _blob_digest(source, len(digest))
                if actual != digest:
                    raise ValueError(f"recovered blob checksum mismatch: {source}: {actual} != {digest}")
                verified.add(digest)
            target.parent.mkdir(parents=True, exist_ok=True)
            if flattened_target:
                target.unlink()
            _link_or_copy(source, target)
            if source_kind == "detached":
                detached_used[digest] = str(source)
        pointer_file.unlink()
        pointer_file.symlink_to(normalized_pointer)
        restored_links += 1

    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(f"missing {len(missing)} Hugging Face cache blobs: {sample}")
    broken = [str(path) for path in hf_cache.glob("models--*/snapshots/**/*") if path.is_symlink() and not path.exists()]
    if broken:
        raise FileNotFoundError(f"restored Hugging Face cache has broken links: {', '.join(broken[:5])}")
    return {
        "restored_links": restored_links,
        "detached_blobs": [
            {"sha256": digest, "source": detached_used[digest]}
            for digest in sorted(detached_used)
        ],
        "renamed_cache_blobs": [
            {"digest": digest, "source": renamed_cache_blobs[digest]}
            for digest in sorted(renamed_cache_blobs)
        ],
        "fallback_cache_blobs": [
            {"digest": digest, "source": fallback_used[digest]}
            for digest in sorted(fallback_used)
        ],
        "missing_blobs": [],
    }


def _extract_zip(archive: Path, output_dir: Path, *, force: bool) -> dict[str, Any]:
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"CRC failure in {archive}: {bad}")
        for info in handle.infolist():
            target = _safe_destination(output_dir, info.filename)
            if info.is_dir():
                if target.is_symlink():
                    raise ValueError(f"archive directory is an existing symlink: {info.filename}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not force and _zip_member_matches(target, info):
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".gems-rag-part")
            with handle.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(target)
            extracted += 1
    return {
        "path": str(archive),
        "sha256": _sha256(archive),
        "bytes": archive.stat().st_size,
        "extracted_files": extracted,
        "skipped_files": skipped,
    }


def _extract_tar(archive: Path, output_dir: Path, *, force: bool) -> dict[str, Any]:
    extracted = 0
    skipped = 0
    with tarfile.open(archive) as handle:
        for member in handle:
            target = _safe_destination(output_dir, member.name)
            if member.isdir():
                if target.is_symlink():
                    raise ValueError(f"archive directory is an existing symlink: {member.name}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported TAR member type: {member.name}")
            if not force and not target.is_symlink() and target.is_file() and target.stat().st_size == member.size:
                skipped += 1
                continue
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"could not read TAR member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".gems-rag-part")
            with source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(target)
            target.chmod(member.mode & 0o777)
            extracted += 1
    return {
        "restored": True,
        "archive": str(archive),
        "output": str(output_dir / "qdrant_db"),
        "extracted_files": extracted,
        "skipped_files": skipped,
    }


def _safe_destination(root: Path, member_name: str) -> Path:
    if not member_name or "\x00" in member_name:
        raise ValueError("archive member has an invalid name")
    root = root.resolve()
    target = Path(os.path.abspath(root / member_name))
    if not target.is_relative_to(root):
        raise ValueError(f"archive member escapes output directory: {member_name}")
    if not target.parent.resolve().is_relative_to(root):
        raise ValueError(f"archive member traverses a parent symlink: {member_name}")
    return target


def _zip_member_matches(path: Path, info: zipfile.ZipInfo) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            crc = binascii.crc32(block, crc)
    return (crc & 0xFFFFFFFF) == info.CRC


def _read_pointer(path: Path) -> str | None:
    try:
        if path.stat().st_size > 256:
            return None
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _is_flattened_pointer(path: Path, digest: str) -> bool:
    pointer = _read_pointer(path)
    if pointer is None:
        return False
    match = POINTER_RE.fullmatch(pointer)
    return match is not None and match.group("blob").split(".", 1)[0] == digest


def _detached_blob_sources(raw_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        prefix = path.name.split("-", 1)[0]
        if re.fullmatch(r"[0-9a-f]{64}", prefix):
            result[prefix] = path
    return result


def _cached_blob_sources(hf_caches: Sequence[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for cache in hf_caches:
        cache = cache.expanduser().resolve()
        if not cache.is_dir():
            raise FileNotFoundError(cache)
        for path in cache.glob("models--*/blobs/*"):
            digest = path.name.split(".", 1)[0]
            if not path.is_file() or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", digest) is None:
                continue
            if _is_flattened_pointer(path, digest):
                continue
            result.setdefault(digest, path)
    return result


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _artifact_summary(mrag_dir: Path) -> dict[str, Any]:
    cache = mrag_dir / "mmrag_cache_v3"
    return {
        "chunks": _line_count(cache / "chunks.jsonl"),
        "figures": _line_count(cache / "figures.jsonl"),
        "gold_qa": _line_count(mrag_dir / "eval" / "gold_qa.jsonl"),
        "page_images": _file_count(mrag_dir / "page_images", ".png"),
        "figure_images": _file_count(mrag_dir / "figures", ".png"),
        "graph_bytes": _file_size(cache / "graph.gpickle"),
        "qdrant_present": (mrag_dir / "qdrant_db").is_dir(),
    }


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_count(path: Path, suffix: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blob_digest(path: Path, digest_length: int) -> str:
    if digest_length == 64:
        return _sha256(path)
    if digest_length != 40:
        raise ValueError(f"unsupported Hugging Face blob digest length: {digest_length}")
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
