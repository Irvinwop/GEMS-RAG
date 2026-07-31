#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path


DEFAULT_RELEASE = Path(__file__).resolve().parents[1]
DEFAULT_MAX_ARCHIVE_BYTES = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the complete anonymous release as one standard ZIP file "
            "below the upload limit."
        )
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
        help="Hard byte limit for the completed ZIP file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing archive or abandoned temporary archive.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def release_files(release: Path) -> list[Path]:
    checksums_path = release / "CHECKSUMS.sha256"
    if not checksums_path.is_file():
        raise FileNotFoundError(checksums_path)
    files = [release / relative for relative in declared_checksums(checksums_path)]
    files.append(checksums_path)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release checksum files are missing: {missing[:10]}")
    return files


def declared_checksums(path: Path) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        expected, separator, relative = line.partition("  ")
        relative_path = Path(relative)
        if (
            not separator
            or len(expected) != 64
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative
        ):
            raise ValueError(f"invalid checksum row at line {line_number}")
        if relative in declared:
            raise ValueError(f"duplicate checksum path: {relative}")
        declared[relative] = expected
    if not declared:
        raise ValueError(f"checksum file is empty: {path}")
    return declared


def verify_release(release: Path, files: list[Path]) -> None:
    checksums_path = release / "CHECKSUMS.sha256"
    manifest_path = release / "RELEASE_MANIFEST.json"
    if not checksums_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"release is missing its integrity files: {release}"
        )

    declared = declared_checksums(checksums_path)

    actual = {
        path.relative_to(release).as_posix()
        for path in files
        if path != checksums_path
    }
    if set(declared) != actual:
        missing = sorted(actual - set(declared))
        extra = sorted(set(declared) - actual)
        raise ValueError(
            "release checksum inventory mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for relative, expected in declared.items():
        actual_digest = sha256(release / relative)
        if actual_digest != expected:
            raise ValueError(f"release checksum mismatch: {relative}")


def validate_archive(
    *,
    archive: Path,
    release: Path,
    files: list[Path],
    max_archive_bytes: int,
) -> None:
    size = archive.stat().st_size
    if size >= max_archive_bytes:
        raise ValueError(
            f"{archive.name} is {size:,} bytes; "
            f"must be below {max_archive_bytes:,}"
        )

    expected = {
        (Path(release.name) / path.relative_to(release)).as_posix()
        for path in files
    }
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            raise ValueError(
                f"{archive.name} contains a corrupt entry: {corrupt}"
            )
        observed = {
            name for name in bundle.namelist() if not name.endswith("/")
        }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "ZIP inventory mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def build_archive(
    *,
    release: Path,
    output: Path,
    max_archive_bytes: int,
    force: bool,
) -> dict[str, object]:
    release = release.expanduser().resolve()
    output = output.expanduser().resolve()
    if max_archive_bytes < 1:
        raise ValueError("max_archive_bytes must be positive")
    if release == output or release in output.parents:
        raise ValueError("archive output must be outside the release directory")
    if output.exists() and output.is_dir():
        raise IsADirectoryError(output)
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")

    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists() and not force:
        raise FileExistsError(
            f"{temporary} exists; pass --force to replace it"
        )
    temporary.unlink(missing_ok=True)

    files = release_files(release)
    verify_release(release, files)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
            strict_timestamps=False,
        ) as bundle:
            for path in files:
                relative = path.relative_to(release)
                bundle.write(
                    path,
                    arcname=(Path(release.name) / relative).as_posix(),
                )
        validate_archive(
            archive=temporary,
            release=release,
            files=files,
            max_archive_bytes=max_archive_bytes,
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "filename": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "files": len(files),
    }


def main() -> int:
    args = parse_args()
    release = args.release.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else release.with_name(f"{release.name}.zip")
    )
    record = build_archive(
        release=release,
        output=output,
        max_archive_bytes=args.max_archive_bytes,
        force=args.force,
    )
    print(f"Upload archive ready: {output}")
    print(f"Size: {format_size(int(record['bytes']))}")
    print(f"SHA-256: {record['sha256']}")
    print(f"Files: {record['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
