#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RELEASE = Path(__file__).resolve().parents[1]
DEFAULT_MAX_PART_BYTES = 500_000_000
UPLOAD_README_NAME = "UPLOAD_README.md"
VECTOR_FILES = {
    Path(
        "indexes/gems-rag/qdrant_db/collection/"
        "mutcd_pages/storage.sqlite"
    ),
    Path(
        "indexes/gems-rag/qdrant_db/collection/"
        "mutcd_figures_visual/storage.sqlite"
    ),
}


@dataclass(frozen=True)
class Part:
    number: int
    group: str
    label: str
    compression: int
    compresslevel: int | None

    @property
    def filename(self) -> str:
        return (
            "mutcd-rag-anonymous-release-"
            f"part-{self.number:02d}-{self.label}.zip"
        )


PARTS = (
    Part(1, "core", "core", zipfile.ZIP_DEFLATED, 6),
    Part(2, "vectors", "qdrant-vectors", zipfile.ZIP_DEFLATED, 1),
    Part(3, "page_images", "page-images", zipfile.ZIP_STORED, None),
    Part(4, "figures", "figures", zipfile.ZIP_STORED, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the anonymous release as independent ZIP files below "
            "a per-file upload limit."
        )
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-part-bytes",
        type=int,
        default=DEFAULT_MAX_PART_BYTES,
        help="Hard byte limit for every ZIP file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing upload-parts directory.",
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


def group_for(relative: Path) -> str:
    if relative in VECTOR_FILES:
        return "vectors"
    parts = relative.parts
    if parts[:3] == ("indexes", "gems-rag", "page_images"):
        return "page_images"
    if parts[:3] == ("indexes", "gems-rag", "figures"):
        return "figures"
    return "core"


def release_files(release: Path) -> list[Path]:
    files = sorted(path for path in release.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"release contains no files: {release}")
    return files


def upload_readme() -> str:
    names = "\n".join(f"- `{part.filename}`" for part in PARTS)
    return f"""# MUTCD RAG anonymous upload set

All four ZIP files are required. Each is an independent, standard ZIP file
below the 512 MB upload limit:

{names}

Extract every part into the same empty directory, in numeric order:

```bash
mkdir assembled
for part in mutcd-rag-anonymous-release-part-*.zip; do
  unzip -q "$part" -d assembled
done
cd assembled/mutcd-rag-anonymous-release
shasum -a 256 -c CHECKSUMS.sha256
```

The parts contain disjoint files, so extraction does not overwrite release
content. The first part also carries this instruction file outside the release
folder. `UPLOAD_PARTS.json` records each ZIP's size and SHA-256 digest.
"""


def write_part(
    *,
    release: Path,
    output: Path,
    part: Part,
    files: Iterable[Path],
    instructions: str,
) -> dict[str, object]:
    archive = output / part.filename
    kwargs: dict[str, object] = {
        "mode": "w",
        "compression": part.compression,
        "allowZip64": True,
    }
    if part.compresslevel is not None:
        kwargs["compresslevel"] = part.compresslevel

    file_list = list(files)
    with zipfile.ZipFile(archive, **kwargs) as bundle:
        if part.number == 1:
            bundle.writestr(UPLOAD_README_NAME, instructions)
        for path in file_list:
            relative = path.relative_to(release)
            bundle.write(path, arcname=(Path(release.name) / relative).as_posix())

    return {
        "part": part.number,
        "group": part.group,
        "filename": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "files": len(file_list),
        "uncompressed_bytes": sum(path.stat().st_size for path in file_list),
        "compression": (
            f"deflate-{part.compresslevel}"
            if part.compression == zipfile.ZIP_DEFLATED
            else "stored"
        ),
    }


def validate_parts(
    *,
    release: Path,
    output: Path,
    expected_files: list[Path],
    records: list[dict[str, object]],
    max_part_bytes: int,
) -> None:
    expected = {
        (Path(release.name) / path.relative_to(release)).as_posix()
        for path in expected_files
    }
    observed: set[str] = set()
    for record in records:
        archive = output / str(record["filename"])
        if archive.stat().st_size > max_part_bytes:
            raise ValueError(
                f"{archive.name} is {archive.stat().st_size:,} bytes; "
                f"limit is {max_part_bytes:,}"
            )
        with zipfile.ZipFile(archive) as bundle:
            corrupt = bundle.testzip()
            if corrupt:
                raise ValueError(f"{archive.name} contains a corrupt entry: {corrupt}")
            for name in bundle.namelist():
                if name.endswith("/") or name == UPLOAD_README_NAME:
                    continue
                if name in observed:
                    raise ValueError(f"duplicate file across ZIP parts: {name}")
                observed.add(name)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"ZIP partition mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )


def build_upload_parts(
    *,
    release: Path,
    output: Path,
    max_part_bytes: int,
    force: bool,
) -> list[dict[str, object]]:
    release = release.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (release / "RELEASE_MANIFEST.json").is_file():
        raise FileNotFoundError(f"not an assembled release: {release}")
    if max_part_bytes < 1:
        raise ValueError("max_part_bytes must be positive")
    if (
        output == release
        or release in output.parents
        or output in release.parents
    ):
        raise ValueError(
            "upload output must not overlap the release directory"
        )
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    files = release_files(release)
    by_group = {
        part.group: [path for path in files if group_for(path.relative_to(release)) == part.group]
        for part in PARTS
    }
    empty = [group for group, paths in by_group.items() if not paths]
    if empty:
        raise ValueError(f"release is missing upload groups: {empty}")

    instructions = upload_readme()
    records = [
        write_part(
            release=release,
            output=output,
            part=part,
            files=by_group[part.group],
            instructions=instructions,
        )
        for part in PARTS
    ]
    validate_parts(
        release=release,
        output=output,
        expected_files=files,
        records=records,
        max_part_bytes=max_part_bytes,
    )

    (output / UPLOAD_README_NAME).write_text(instructions, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "release_folder": release.name,
        "max_part_bytes": max_part_bytes,
        "all_parts_required": True,
        "release_manifest_sha256": sha256(release / "RELEASE_MANIFEST.json"),
        "release_checksums_sha256": sha256(release / "CHECKSUMS.sha256"),
        "parts": records,
    }
    (output / "UPLOAD_PARTS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def main() -> int:
    args = parse_args()
    release = args.release.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else release.parent / f"{release.name}-upload-parts"
    )
    records = build_upload_parts(
        release=release,
        output=output,
        max_part_bytes=args.max_part_bytes,
        force=args.force,
    )
    print(f"Upload set ready: {output}")
    for record in records:
        print(
            f"  {record['filename']}: "
            f"{format_size(int(record['bytes']))} "
            f"({record['files']} files)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
