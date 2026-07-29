#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT / "indexes" / "gems-rag"
MANIFEST_NAME = "visual_assets.json"
READY_NAME = ".visual_assets_ready.json"
VISUAL_COLLECTIONS = ("mutcd_pages", "mutcd_figures_visual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the full GEMS-RAG visual databases and media from "
            "the losslessly packed release assets."
        )
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing visual collections and media files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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


def load_manifest(assets: Path) -> dict[str, Any]:
    path = assets / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported visual asset manifest: {path}")
    return manifest


def fingerprint(assets: Path, manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    for relative in (
        manifest["qdrant_archive"]["path"],
        manifest["source_pdf"]["path"],
        manifest["figures"]["metadata_path"],
    ):
        path = assets / relative
        digest.update(relative.encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def verify_source_assets(assets: Path, manifest: dict[str, Any]) -> None:
    checks = (
        manifest["qdrant_archive"],
        manifest["source_pdf"],
        manifest["figures"]["metadata"],
    )
    for identity in checks:
        path = assets / identity["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != identity["sha256"]:
            raise ValueError(f"source asset checksum mismatch: {path}")


def extract_visual_qdrant(
    assets: Path,
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required; run scripts/setup_environments.sh "
            "gems-rag first"
        ) from exc

    archive = assets / manifest["qdrant_archive"]["path"]
    expected = manifest["qdrant_archive"]["collections"]
    qdrant = assets / "qdrant_db"
    temporary = assets / ".visual_qdrant.materializing"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    allowed_roots = {
        PurePosixPath("collection") / name for name in VISUAL_COLLECTIONS
    }
    try:
        with archive.open("rb") as compressed:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as bundle:
                    for member in bundle:
                        relative = PurePosixPath(member.name)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError(
                                f"unsafe visual archive member: {member.name}"
                            )
                        if not any(
                            relative == root or root in relative.parents
                            for root in allowed_roots
                        ):
                            raise ValueError(
                                f"unexpected visual archive member: "
                                f"{member.name}"
                            )
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ValueError(
                                f"unsupported visual archive member: "
                                f"{member.name}"
                            )
                        source = bundle.extractfile(member)
                        if source is None:
                            raise ValueError(
                                f"cannot read visual archive member: "
                                f"{member.name}"
                            )
                        target = temporary.joinpath(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("wb") as handle:
                            shutil.copyfileobj(source, handle, 4 * 1024 * 1024)

        for name in VISUAL_COLLECTIONS:
            source = temporary / "collection" / name
            storage = source / "storage.sqlite"
            identity = expected[name]["storage"]
            if (
                not storage.is_file()
                or storage.stat().st_size != identity["bytes"]
                or sha256(storage) != identity["sha256"]
            ):
                raise ValueError(
                    f"materialized Qdrant checksum mismatch: {name}"
                )

            destination = qdrant / "collection" / name
            existing = destination / "storage.sqlite"
            if (
                not force
                and existing.is_file()
                and existing.stat().st_size == identity["bytes"]
                and sha256(existing) == identity["sha256"]
            ):
                continue
            shutil.rmtree(destination, ignore_errors=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def render_pages(
    assets: Path,
    manifest: dict[str, Any],
    document: Any,
    fitz_module: Any,
    *,
    force: bool,
) -> int:
    page_config = manifest["pages"]
    expected = int(page_config["count"])
    if document.page_count != expected:
        raise ValueError(
            f"PDF page count mismatch: expected {expected}, "
            f"found {document.page_count}"
        )
    dpi = int(page_config["dpi"])
    matrix = fitz_module.Matrix(dpi / 72.0, dpi / 72.0)
    output = assets / page_config["directory"]
    output.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for index in range(expected):
        target = output / f"page_{index + 1:04d}.png"
        identity = page_config["files"][target.name]
        if (
            target.is_file()
            and target.stat().st_size == identity["bytes"]
            and sha256(target) == identity["sha256"]
            and not force
        ):
            continue
        temporary = target.with_name(f".{target.name}.tmp.png")
        try:
            pixmap = document[index].get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            pixmap.save(str(temporary))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        rendered += 1
        if rendered % 100 == 0:
            print(f"  rendered pages: {rendered}", flush=True)
    return rendered


def load_figure_rows(assets: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = assets / manifest["figures"]["metadata_path"]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = int(manifest["figures"]["canonical_count"])
    if len(rows) != expected:
        raise ValueError(
            f"figure metadata count mismatch: expected {expected}, "
            f"found {len(rows)}"
        )
    names = [Path(str(row["image_path"])).name for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("figure metadata contains duplicate image paths")
    return rows


def render_figures(
    assets: Path,
    rows: list[dict[str, Any]],
    document: Any,
    fitz_module: Any,
    identities: dict[str, Any],
    *,
    force: bool,
) -> int:
    output = assets / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for row in rows:
        target = output / Path(str(row["image_path"])).name
        identity = identities[target.name]
        if (
            target.is_file()
            and target.stat().st_size == identity["bytes"]
            and sha256(target) == identity["sha256"]
            and not force
        ):
            continue
        page_number = int(row["page_pdf"])
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"invalid figure page for {target.name}: {page_number}"
            )
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid figure bbox for {target.name}")
        dpi = int(row.get("dpi") or 220)
        matrix = fitz_module.Matrix(dpi / 72.0, dpi / 72.0)
        rectangle = fitz_module.Rect(*map(float, bbox))
        temporary = target.with_name(f".{target.name}.tmp.png")
        try:
            pixmap = document[page_number - 1].get_pixmap(
                matrix=matrix,
                clip=rectangle,
                alpha=False,
            )
            pixmap.save(str(temporary))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        rendered += 1
        if rendered % 100 == 0:
            print(f"  rendered figures: {rendered}", flush=True)
    return rendered


def materialize_media(
    assets: Path,
    manifest: dict[str, Any],
    *,
    force: bool,
) -> tuple[int, int]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required; run scripts/setup_environments.sh "
            "gems-rag first"
        ) from exc
    expected_version = str(manifest["renderer"]["version"])
    if fitz.__version__ != expected_version:
        raise RuntimeError(
            f"visual media materialization requires PyMuPDF "
            f"{expected_version}; found {fitz.__version__}"
        )

    document = fitz.open(str(assets / manifest["source_pdf"]["path"]))
    try:
        pages = render_pages(
            assets,
            manifest,
            document,
            fitz,
            force=force,
        )
        rows = load_figure_rows(assets, manifest)
        figures = render_figures(
            assets,
            rows,
            document,
            fitz,
            manifest["figures"]["files"],
            force=force,
        )
    finally:
        document.close()
    return pages, figures


def apply_media_overrides(
    assets: Path,
    manifest: dict[str, Any],
) -> int:
    copied = 0
    for group in ("pages", "figures"):
        directory = manifest[group]["directory"]
        identities = manifest[group]["files"]
        source_root = assets / "media_overrides" / directory
        if not source_root.is_dir():
            continue
        destination_root = assets / directory
        destination_root.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_root.glob("*.png")):
            identity = identities.get(source.name)
            if identity is None:
                raise ValueError(f"unexpected media override: {source}")
            if (
                source.stat().st_size != identity["bytes"]
                or sha256(source) != identity["sha256"]
            ):
                raise ValueError(f"media override checksum mismatch: {source}")
            destination = destination_root / source.name
            temporary = destination.with_name(f".{destination.name}.override")
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            copied += 1
    return copied


def validate_materialized(assets: Path, manifest: dict[str, Any]) -> None:
    expected_collections = manifest["qdrant_archive"]["collections"]
    for name in VISUAL_COLLECTIONS:
        storage = assets / "qdrant_db" / "collection" / name / "storage.sqlite"
        identity = expected_collections[name]["storage"]
        if (
            not storage.is_file()
            or storage.stat().st_size != identity["bytes"]
            or sha256(storage) != identity["sha256"]
        ):
            raise ValueError(f"visual Qdrant collection is incomplete: {name}")

    for group in ("pages", "figures"):
        directory = assets / manifest[group]["directory"]
        expected_files = manifest[group]["files"]
        observed = {path.name for path in directory.glob("*.png")}
        if observed != set(expected_files):
            missing = sorted(set(expected_files) - observed)
            extra = sorted(observed - set(expected_files))
            raise ValueError(
                f"materialized {group} inventory mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for name, identity in expected_files.items():
            path = directory / name
            if (
                path.stat().st_size != identity["bytes"]
                or sha256(path) != identity["sha256"]
            ):
                raise ValueError(
                    f"materialized media checksum mismatch: {path}"
                )


def main() -> int:
    args = parse_args()
    assets = args.assets.expanduser().resolve()
    manifest = load_manifest(assets)
    verify_source_assets(assets, manifest)
    expected_fingerprint = fingerprint(assets, manifest)
    ready_path = assets / READY_NAME
    if ready_path.is_file() and not args.force:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("fingerprint") == expected_fingerprint:
            validate_materialized(assets, manifest)
            print(f"Visual assets already materialized: {assets}")
            return 0

    print("Materializing losslessly packed visual Qdrant collections", flush=True)
    extract_visual_qdrant(assets, manifest, force=args.force)
    print("Rendering page and figure media from the included PDF", flush=True)
    pages, figures = materialize_media(
        assets,
        manifest,
        force=args.force,
    )
    overrides = apply_media_overrides(assets, manifest)
    validate_materialized(assets, manifest)
    atomic_write_json(
        ready_path,
        {
            "schema_version": 1,
            "fingerprint": expected_fingerprint,
            "qdrant_collections": list(VISUAL_COLLECTIONS),
            "pages": int(manifest["pages"]["count"]),
            "figures": int(manifest["figures"]["count"]),
        },
    )
    print(
        f"Visual assets ready: {assets} "
        f"(rendered {pages} pages, {figures} canonical figures; "
        f"applied {overrides} exact media overrides)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
