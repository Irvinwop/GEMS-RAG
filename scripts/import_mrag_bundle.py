#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.mrag_bundle import import_mrag_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a split MRAG Drive bundle into ignored project data.")
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--no-verify-detached-blobs", action="store_true")
    parser.add_argument(
        "--fallback-hf-cache",
        action="append",
        default=[],
        type=Path,
        help="Use verified blobs from another Hugging Face cache when Drive flattened them.",
    )
    args = parser.parse_args()
    report = import_mrag_bundle(
        args.raw_dir,
        args.output_dir,
        force=args.force,
        restore_qdrant=not args.skip_qdrant,
        verify_detached_blobs=not args.no_verify_detached_blobs,
        fallback_hf_caches=args.fallback_hf_cache,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
