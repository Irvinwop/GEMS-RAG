#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gems_rag.analysis import load_run_rows, summarize_rows, write_csv


def main() -> int:
    args = _parse_args()
    rows = load_run_rows(args.runs)
    summary = summarize_rows(rows)
    if args.csv:
        write_csv(args.csv, summary)
    print(json.dumps({"runs": str(args.runs), "rows": len(rows), "groups": summary}, indent=2, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a GEMS-RAG runs.jsonl by retriever/context/model.")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
