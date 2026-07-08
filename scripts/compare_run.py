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

from gem_rags.analysis import DEFAULT_METRICS, compare_conditions, flatten_pairs, load_run_rows, parse_filter, write_csv


def main() -> int:
    args = _parse_args()
    result = compare_conditions(
        load_run_rows(args.runs),
        baseline_filter=parse_filter(args.baseline),
        candidate_filter=parse_filter(args.candidate),
        metrics=args.metric,
        match_fields=args.match_field,
    )
    if args.csv:
        write_csv(args.csv, result["metrics"])
    if args.pairs_csv:
        write_csv(args.pairs_csv, flatten_pairs(result["pairs"]))
    if not args.include_pairs:
        result = {key: value for key, value in result.items() if key != "pairs"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched GEM-RAG ablation conditions.")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--baseline", action="append", required=True, help="Baseline filter field=value. Repeatable.")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate filter field=value. Repeatable.")
    parser.add_argument("--metric", action="append", default=None, help=f"Metric to compare. Defaults: {', '.join(DEFAULT_METRICS)}")
    parser.add_argument("--match-field", action="append", default=None, help="Field used to match rows. Defaults to unchanged core config fields.")
    parser.add_argument("--csv", type=Path, help="Optional metric-summary CSV output.")
    parser.add_argument("--pairs-csv", type=Path, help="Optional matched-pair CSV output.")
    parser.add_argument("--include-pairs", action="store_true", help="Include matched pairs in JSON output.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
