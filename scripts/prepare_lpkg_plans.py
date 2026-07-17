#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems_rag.data import load_qa_items
from gems_rag.config import DEFAULT_QA_PATH as DEFAULT_QA_RELATIVE_PATH
from gems_rag.manuscript_retrievers import load_lpkg_plans, parse_lpkg_subquestions

DEFAULT_QA_PATH = ROOT / DEFAULT_QA_RELATIVE_PATH
DEFAULT_PLANS_PATH = ROOT / "data" / "working" / "lpkg" / "generated_plans.jsonl"
DEFAULT_REPO = ROOT / "external" / "rag-implementations" / "lpkg"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare official-syntax LPKG plans for the GEMS-RAG harness."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--predictions", type=Path, required=True)
    normalize.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    normalize.add_argument("--out", type=Path, default=DEFAULT_PLANS_PATH)
    normalize.add_argument("--force", action="store_true")

    atomic = subparsers.add_parser(
        "atomic",
        help="Generate a deterministic one-step fallback when no learned planner checkpoint is available.",
    )
    atomic.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    atomic.add_argument("--out", type=Path, default=DEFAULT_PLANS_PATH)
    atomic.add_argument("--force", action="store_true")

    check = subparsers.add_parser("check")
    check.add_argument("--plans", type=Path, default=DEFAULT_PLANS_PATH)
    check.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    check.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    return parser


def normalize_predictions(
    predictions_path: Path,
    qa_path: Path,
    out_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {out_path}; pass --force")
    predictions = _prediction_rows(predictions_path)
    items = load_qa_items(qa_path)
    if len(predictions) != len(items):
        raise ValueError(
            f"prediction/QA row mismatch: {len(predictions)} predictions for {len(items)} QA items"
        )

    rows = []
    invalid = []
    for index, (prediction, item) in enumerate(zip(predictions, items, strict=True), 1):
        plan = prediction.get("predict") or prediction.get("plan")
        if not isinstance(plan, str) or not parse_lpkg_subquestions(plan):
            invalid.append(index)
            continue
        rows.append(
            {
                "qa_id": item.qa_id,
                "question": item.question,
                "predict": plan,
                "label": prediction.get("label"),
                "source": str(predictions_path),
                "planner_format": "official_lpkg_generated_predictions",
                "planner_model": prediction.get("planner_model"),
                "planner_checkpoint": prediction.get("planner_checkpoint"),
            }
        )
    if invalid:
        raise ValueError(f"unparseable LPKG plans at prediction rows: {invalid}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "status": "normalized",
        "predictions": str(predictions_path),
        "qa_path": str(qa_path),
        "plans": str(out_path),
        "plan_count": len(rows),
    }


def generate_atomic_plans(
    qa_path: Path,
    out_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write valid one-step LPKG plans without claiming learned planning behavior."""
    if out_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {out_path}; pass --force")
    items = load_qa_items(qa_path)
    if not items:
        raise ValueError(f"QA file contains no items: {qa_path}")

    rows = []
    for item in items:
        question = json.dumps(item.question, ensure_ascii=False)
        plan = "\n".join(
            (
                'Thought1: str = "Use one direct retrieval step because the learned planner checkpoint '
                'was not released."',
                f"Sub_Question_1: str = {question}",
                "Info_1: str = Search(query = Sub_Question_1, thought = Thought1)",
                "Ans_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)",
                "Final_Answer: str = Finish_The_Plan(Answer = Ans_1)",
            )
        )
        rows.append(
            {
                "qa_id": item.qa_id,
                "question": item.question,
                "predict": plan,
                "label": None,
                "source": str(qa_path),
                "planner_format": "official_lpkg_atomic_fallback",
                "planner_model": None,
                "planner_checkpoint": "unavailable_upstream",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "status": "generated",
        "qa_path": str(qa_path),
        "plans": str(out_path),
        "plan_count": len(rows),
        "planner_format": "official_lpkg_atomic_fallback",
        "planner_checkpoint": "unavailable_upstream",
        "scientific_scope": "availability_smoke_not_learned_planner_reproduction",
    }


def check_plans(plans_path: Path, qa_path: Path, repo: Path = DEFAULT_REPO) -> dict[str, Any]:
    repo_found = (repo / "parser" / "parse_result.py").exists()
    plans_found = plans_path.exists()
    qa_found = qa_path.exists()
    report: dict[str, Any] = {
        "repo": str(repo),
        "repo_found": repo_found,
        "plans": str(plans_path),
        "plans_found": plans_found,
        "qa_path": str(qa_path),
        "qa_found": qa_found,
        "environment_ready": repo_found and qa_found,
        "plan_count": 0,
        "qa_count": 0,
        "missing_qa_ids": [],
        "unparseable_qa_ids": [],
    }
    if not (repo_found and plans_found and qa_found):
        report["runnable"] = False
        report["notes"] = "Clone the official LPKG repo and normalize its generated predictions before running."
        return report
    try:
        plans = load_lpkg_plans(plans_path)
        items = load_qa_items(qa_path)
    except Exception as exc:
        report["runnable"] = False
        report["error"] = repr(exc)
        return report

    by_qa_id = {str(row.get("qa_id")): row for row in plans if row.get("qa_id")}
    report["plan_count"] = len(plans)
    report["qa_count"] = len(items)
    report["missing_qa_ids"] = [item.qa_id for item in items if item.qa_id not in by_qa_id]
    report["unparseable_qa_ids"] = [
        qa_id
        for qa_id, row in by_qa_id.items()
        if not parse_lpkg_subquestions(str(row.get("predict") or ""))
    ]
    report["planner_formats"] = sorted(
        {str(row.get("planner_format") or "unspecified") for row in plans}
    )
    report["planner_checkpoints"] = sorted(
        {str(row.get("planner_checkpoint") or "unspecified") for row in plans}
    )
    report["runnable"] = not report["missing_qa_ids"] and not report["unparseable_qa_ids"]
    report["notes"] = (
        "Official LPKG plan syntax is ready for shared-corpus iterative retrieval."
        if report["runnable"]
        else "Plans must cover every selected QA item and contain parseable Sub_Question_n assignments."
    )
    return report


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict) and (payload.get("predict") or payload.get("plan")):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"official LPKG predictions must be a JSON list or JSONL objects: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "normalize":
        try:
            report = normalize_predictions(
                args.predictions,
                args.qa_path,
                args.out,
                force=args.force,
            )
        except Exception as exc:
            print(json.dumps({"status": "blocked", "error": repr(exc)}, indent=2))
            return 2
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "atomic":
        try:
            report = generate_atomic_plans(args.qa_path, args.out, force=args.force)
        except Exception as exc:
            print(json.dumps({"status": "blocked", "error": repr(exc)}, indent=2))
            return 2
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    report = check_plans(args.plans, args.qa_path, args.repo)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
