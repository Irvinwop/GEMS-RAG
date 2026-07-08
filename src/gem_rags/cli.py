from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_run, compare_conditions, flatten_pairs, load_run_rows, parse_filter, summarize_rows, validate_run, write_csv
from .config import experiment_config_to_dict, load_experiment_config, write_experiment_config
from .data import load_qa_items
from .matrix import filter_ready_config, load_model_specs_file, materialize_config, parse_csv, parse_grader_spec, parse_model_spec
from .mrag_eval_import import import_mrag_eval
from .planning import plan_experiment
from .preflight import preflight_config
from .qa_sets import load_qa_ids_file, make_qa_split, summarize_qa_path, write_qa_split
from .regrade import regrade_run
from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gem-rags")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect a QA file.")
    inspect.add_argument("--qa-path", type=Path, default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl"))
    inspect.add_argument("--limit", type=int, default=3)

    qa_summary = sub.add_parser("qa-summary", help="Summarize a gold QA JSONL file.")
    qa_summary.add_argument("--qa-path", type=Path, default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl"))
    qa_summary.add_argument("--qa-ids", help="Comma-separated QA IDs to summarize.")
    qa_summary.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to summarize.")
    qa_summary.add_argument("--limit", type=int, help="Optional limit after QA ID filtering.")

    qa_split = sub.add_parser("qa-split", help="Create a deterministic QA ID split for ablation sweeps.")
    qa_split.add_argument("--qa-path", type=Path, default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl"))
    qa_split.add_argument("--size", type=int, required=True, help="Number of QA IDs to select.")
    qa_split.add_argument("--seed", type=int, default=0, help="Deterministic random seed.")
    qa_split.add_argument("--strategy", choices=["balanced", "proportional"], default="balanced")
    qa_split.add_argument("--output", type=Path, help="Write split JSON to this path; stdout when omitted.")

    run = sub.add_parser("run", help="Run an experiment config.")
    run.add_argument("config", type=Path)
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument("--overwrite", action="store_true", help="Replace the current runs.jsonl for this experiment.")
    run_mode.add_argument("--resume", action="store_true", help="Skip rows already present in runs.jsonl.")
    run_mode.add_argument("--retry-errors", action="store_true", help="Keep clean existing rows and rerun rows with retrieval/model/judge errors.")

    validate = sub.add_parser("validate", help="Validate run completeness, duplicates, and error counts against a config.")
    validate.add_argument("config", type=Path)
    validate.add_argument("--runs", type=Path, help="Run JSONL path. Defaults to output_dir/name/runs.jsonl from the config.")
    validate.add_argument("--allow-errors", action="store_true", help="Do not fail validation solely because retrieval/model/judge errors are present.")
    validate.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails.")

    regrade = sub.add_parser("regrade", help="Re-run grading over an existing runs.jsonl without rerunning retrieval or answer generation.")
    regrade.add_argument("config", type=Path)
    regrade.add_argument("--runs", type=Path, help="Input run JSONL. Defaults to output_dir/name/runs.jsonl from the config.")
    regrade.add_argument("--output", type=Path, help="Output run JSONL. Defaults to output_dir/name/regraded-runs.jsonl.")
    regrade.add_argument("--grader", help="Override grader as provider:model[,key=value...].")
    regrade.add_argument("--only-missing", action="store_true", help="Only regrade rows with missing judge scores or a judge_error.")
    regrade.add_argument("--strict", action="store_true", help="Exit non-zero if any row cannot be regraded cleanly.")

    analyze = sub.add_parser("analyze", help="Write summary and matched-pair ablation comparison artifacts.")
    analyze.add_argument("runs", type=Path, help="Run JSONL path.")
    analyze.add_argument("--output-dir", type=Path, help="Output directory. Defaults to runs parent / analysis.")
    analyze.add_argument("--qa-path", type=Path, help="Gold QA JSONL path used to add refusal/reference/figure strata artifacts.")
    analyze.add_argument("--filter", action="append", default=[], help="Restrict rows with field=value. Repeatable.")
    analyze.add_argument("--axis", help="Field to compare across, such as model, retriever, or context_mode.")
    analyze.add_argument("--baseline", help="Baseline value for --axis.")
    analyze.add_argument("--candidate", action="append", default=[], help="Candidate value for --axis. Repeatable; defaults to all observed non-baseline values.")
    analyze.add_argument("--metric", action="append", help="Metric to compare. Repeatable; defaults to standard metrics.")
    analyze.add_argument("--match-field", action="append", help="Override matched-pair fields. Repeatable.")
    analyze.add_argument("--no-pairs", action="store_true", help="Do not write per-question matched-pair CSV files.")

    import_mrag = sub.add_parser("import-mrag-eval", help="Normalize downloaded MRAG eval runs/scored JSONL into harness run rows.")
    import_mrag.add_argument("--mrag-dir", type=Path, default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG"))
    import_mrag.add_argument("--runs", type=Path, help="Source MRAG eval runs.jsonl. Defaults to mrag-dir/eval/runs.jsonl.")
    import_mrag.add_argument("--scored", type=Path, help="Source MRAG eval scored.jsonl. Defaults to mrag-dir/eval/scored.jsonl.")
    import_mrag.add_argument("--output", type=Path, default=Path("runs/mrag-prior-eval/runs.jsonl"))
    import_mrag.add_argument("--experiment-name", default="mrag-prior-eval")
    import_mrag.add_argument("--retriever-name", default="mrag_reference_prior")
    import_mrag.add_argument("--context-mode", default="injected")
    import_mrag.add_argument("--grader-name", default="mrag_prior_judge")
    import_mrag.add_argument("--overwrite", action="store_true")
    import_mrag.add_argument("--strict", action="store_true", help="Exit non-zero if any run row cannot be matched to QA or score rows.")

    preflight = sub.add_parser("preflight", help="Validate config paths, providers, credentials, and adapter readiness.")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--no-external-checks", action="store_true", help="Do not run known external adapter check commands.")
    preflight.add_argument("--timeout-s", type=int, default=30, help="Timeout per external adapter check.")
    preflight.add_argument("--strict", action="store_true", help="Exit non-zero when the preflight report is blocked.")

    materialize = sub.add_parser("materialize", help="Create a concrete experiment config from a template.")
    materialize.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(materialize, include_output=True)

    plan = sub.add_parser("plan", help="Materialize and enumerate a run matrix before spending model calls.")
    plan.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(plan, include_output=False)
    plan.add_argument("--preflight", action="store_true", help="Attach preflight readiness and blocking statuses.")
    plan.add_argument("--output", type=Path, help="Optional JSON plan output path.")
    plan.add_argument("--csv", type=Path, help="Optional condition-level CSV output path.")

    sweep = sub.add_parser("sweep", help="Materialize, preflight, run, summarize, and compare an ablation config.")
    sweep.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(sweep, include_output=False)
    sweep.add_argument("--config-output", type=Path, help="Materialized config path. Defaults under the run directory.")
    run_mode = sweep.add_mutually_exclusive_group()
    run_mode.add_argument("--overwrite", action="store_true", help="Replace the current runs.jsonl for this experiment.")
    run_mode.add_argument("--resume", action="store_true", help="Skip rows already present in runs.jsonl.")
    run_mode.add_argument("--retry-errors", action="store_true", help="Keep clean existing rows and rerun rows with retrieval/model/judge errors.")
    sweep.add_argument("--no-context-compare", action="store_true", help="Do not write injected vs tool_explore comparison artifacts.")
    sweep.add_argument("--allow-run-errors", action="store_true", help="Do not fail sweep validation solely because retrieval/model/judge errors are present.")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        items = load_qa_items(args.qa_path, limit=args.limit)
        print(json.dumps([item.raw for item in items], indent=2, ensure_ascii=False))
        return 0
    if args.command == "qa-summary":
        qa_ids = _qa_ids_from_args(args)
        print(json.dumps(summarize_qa_path(args.qa_path, limit=args.limit, qa_ids=qa_ids), indent=2, ensure_ascii=False))
        return 0
    if args.command == "qa-split":
        items = load_qa_items(args.qa_path)
        payload = {"qa_path": str(args.qa_path), **make_qa_split(items, size=args.size, seed=args.seed, strategy=args.strategy)}
        if args.output:
            write_qa_split(args.output, payload)
            print(args.output)
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run":
        output = run_experiment(
            load_experiment_config(args.config),
            overwrite=args.overwrite,
            resume=args.resume,
            retry_errors=args.retry_errors,
        )
        print(output)
        return 0
    if args.command == "validate":
        report = validate_run(load_experiment_config(args.config), args.runs, allow_errors=args.allow_errors)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and not report["ok"] else 0
    if args.command == "regrade":
        config = load_experiment_config(args.config)
        report = regrade_run(
            config,
            runs_path=args.runs,
            output_path=args.output,
            grader=parse_grader_spec(args.grader) if args.grader else None,
            only_missing=args.only_missing,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and not report["ok"] else 0
    if args.command == "analyze":
        report = analyze_run(
            args.runs,
            output_dir=args.output_dir,
            filters=parse_filter(args.filter),
            qa_path=args.qa_path,
            axis=args.axis,
            baseline=args.baseline,
            candidates=args.candidate,
            metrics=args.metric,
            match_fields=args.match_field,
            write_pairs=not args.no_pairs,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "import-mrag-eval":
        report = import_mrag_eval(
            args.mrag_dir,
            args.output,
            runs_path=args.runs,
            scored_path=args.scored,
            experiment_name=args.experiment_name,
            retriever_name=args.retriever_name,
            context_mode=args.context_mode,
            grader_name=args.grader_name,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and not report["ok"] else 0
    if args.command == "preflight":
        report = preflight_config(
            load_experiment_config(args.config),
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and not report["ok"] else 0
    if args.command == "materialize":
        config = _materialize_from_args(args)
        if args.output:
            write_experiment_config(config, args.output)
            print(args.output)
        else:
            print(json.dumps(experiment_config_to_dict(config), indent=2, ensure_ascii=False))
        return 0
    if args.command == "plan":
        config = _materialize_from_args(args)
        preflight_report = None
        if args.preflight:
            preflight_report = preflight_config(
                config,
                check_external=not args.no_external_checks,
                timeout_s=args.timeout_s,
            )
        report = plan_experiment(config, preflight_report=preflight_report)
        if args.output:
            _write_json(args.output, report)
        if args.csv:
            write_csv(args.csv, report["conditions"])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if preflight_report is not None and not preflight_report["ok"] else 0
    if args.command == "sweep":
        config = _materialize_from_args(args)
        run_dir = config.output_dir / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_output = args.config_output or run_dir / "materialized_config.json"
        write_experiment_config(config, config_output)

        preflight_report = preflight_config(
            config,
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
        )
        preflight_path = run_dir / "preflight.json"
        _write_json(preflight_path, preflight_report)
        if not preflight_report["ok"]:
            print(json.dumps({"status": "blocked", "config": str(config_output), "preflight": str(preflight_path)}, indent=2))
            return 2

        runs_path = run_experiment(config, overwrite=args.overwrite, resume=args.resume, retry_errors=args.retry_errors)
        rows = load_run_rows(runs_path)
        summary = {"runs": str(runs_path), "rows": len(rows), "groups": summarize_rows(rows)}
        summary_json = run_dir / "summary.json"
        summary_csv = run_dir / "summary.csv"
        _write_json(summary_json, summary)
        write_csv(summary_csv, summary["groups"])
        validation = validate_run(config, runs_path, allow_errors=args.allow_run_errors)
        validation_json = run_dir / "validation.json"
        _write_json(validation_json, validation)

        result = {
            "status": "complete" if validation["ok"] else "failed",
            "config": str(config_output),
            "preflight": str(preflight_path),
            "runs": str(runs_path),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "validation_json": str(validation_json),
            "validation_ok": validation["ok"],
            "rows": len(rows),
        }
        if not args.no_context_compare and {"injected", "tool_explore"}.issubset(set(config.context_modes)):
            comparison = compare_conditions(
                rows,
                baseline_filter={"context_mode": "injected"},
                candidate_filter={"context_mode": "tool_explore"},
            )
            comparison_without_pairs = {key: value for key, value in comparison.items() if key != "pairs"}
            context_compare_json = run_dir / "context-compare.json"
            context_compare_csv = run_dir / "context-compare.csv"
            context_pairs_csv = run_dir / "context-pairs.csv"
            _write_json(context_compare_json, comparison_without_pairs)
            write_csv(context_compare_csv, comparison_without_pairs["metrics"])
            write_csv(context_pairs_csv, flatten_pairs(comparison["pairs"]))
            result.update(
                {
                    "context_compare_json": str(context_compare_json),
                    "context_compare_csv": str(context_compare_csv),
                    "context_pairs_csv": str(context_pairs_csv),
                    "matched_context_pairs": comparison_without_pairs["matched_pairs"],
                }
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if validation["ok"] else 2
    raise AssertionError(args.command)


def _add_materialize_args(parser: argparse.ArgumentParser, *, include_output: bool) -> None:
    if include_output:
        parser.add_argument("--output", type=Path, help="Write the materialized config to this path; stdout when omitted.")
    parser.add_argument("--name", help="Override experiment name.")
    parser.add_argument("--limit", type=int, help="Override dataset question limit.")
    parser.add_argument("--qa-ids", help="Comma-separated QA IDs to run.")
    parser.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to run.")
    parser.add_argument("--retrievers", help="Comma-separated retriever names to keep.")
    parser.add_argument("--drop-retrievers", help="Comma-separated retriever names to drop.")
    parser.add_argument("--context-modes", help="Comma-separated context modes.")
    parser.add_argument(
        "--model",
        action="append",
        help="Replace model matrix with provider:model[,key=value...] specs. Repeat for multiple models.",
    )
    parser.add_argument("--models-file", type=Path, help="Replace model matrix from JSON or plain provider:model spec lines.")
    parser.add_argument("--grader", help="Override grader as provider:model[,key=value...].")
    parser.add_argument("--max-evidence-chars", type=int, help="Override max evidence chars.")
    parser.add_argument("--ready-only", action="store_true", help="Drop retrievers/models not ready under preflight.")
    parser.add_argument("--no-external-checks", action="store_true", help="Do not run external checks for --ready-only and sweep preflight.")
    parser.add_argument("--allow-not-checked", action="store_true", help="Keep not_checked retrievers during --ready-only.")
    parser.add_argument("--timeout-s", type=int, default=30, help="Timeout per external adapter check.")


def _materialize_from_args(args: argparse.Namespace):
    qa_ids = _qa_ids_from_args(args)
    models = _models_from_args(args)
    config = materialize_config(
        load_experiment_config(args.config),
        name=args.name,
        limit=args.limit,
        qa_ids=qa_ids,
        retriever_names=parse_csv(args.retrievers),
        drop_retriever_names=parse_csv(args.drop_retrievers),
        context_modes=parse_csv(args.context_modes),
        models=models,
        grader=parse_grader_spec(args.grader) if args.grader else None,
        max_evidence_chars=args.max_evidence_chars,
    )
    if args.ready_only:
        config, _ = filter_ready_config(
            config,
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
            allow_not_checked=args.allow_not_checked,
        )
    return config


def _models_from_args(args: argparse.Namespace):
    inline = getattr(args, "model", None)
    file_path = getattr(args, "models_file", None)
    if inline and file_path:
        raise ValueError("--model and --models-file are mutually exclusive")
    if file_path:
        return load_model_specs_file(file_path)
    return [parse_model_spec(spec) for spec in inline] if inline else None


def _qa_ids_from_args(args: argparse.Namespace) -> list[str] | None:
    inline = parse_csv(getattr(args, "qa_ids", None))
    file_path = getattr(args, "qa_ids_file", None)
    if inline and file_path:
        raise ValueError("--qa-ids and --qa-ids-file are mutually exclusive")
    if file_path:
        return load_qa_ids_file(file_path)
    return inline


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
