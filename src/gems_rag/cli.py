from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .ablation_bundle import prepare_ablation_bundle
from .analysis import analyze_run, compare_conditions, flatten_pairs, leaderboard_rows, load_run_rows, parse_filter, summarize_rows, validate_run, write_csv
from .config import DEFAULT_MRAG_DIR, DEFAULT_QA_PATH, experiment_config_to_dict, load_experiment_config, write_experiment_config
from .context_segments import write_context_segments
from .control_plane import serve_gui
from .data import load_qa_items
from .external_setup import add_external_index_args, build_external_indexes, external_index_exit_code
from .matrix import filter_ready_config, load_model_specs_file, materialize_config, parse_csv, parse_grader_spec, parse_model_spec
from .manuscript_rags import load_manuscript_rag_catalog, validate_manuscript_rag_coverage
from .manual import manual_status, write_manual_manifest
from .model_catalog import (
    catalog_entries_to_models_payload,
    catalog_pricing_payload,
    load_model_catalog,
    pricing_coverage_for_config,
    render_model_specs,
    select_model_catalog,
)
from .mrag_eval_import import import_mrag_eval
from .planning import evaluate_plan_budget, plan_experiment
from .preflight import preflight_config
from .qa_sets import (
    evaluate_qa_coverage,
    load_qa_ids_file,
    make_qa_split,
    qa_coverage_for_selection,
    summarize_qa_path,
    write_qa_split,
)
from .rag_audit import audit_retrievers, write_rag_audit
from .regrade import regrade_run
from .retriever_catalog import catalog_entries_to_retrievers_payload, load_retriever_catalog, load_retriever_specs_file, select_retriever_catalog
from .retriever_profiles import apply_retriever_profile, load_retriever_profile
from .run_bundles import export_run_bundle, import_pro_grades
from .runner import run_experiment
from .upstream_exports import add_upstream_export_args, upstream_export_from_args

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gems-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect a QA file.")
    inspect.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    inspect.add_argument("--limit", type=int, default=3)

    gui = sub.add_parser("gui", help="Start the local model picker.")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")

    qa_summary = sub.add_parser("qa-summary", help="Summarize a gold QA JSONL file.")
    qa_summary.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    qa_summary.add_argument("--qa-ids", help="Comma-separated QA IDs to summarize.")
    qa_summary.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to summarize.")
    qa_summary.add_argument("--limit", type=int, help="Optional limit after QA ID filtering.")

    qa_split = sub.add_parser("qa-split", help="Create a deterministic QA ID split for ablation sweeps.")
    qa_split.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    qa_split.add_argument("--size", type=int, required=True, help="Number of QA IDs to select.")
    qa_split.add_argument("--seed", type=int, default=0, help="Deterministic random seed.")
    qa_split.add_argument("--strategy", choices=["balanced", "proportional"], default="balanced")
    qa_split.add_argument("--output", type=Path, help="Write split JSON to this path; stdout when omitted.")

    model_matrix = sub.add_parser("model-matrix", help="Generate provider:model spec lines from a model catalog.")
    model_matrix.add_argument("catalog", type=Path, nargs="?", default=Path("configs/model-catalog.example.json"))
    model_matrix.add_argument("--providers", help="Comma-separated providers to include, such as openai,anthropic,xai,qwen,local_openai.")
    model_matrix.add_argument("--sizes", help="Comma-separated model size labels to include.")
    model_matrix.add_argument("--roles", default="answer", help="Comma-separated roles to include. Defaults to answer.")
    model_matrix.add_argument("--tags", help="Comma-separated tags; selected entries must include all requested tags.")
    model_matrix.add_argument("--include-disabled", action="store_true", help="Include catalog entries marked enabled=false.")
    model_matrix.add_argument("--format", choices=["plain", "json"], default="plain", help="Output plain model spec lines or JSON.")
    model_matrix.add_argument("--output", type=Path, help="Write output to this file; stdout when omitted.")

    retriever_matrix = sub.add_parser("retriever-matrix", help="Generate retriever config JSON from a retriever catalog.")
    retriever_matrix.add_argument("catalog", type=Path, nargs="?", default=Path("configs/retriever-catalog.example.json"))
    retriever_matrix.add_argument("--families", help="Comma-separated retriever families to include, such as lightrag,graphrag.")
    retriever_matrix.add_argument("--modes", help="Comma-separated retrieval modes to include.")
    retriever_matrix.add_argument("--tags", help="Comma-separated tags; selected entries must include all requested tags.")
    retriever_matrix.add_argument("--include-disabled", action="store_true", help="Include catalog entries marked enabled=false.")
    retriever_matrix.add_argument("--output", type=Path, help="Write retriever JSON to this file; stdout when omitted.")

    apply_profile = sub.add_parser(
        "apply-retriever-profile",
        help="Apply bounded index command options to a materialized experiment config.",
    )
    apply_profile.add_argument("config", type=Path)
    apply_profile.add_argument("profile", type=Path)
    apply_profile.add_argument("--output", type=Path, required=True)

    segment_contexts = sub.add_parser(
        "segment-contexts",
        help="Write one resumable experiment config per compatible context mode.",
    )
    segment_contexts.add_argument("config", type=Path)
    segment_contexts.add_argument("--context-modes", help="Comma-separated modes; defaults to all four.")
    segment_contexts.add_argument("--output-dir", type=Path, required=True)

    manuscript_coverage = sub.add_parser(
        "manuscript-coverage",
        help="Verify that every audited manuscript RAG has an enabled retriever integration.",
    )
    manuscript_coverage.add_argument(
        "--manuscript-catalog",
        type=Path,
        default=Path("configs/manuscript-rags.json"),
    )
    manuscript_coverage.add_argument(
        "--retriever-catalog",
        type=Path,
        default=Path("configs/retriever-catalog.example.json"),
    )
    manuscript_coverage.add_argument("--output", type=Path)

    manual = sub.add_parser("manual-status", help="Verify the MUTCD PDF and every manual-derived evaluation artifact.")
    manual.add_argument("--mrag-dir", type=Path, default=DEFAULT_MRAG_DIR)
    manual.add_argument("--output", type=Path, help="Write the reproducible manual manifest to this path.")
    manual.add_argument("--strict", action="store_true", help="Exit non-zero when any manual artifact check fails.")

    prepare_ablation = sub.add_parser("prepare-ablation", help="Write a catalog-driven ablation bundle without running model calls.")
    prepare_ablation.add_argument("config", type=Path, help="Base experiment config.")
    prepare_ablation.add_argument("--name", help="Experiment name for the materialized config and run directory.")
    prepare_ablation.add_argument("--output-dir", type=Path, help="Bundle output directory. Defaults under data/working/ablation-bundles/<name>.")
    prepare_ablation.add_argument("--qa-size", type=int, help="Create a deterministic QA split with this many IDs.")
    prepare_ablation.add_argument("--qa-seed", type=int, default=0)
    prepare_ablation.add_argument("--qa-strategy", choices=["balanced", "proportional"], default="balanced")
    prepare_ablation.add_argument("--qa-ids", help="Comma-separated QA IDs to use instead of creating a split.")
    prepare_ablation.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to use instead of creating a split.")
    prepare_ablation.add_argument("--limit", type=int, help="Optional dataset limit when not using explicit QA IDs.")
    prepare_ablation.add_argument("--model-catalog", type=Path, default=Path("configs/model-catalog.example.json"))
    prepare_ablation.add_argument("--model-providers", help="Comma-separated model providers to include.")
    prepare_ablation.add_argument("--model-sizes", help="Comma-separated model size labels to include.")
    prepare_ablation.add_argument("--model-tags", help="Comma-separated model tags; selected entries must include all requested tags.")
    prepare_ablation.add_argument("--include-disabled-models", action="store_true")
    prepare_ablation.add_argument("--grader-from-catalog", action="store_true", help="Select exactly one role=grader entry from --model-catalog.")
    prepare_ablation.add_argument("--grader-providers", help="Comma-separated grader providers to include when --grader-from-catalog is set.")
    prepare_ablation.add_argument("--grader-sizes", help="Comma-separated grader size labels to include when --grader-from-catalog is set.")
    prepare_ablation.add_argument("--grader-tags", help="Comma-separated grader tags; selected grader must include all requested tags.")
    prepare_ablation.add_argument("--include-disabled-graders", action="store_true")
    prepare_ablation.add_argument("--retriever-catalog", type=Path, default=Path("configs/retriever-catalog.example.json"))
    prepare_ablation.add_argument("--retriever-families", help="Comma-separated retriever families to include.")
    prepare_ablation.add_argument("--retriever-modes", help="Comma-separated retriever modes to include.")
    prepare_ablation.add_argument("--retriever-tags", help="Comma-separated retriever tags; selected entries must include all requested tags.")
    prepare_ablation.add_argument("--include-disabled-retrievers", action="store_true")
    prepare_ablation.add_argument("--context-modes", help="Comma-separated context modes.")
    prepare_ablation.add_argument("--grader", help="Override grader as provider:model[,key=value...].")
    prepare_ablation.add_argument("--max-evidence-chars", type=int, help="Override max evidence chars.")
    prepare_ablation.add_argument("--dry-run", action="store_true", help="Materialize a config that never calls answer or judge models.")
    _add_budget_args(prepare_ablation)
    _add_qa_coverage_args(prepare_ablation)
    prepare_ablation.add_argument(
        "--max-total-cost-usd",
        type=_nonnegative_float,
        help="Propagate an observed post-run USD cost ceiling to generated sweep and validation commands.",
    )
    prepare_ablation.add_argument("--preflight", action="store_true", help="Attach preflight readiness to the plan bundle.")
    prepare_ablation.add_argument("--no-external-checks", action="store_true", help="Do not run external checks when --preflight is set.")
    prepare_ablation.add_argument("--timeout-s", type=int, default=30, help="Timeout per external adapter check.")
    prepare_ablation.add_argument("--strict", action="store_true", help="Exit non-zero when any bundle launch gate is blocked.")

    external_indexes = sub.add_parser("external-indexes", help="Check or build ignored local indexes for cloned external RAG adapters.")
    add_external_index_args(external_indexes)

    upstream_inputs = sub.add_parser("upstream-inputs", help="Export harness QA/evidence rows for cloned upstream Self-RAG and CRAG eval scripts.")
    add_upstream_export_args(upstream_inputs)

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
    validate.add_argument("--max-total-tokens", type=int, help="Fail when observed answer plus judge token usage exceeds this limit.")
    validate.add_argument("--model-catalog", type=Path, help="Model catalog containing pricing metadata for observed cost validation.")
    validate.add_argument("--max-total-cost-usd", type=_nonnegative_float, help="Fail when fully priced observed model usage exceeds this USD limit.")
    validate.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails.")

    regrade = sub.add_parser("regrade", help="Re-run grading over an existing runs.jsonl without rerunning retrieval or answer generation.")
    regrade.add_argument("config", type=Path)
    regrade.add_argument("--runs", type=Path, help="Input run JSONL. Defaults to output_dir/name/runs.jsonl from the config.")
    regrade.add_argument("--output", type=Path, help="Output run JSONL. Defaults to output_dir/name/regraded-runs.jsonl.")
    regrade.add_argument("--grader", help="Override grader as provider:model[,key=value...].")
    regrade.add_argument("--only-missing", action="store_true", help="Only regrade rows with missing judge scores or a judge_error.")
    regrade.add_argument("--strict", action="store_true", help="Exit non-zero if any row cannot be regraded cleanly.")

    export_bundle = sub.add_parser("export-bundle", help="Archive a run or create a self-contained GPT Pro grading ZIP.")
    export_bundle.add_argument("runs", type=Path, help="runs.jsonl or its run directory.")
    export_bundle.add_argument("--output", type=Path)
    export_bundle.add_argument("--qa-path", type=Path, help="Gold QA JSONL; inferred from materialized_config.json when possible.")
    export_bundle.add_argument(
        "--grader-spec",
        type=Path,
        help="Markdown evaluation specification to attach verbatim to the bundle.",
    )
    export_bundle.add_argument("--mode", choices=["archive", "gpt_pro"], default="gpt_pro")

    import_grades = sub.add_parser("import-pro-grades", help="Merge GPT Pro grades into an existing run without rerunning answers.")
    import_grades.add_argument("runs", type=Path, help="runs.jsonl or its run directory.")
    import_grades.add_argument("grades", type=Path, help="grades.jsonl or a ZIP containing grades.jsonl.")
    import_grades.add_argument("--output", type=Path)
    import_grades.add_argument("--strict", action="store_true", help="Exit non-zero unless every run row has a matching grade.")

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
    analyze.add_argument("--model-catalog", type=Path, help="Optional model catalog with pricing metadata for observed cost analysis.")
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

    rag_audit = sub.add_parser(
        "rag-audit",
        help="Preflight each RAG and smoke-test every context mode it supports.",
    )
    rag_audit.add_argument("config", type=Path)
    rag_audit.add_argument("--no-external-checks", action="store_true", help="Do not run external adapter readiness checks.")
    rag_audit.add_argument("--timeout-s", type=int, default=30, help="Timeout per external adapter check.")
    rag_audit.add_argument("--output", type=Path, help="Write the full JSON report to this path.")
    rag_audit.add_argument("--strict", action="store_true", help="Exit non-zero unless every selected RAG passes.")

    materialize = sub.add_parser("materialize", help="Create a concrete experiment config from a template.")
    materialize.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(materialize, include_output=True)

    plan = sub.add_parser("plan", help="Materialize and enumerate a run matrix before spending model calls.")
    plan.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(plan, include_output=False)
    plan.add_argument("--preflight", action="store_true", help="Attach preflight readiness and blocking statuses.")
    plan.add_argument("--output", type=Path, help="Optional JSON plan output path.")
    plan.add_argument("--csv", type=Path, help="Optional condition-level CSV output path.")
    _add_budget_args(plan)
    _add_qa_coverage_args(plan)

    sweep = sub.add_parser("sweep", help="Materialize, preflight, run, summarize, and compare an ablation config.")
    sweep.add_argument("config", type=Path, help="Base experiment config.")
    _add_materialize_args(sweep, include_output=False)
    sweep.add_argument("--config-output", type=Path, help="Materialized config path. Defaults under the run directory.")
    run_mode = sweep.add_mutually_exclusive_group()
    run_mode.add_argument("--overwrite", action="store_true", help="Replace the current runs.jsonl for this experiment.")
    run_mode.add_argument("--resume", action="store_true", help="Skip rows already present in runs.jsonl.")
    run_mode.add_argument("--retry-errors", action="store_true", help="Keep clean existing rows and rerun rows with retrieval/model/judge errors.")
    sweep.add_argument("--no-context-compare", action="store_true", help="Do not write injected vs tool context comparison artifacts.")
    sweep.add_argument("--allow-run-errors", action="store_true", help="Do not fail sweep validation solely because retrieval/model/judge errors are present.")
    _add_budget_args(sweep)
    _add_qa_coverage_args(sweep)
    sweep.add_argument("--model-catalog", type=Path, help="Model catalog containing pricing metadata for summaries and post-run validation.")
    sweep.add_argument("--max-total-cost-usd", type=_nonnegative_float, help="Fail post-run validation when fully priced observed usage exceeds this USD limit.")

    args = parser.parse_args(argv)
    if (
        args.command in {"validate", "sweep"}
        and getattr(args, "max_total_cost_usd", None) is not None
        and getattr(args, "model_catalog", None) is None
    ):
        parser.error("--max-total-cost-usd requires --model-catalog")
    os.chdir(ROOT)
    if args.command == "gui":
        serve_gui(args.host, args.port, open_browser=not args.no_open)
        return 0
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
    if args.command == "model-matrix":
        entries = select_model_catalog(
            load_model_catalog(args.catalog),
            providers=parse_csv(args.providers),
            sizes=parse_csv(args.sizes),
            roles=parse_csv(args.roles),
            tags=parse_csv(args.tags),
            include_disabled=args.include_disabled,
        )
        if args.format == "json":
            text = json.dumps(catalog_entries_to_models_payload(entries), indent=2, ensure_ascii=False) + "\n"
        else:
            text = render_model_specs(entries)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text, end="")
        return 0
    if args.command == "retriever-matrix":
        entries = select_retriever_catalog(
            load_retriever_catalog(args.catalog),
            families=parse_csv(args.families),
            modes=parse_csv(args.modes),
            tags=parse_csv(args.tags),
            include_disabled=args.include_disabled,
        )
        text = json.dumps(catalog_entries_to_retrievers_payload(entries), indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text, end="")
        return 0
    if args.command == "apply-retriever-profile":
        config, report = apply_retriever_profile(
            load_experiment_config(args.config),
            load_retriever_profile(args.profile),
        )
        write_experiment_config(config, args.output)
        print(
            json.dumps(
                {"status": "ready", "output": str(args.output), **report},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "segment-contexts":
        report = write_context_segments(
            load_experiment_config(args.config),
            args.output_dir,
            context_modes=parse_csv(args.context_modes),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "manuscript-coverage":
        report = validate_manuscript_rag_coverage(
            load_manuscript_rag_catalog(args.manuscript_catalog),
            load_retriever_catalog(args.retriever_catalog),
        )
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text, end="")
        return 0 if report["ok"] else 2
    if args.command == "manual-status":
        report = manual_status(mrag_dir=args.mrag_dir)
        if args.output:
            write_manual_manifest(args.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and report["status"] != "ready" else 0
    if args.command == "prepare-ablation":
        report = prepare_ablation_bundle(
            base_config_path=args.config,
            name=args.name,
            output_dir=args.output_dir,
            qa_size=args.qa_size,
            qa_seed=args.qa_seed,
            qa_strategy=args.qa_strategy,
            qa_ids=_qa_ids_from_args(args),
            limit=args.limit,
            model_catalog_path=args.model_catalog,
            model_providers=parse_csv(args.model_providers),
            model_sizes=parse_csv(args.model_sizes),
            model_tags=parse_csv(args.model_tags),
            include_disabled_models=args.include_disabled_models,
            grader_from_catalog=args.grader_from_catalog,
            grader_providers=parse_csv(args.grader_providers),
            grader_sizes=parse_csv(args.grader_sizes),
            grader_tags=parse_csv(args.grader_tags),
            include_disabled_graders=args.include_disabled_graders,
            retriever_catalog_path=args.retriever_catalog,
            retriever_families=parse_csv(args.retriever_families),
            retriever_modes=parse_csv(args.retriever_modes),
            retriever_tags=parse_csv(args.retriever_tags),
            include_disabled_retrievers=args.include_disabled_retrievers,
            context_modes=parse_csv(args.context_modes),
            grader=parse_grader_spec(args.grader) if args.grader else None,
            max_evidence_chars=args.max_evidence_chars,
            dry_run=True if args.dry_run else None,
            attach_preflight=args.preflight,
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
            max_rows=args.max_rows,
            max_total_model_calls=args.max_total_model_calls,
            max_paid_model_calls=args.max_paid_model_calls,
            min_qa_per_stratum=args.min_qa_per_stratum,
            max_total_cost_usd=args.max_total_cost_usd,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and report["status"] == "blocked" else 0
    if args.command == "external-indexes":
        report = build_external_indexes(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return external_index_exit_code(report, args)
    if args.command == "upstream-inputs":
        report = upstream_export_from_args(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
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
        model_pricing = _model_pricing_from_catalog(args.model_catalog)
        report = validate_run(
            load_experiment_config(args.config),
            args.runs,
            allow_errors=args.allow_errors,
            max_total_tokens=args.max_total_tokens,
            max_total_cost_usd=args.max_total_cost_usd,
            model_pricing=model_pricing,
            pricing_source=str(args.model_catalog) if args.model_catalog else None,
        )
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
    if args.command == "export-bundle":
        report = export_run_bundle(
            args.runs,
            output_path=args.output,
            qa_path=args.qa_path,
            mode=args.mode,
            grader_spec_path=args.grader_spec,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "import-pro-grades":
        report = import_pro_grades(args.runs, args.grades, output_path=args.output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2 if args.strict and not report["ok"] else 0
    if args.command == "analyze":
        model_pricing = catalog_pricing_payload(load_model_catalog(args.model_catalog)) if args.model_catalog else None
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
            model_pricing=model_pricing,
            pricing_source=str(args.model_catalog) if args.model_catalog else None,
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
    if args.command == "rag-audit":
        report = audit_retrievers(
            load_experiment_config(args.config),
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
        )
        if args.output:
            write_rag_audit(report, args.output)
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
        qa_coverage, qa_coverage_gate = _qa_coverage_from_args(config, args)
        report["qa_coverage"] = qa_coverage
        budget = _budget_from_args(args, report)
        if budget is not None:
            report["budget"] = budget
        if args.output:
            _write_json(args.output, report)
        if args.csv:
            write_csv(args.csv, report["conditions"])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if budget is not None and not budget["ok"]:
            return 2
        if qa_coverage_gate is not None and not qa_coverage_gate["ok"]:
            return 2
        return 2 if preflight_report is not None and not preflight_report["ok"] else 0
    if args.command == "sweep":
        config = _materialize_from_args(args)
        model_pricing = _model_pricing_from_catalog(args.model_catalog)
        pricing_coverage = pricing_coverage_for_config(config, model_pricing)
        run_dir = config.output_dir / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_output = args.config_output or run_dir / "materialized_config.json"
        write_experiment_config(config, config_output)

        plan_report = plan_experiment(config)
        qa_coverage, qa_coverage_gate = _qa_coverage_from_args(config, args)
        plan_report["qa_coverage"] = qa_coverage
        budget = _budget_from_args(args, plan_report)
        if budget is not None:
            plan_report["budget"] = budget
        if args.max_total_cost_usd is not None:
            plan_report["observed_cost_limit_usd"] = args.max_total_cost_usd
            plan_report["pricing_coverage"] = pricing_coverage
        plan_path = run_dir / "plan.json"
        qa_coverage_json = run_dir / "qa_coverage.json"
        qa_coverage_csv = run_dir / "qa_coverage.csv"
        _write_json(plan_path, plan_report)
        _write_json(qa_coverage_json, qa_coverage)
        write_csv(qa_coverage_csv, qa_coverage["strata"])
        if qa_coverage_gate is not None and not qa_coverage_gate["ok"]:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": "qa_coverage",
                        "config": str(config_output),
                        "plan_json": str(plan_path),
                        "qa_coverage_json": str(qa_coverage_json),
                        "qa_coverage_csv": str(qa_coverage_csv),
                        "qa_coverage_gate": qa_coverage_gate,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 2
        if budget is not None and not budget["ok"]:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": "budget",
                        "config": str(config_output),
                        "plan_json": str(plan_path),
                        "budget": budget,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 2
        if args.max_total_cost_usd is not None and not pricing_coverage["ok"]:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": "pricing_coverage",
                        "config": str(config_output),
                        "plan_json": str(plan_path),
                        "pricing_coverage": pricing_coverage,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 2

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
        summary = {
            "runs": str(runs_path),
            "rows": len(rows),
            "pricing_source": str(args.model_catalog) if args.model_catalog else None,
            "groups": summarize_rows(rows, model_pricing=model_pricing),
        }
        summary_json = run_dir / "summary.json"
        summary_csv = run_dir / "summary.csv"
        _write_json(summary_json, summary)
        write_csv(summary_csv, summary["groups"])
        leaderboard = leaderboard_rows(summary["groups"])
        leaderboard_json = run_dir / "leaderboard.json"
        leaderboard_csv = run_dir / "leaderboard.csv"
        _write_json(leaderboard_json, {"runs": str(runs_path), "rows": leaderboard})
        write_csv(leaderboard_csv, leaderboard)
        validation = validate_run(
            config,
            runs_path,
            allow_errors=args.allow_run_errors,
            max_total_cost_usd=args.max_total_cost_usd,
            model_pricing=model_pricing,
            pricing_source=str(args.model_catalog) if args.model_catalog else None,
        )
        validation_json = run_dir / "validation.json"
        _write_json(validation_json, validation)

        result = {
            "status": "complete" if validation["ok"] else "failed",
            "config": str(config_output),
            "plan_json": str(plan_path),
            "qa_coverage_json": str(qa_coverage_json),
            "qa_coverage_csv": str(qa_coverage_csv),
            "preflight": str(preflight_path),
            "runs": str(runs_path),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "leaderboard_json": str(leaderboard_json),
            "leaderboard_csv": str(leaderboard_csv),
            "validation_json": str(validation_json),
            "validation_ok": validation["ok"],
            "cost_coverage_ok": validation["cost"]["coverage_ok"],
            "total_cost_usd": validation["cost"]["total_cost_usd"],
            "rows": len(rows),
        }
        if not args.no_context_compare and "injected" in set(config.context_modes):
            context_comparisons = {}
            for candidate_mode, stem in [
                ("tool_explore", "context"),
                ("tool_search", "context-tool-search"),
                ("tool_native", "context-tool-native"),
            ]:
                if candidate_mode not in set(config.context_modes):
                    continue
                comparison = compare_conditions(
                    rows,
                    baseline_filter={"context_mode": "injected"},
                    candidate_filter={"context_mode": candidate_mode},
                    model_pricing=model_pricing,
                )
                comparison_without_pairs = {key: value for key, value in comparison.items() if key != "pairs"}
                context_compare_json = run_dir / f"{stem}-compare.json"
                context_compare_csv = run_dir / f"{stem}-compare.csv"
                context_pairs_csv = run_dir / f"{stem}-pairs.csv"
                _write_json(context_compare_json, comparison_without_pairs)
                write_csv(context_compare_csv, comparison_without_pairs["metrics"])
                write_csv(context_pairs_csv, flatten_pairs(comparison["pairs"]))
                context_comparisons[candidate_mode] = {
                    "json": str(context_compare_json),
                    "csv": str(context_compare_csv),
                    "pairs_csv": str(context_pairs_csv),
                    "matched_pairs": comparison_without_pairs["matched_pairs"],
                }
                if candidate_mode == "tool_explore":
                    result.update(
                        {
                            "context_compare_json": str(context_compare_json),
                            "context_compare_csv": str(context_compare_csv),
                            "context_pairs_csv": str(context_pairs_csv),
                            "matched_context_pairs": comparison_without_pairs["matched_pairs"],
                        }
                    )
            if context_comparisons:
                result["context_comparisons"] = context_comparisons
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if validation["ok"] else 2
    raise AssertionError(args.command)


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-rows", type=int, help="Fail when the materialized plan exceeds this many run rows.")
    parser.add_argument("--max-total-model-calls", type=int, help="Fail when logical answer+judge model calls exceed this limit.")
    parser.add_argument("--max-paid-model-calls", type=int, help="Fail when estimated paid model calls exceed this limit.")


def _add_qa_coverage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-qa-per-stratum",
        type=_positive_int,
        help="Fail unless each available refusal/figure/reference stratum has at least this many selected QA items.",
    )


def _add_materialize_args(parser: argparse.ArgumentParser, *, include_output: bool) -> None:
    if include_output:
        parser.add_argument("--output", type=Path, help="Write the materialized config to this path; stdout when omitted.")
    parser.add_argument("--name", help="Override experiment name.")
    parser.add_argument("--limit", type=int, help="Override dataset question limit.")
    parser.add_argument("--qa-ids", help="Comma-separated QA IDs to run.")
    parser.add_argument("--qa-ids-file", type=Path, help="JSON/list/newline file of QA IDs to run.")
    parser.add_argument("--retrievers", help="Comma-separated retriever names to keep.")
    parser.add_argument("--retrievers-file", type=Path, help="Replace retriever matrix from JSON generated by retriever-matrix.")
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
    parser.add_argument("--dry-run", action="store_true", help="Materialize a config that never calls answer or judge models.")
    parser.add_argument("--ready-only", action="store_true", help="Drop retrievers/models not ready under preflight.")
    parser.add_argument("--no-external-checks", action="store_true", help="Do not run external checks for --ready-only and sweep preflight.")
    parser.add_argument("--allow-not-checked", action="store_true", help="Keep not_checked retrievers during --ready-only.")
    parser.add_argument("--timeout-s", type=int, default=30, help="Timeout per external adapter check.")


def _materialize_from_args(args: argparse.Namespace):
    qa_ids = _qa_ids_from_args(args)
    models = _models_from_args(args)
    retrievers = _retrievers_from_args(args)
    config = materialize_config(
        load_experiment_config(args.config),
        name=args.name,
        limit=args.limit,
        qa_ids=qa_ids,
        retrievers=retrievers,
        retriever_names=parse_csv(args.retrievers),
        drop_retriever_names=parse_csv(args.drop_retrievers),
        context_modes=parse_csv(args.context_modes),
        models=models,
        grader=parse_grader_spec(args.grader) if args.grader else None,
        max_evidence_chars=args.max_evidence_chars,
        dry_run=True if args.dry_run else None,
    )
    if args.ready_only:
        config, _ = filter_ready_config(
            config,
            check_external=not args.no_external_checks,
            timeout_s=args.timeout_s,
            allow_not_checked=args.allow_not_checked,
        )
    return config


def _budget_from_args(args: argparse.Namespace, plan: dict):
    return evaluate_plan_budget(
        plan,
        max_rows=getattr(args, "max_rows", None),
        max_total_model_calls=getattr(args, "max_total_model_calls", None),
        max_paid_model_calls=getattr(args, "max_paid_model_calls", None),
    )


def _qa_coverage_from_args(config, args: argparse.Namespace):
    coverage = qa_coverage_for_selection(
        config.dataset.qa_path,
        limit=config.dataset.limit,
        qa_ids=config.dataset.qa_ids,
    )
    gate = evaluate_qa_coverage(
        coverage,
        min_selected_per_stratum=getattr(args, "min_qa_per_stratum", None),
    )
    if gate is not None:
        coverage["gate"] = gate
    return coverage, gate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def _model_pricing_from_catalog(path: Path | None):
    return catalog_pricing_payload(load_model_catalog(path)) if path else None


def _models_from_args(args: argparse.Namespace):
    inline = getattr(args, "model", None)
    file_path = getattr(args, "models_file", None)
    if inline and file_path:
        raise ValueError("--model and --models-file are mutually exclusive")
    if file_path:
        return load_model_specs_file(file_path)
    return [parse_model_spec(spec) for spec in inline] if inline else None


def _retrievers_from_args(args: argparse.Namespace):
    retriever_names = parse_csv(getattr(args, "retrievers", None))
    file_path = getattr(args, "retrievers_file", None)
    if retriever_names and file_path:
        raise ValueError("--retrievers and --retrievers-file are mutually exclusive")
    if file_path:
        return load_retriever_specs_file(file_path)
    return None


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
