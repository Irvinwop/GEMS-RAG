# GEMS-RAG

Harness workspace for running RAG ablation experiments across model providers, retrieval strategies, and grading configurations.
The `gems-rag` CLI normalizes its working directory to the repository root, so tracked configs can use repo-relative paths from any launch directory; use absolute paths for files outside the harness workspace.

Start the local model picker with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli gui
```

The GUI opens at `http://127.0.0.1:8765/` as one page with checkboxes for every configured RAG, context-delivery mode, and RAG-capable text or vision model. Its question-source control defaults to the 150 immutable MUTCD-150 questions from the updated MRAG clone and also offers the 49 curated gold question/answer records from the supplied snapshot. MUTCD-150 is explicitly question-only: generated upstream answers are never treated as gold, grading ZIP rows carry `has_gold_answer=false`, and the authoritative MUTCD PDF is included for the final grader. The gold-reference oracle is disabled for that dataset. RAGs are segmented by interaction type, and incompatible delivery modes are disabled instead of producing invalid matrices. The RAG inference profile separately records the chat, embedding, and vision models used internally by model-assisted retrievers; it can target OpenAI or a keyless local OpenAI-compatible endpoint without changing the answer-model matrix or grader. Run setup includes the ignored output folder and ZIP filename; **Run / resume** appends one fsynced JSONL row per completed condition, skips existing condition keys after interruption, repairs a truncated final row, blocks concurrent writers, and creates the named GPT Pro ZIP when the matrix finishes. The browser restores the setup and progress after reload. The answer-model catalog covers current and prior OpenAI, Anthropic, xAI, and Qwen sizes, the manuscript's historical Qwen VLMs, and local OpenAI-compatible aliases. Media generators, speech-only or realtime models, embeddings, rerankers, and retired dated snapshots are intentionally excluded. API tokens are stored in the ignored `.env` file with mode `0600`; their values are never returned by the API. GraphRAG reuses `OPENAI_API_KEY` under the OpenAI RAG profile instead of requiring a separate service key. The credential-bearing server accepts loopback connections only.

Verify the actual MUTCD manual and every derived evaluation artifact with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli manual-status \
  --output data/working/manual/manifest.json \
  --strict
```

The manifest records the PDF checksum and page count, extracted and canonical chunk counts, page renders, figures, graph, gold QA, and the ingestion lineage for all manuscript RAGs. Controlled ablations use one canonical PDF-derived corpus by default. `external-indexes --ingestion-mode native_pdf` switches PaperQA2 and RAG-Anything to their upstream raw-PDF parsers; MegaRAG already consumes the PDF directly, and VisRAG consumes its complete page render. Methods without an upstream PDF parser remain on the verified shared corpus so parser differences are not confused with retrieval differences.

Local-only inputs are intentionally ignored:

- `data/raw/` stores downloaded datasets and archives.
- `manuscript-draft/` stores the current manuscript draft.
- `external/MRAG_stp2/` stores the cloned reference implementation.
- `external/rag-implementations/` stores cloned comparison RAG repositories.

The harness makes it cheap to compare:

- automatic context injection versus model-driven data exploration through a two-step search/open tool loop
- different RAG pipelines, including dependency-free in-memory and Qdrant-backed local vector baselines
- model families and sizes across Anthropic, Grok, OpenAI, Qwen, and local runners
- grader configurations, with GPT-5.6 Sol at `xhigh` as the current quality-first default

See [docs/implementation-inventory.md](docs/implementation-inventory.md) for the current local data, reference implementation, and cloned external RAG inventory.

Local MRAG metadata repairs can be checked or re-applied with:

```bash
.venv/bin/python scripts/repair_mrag_metadata.py --dry-run
```

External RAG input corpora can be exported directly when you want to inspect them:

```bash
python3 scripts/export_mrag_corpus.py
```

Inspect the default MUTCD-150 source, or summarize and slice the curated gold file before a paid gold-referenced sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli inspect
PYTHONPATH=src .venv/bin/python -m gems_rag.cli qa-summary \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl
PYTHONPATH=src .venv/bin/python -m gems_rag.cli qa-split \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
  --size 12 \
  --seed 20260708 \
  --strategy balanced \
  --output data/working/qa-splits/balanced-12.json
```

The balanced split strategy cycles across refusal, figure-grounding, and referenced/unreferenced strata so small sweeps do not only test the first contiguous rows in `gold_qa.jsonl`.

Import the downloaded MRAG prior generated/scored runs into the harness schema:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli import-mrag-eval --overwrite --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli validate configs/mrag-prior-eval.json \
  --runs runs/mrag-prior-eval/runs.jsonl \
  --max-total-tokens 500000 \
  --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/mrag-prior-eval/runs.jsonl \
  --output-dir runs/mrag-prior-eval/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis model \
  --baseline qwen3-vl-flash
```

This preserves the prior Qwen VLM answers and judge scores while enriching them with local chunk, figure, and page evidence from the extracted MRAG cache.
Run rows retain answer-model metadata in `model_raw` and grader metadata in `grader_raw` so imported and newly generated answers can be audited or regraded later.
`analyze` writes `analysis.json`, `summary.*`, and one metrics/pairs comparison set for every observed non-baseline model under the selected axis. With `--qa-path`, it also writes `strata-summary.csv` and `strata-comparisons.csv` for refusal, figure-backed, reference-backed, reference-count, reference-content-type, and question-type slices.
`configs/mrag-prior-eval.json` is for structural validation and comparison of imported historical rows; preflight will still report missing Qwen credentials unless `DASHSCOPE_API_KEY` is configured for fresh model calls.
Use `gems-rag validate --max-total-tokens N --strict` after paid runs to fail CI or shell scripts when observed answer plus judge token usage exceeds the run budget.

Run the local smoke matrix with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/smoke.local.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli run configs/smoke.local.json --overwrite
PYTHONPATH=src .venv/bin/python -m gems_rag.cli validate configs/smoke.local.json --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli manuscript-coverage
```

External adapter indexes and heavyweight package environments are local and ignored. Bootstrap the currently supported upstream environments with:

```bash
scripts/bootstrap_external_envs.sh
```

By default this installs the lighter command-backed adapters and prepares GraphRAG, VisRAG manifests, and PaperQA deferred chunks. To also build isolated heavy dependency envs for MegaRAG, GFM-RAG, DPR, MRAG reference, HippoRAG, and VisRAG, run:

```bash
BOOTSTRAP_HEAVY_RAGS=1 scripts/bootstrap_external_envs.sh
```

The heavy wrappers automatically re-run themselves under their adapter-specific `data/working/venvs/<adapter>/bin/python` interpreters when those ignored envs exist. MegaRAG keeps its required LightRAG `v1.4.3` separate from the newer standalone LightRAG baseline. MegaRAG, HippoRAG, and VisRAG use native Python 3.12 retrieval environments on Apple Silicon. Every API-backed index wrapper publishes a source/config-bound completion marker only after upstream indexing succeeds, and PaperQA2 replaces its pickle atomically. HippoRAG lazily loads optional CUDA backends, while VisRAG checkpoints every encoded page batch and keeps its large retriever warm across queries; interrupted indexing can be continued without treating partial state as ready.

Then build whatever command-backed external indexes are possible in the current environment. The setup builder exports shared MRAG corpus inputs before corpus-backed adapters index:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes --dry-run
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes \
  --config data/working/ablation-bundles/local-policy-small-medium/materialized_config.json \
  --dry-run
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The builder runs adapter readiness checks, skips missing heavy environments instead of failing the whole setup, and emits JSON with `query_ready`, `needs_index`, `needs_environment`, `needs_model_service`, `check_only_not_ready`, and a per-adapter `setup_plan` in addition to the lower-level `built`, `already_ready`, `would_run`, `skipped`, and `failed` lists. Local OpenAI-compatible checks probe `<base-url>/models`; a dummy key no longer makes an adapter appear ready when the endpoint is down or rejects authorization. GraphRAG writes the same base URL into the generated completion and embedding `api_base` settings. Use `--config path/to/materialized_config.json` to derive the subset from command-backed retrievers in a prepared sweep, `--only graphrag,lightrag` for a manual subset, `--force` to rebuild ready adapters, and `--strict-skips` when skipped adapters should fail CI. The legacy `scripts/build_external_indexes.py` entrypoint delegates to the same package code.
When `--config` references a materialized GUI run, the setup builder inherits its complete `rag_backend` profile, including endpoint, credential mode, chat model, embedding model and dimensions, and vision model. Backend-dependent completion markers prevent indexes built with a different profile from being reported ready.

Self-RAG and CRAG can consume harness retrieval results through upstream-compatible eval input exports:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli upstream-inputs \
  --retriever-kind bm25_graph \
  --top-k 10 \
  --out-dir data/working/upstream_eval_inputs
```

This writes ignored `selfrag_input.jsonl` and CRAG `question [SEP] passage` files under `data/working/upstream_eval_inputs/`, plus a manifest recording the retriever, row counts, upstream repo entrypoint checks, and the Self-RAG/CRAG command arrays to run next. The legacy `scripts/export_upstream_eval_inputs.py` entrypoint delegates to the same package command.

For one-off debugging, the underlying index commands are:

```bash
.venv/bin/python scripts/query_graphrag_index.py prepare --force
.venv/bin/python scripts/query_graphrag_index.py init
.venv/bin/python scripts/query_graphrag_index.py index
.venv/bin/python scripts/query_lightrag_index.py index
.venv/bin/python scripts/query_raganything_index.py index
.venv/bin/python scripts/query_hipporag_index.py index
.venv/bin/python scripts/query_visrag_index.py prepare --scope pages
.venv/bin/python scripts/query_visrag_index.py index
.venv/bin/python scripts/query_paperqa_index.py index --defer-embedding
.venv/bin/python scripts/query_megarag_index.py prepare
.venv/bin/python scripts/query_megarag_index.py index
.venv/bin/python scripts/prepare_lpkg_plans.py normalize --predictions /path/to/generated_predictions.jsonl
.venv/bin/python scripts/prepare_lpkg_plans.py check
```

Then use `configs/external-rag.template.json` as the starting point for command-backed external runs.
Preflight an ablation config before spending model calls:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/ablation.template.json
```

The preflight report estimates run rows and lists dataset, retriever, model, grader, credential, and external-adapter blockers.
Smoke-test each selected RAG in every context mode that RAG actually supports with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli rag-audit configs/ablation.template.json \
  --output data/working/audits/ablation-rags.json
```

The report keeps readiness blockers separate from execution failures. Query-driven RAGs support all four modes; the LPKG planner and gold-reference oracle support injected/explore modes; the no-retrieval control supports injection only.
Materialize a smaller concrete ablation config without editing JSON by hand:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli materialize configs/ablation.template.json \
  --output configs/generated/local-tool-explore.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only
```

Model and grader specs use `provider:model[,key=value...]`. For large provider sweeps, put one model spec per line in a file like `configs/model-matrix.example.txt` and pass `--models-file path/to/models.txt` to `materialize`, `plan`, or `sweep`; edit any local endpoint aliases to match your server before running paid calls. OpenAI entries can set `api=responses` and `reasoning_effort=low|medium|high|xhigh`; unresolved model placeholders such as `replace-with-*`, `*-placeholder`, and `*-or-successor` are blocked by preflight.
Set `vision=true` only for models that accept image input. Retrieved page and figure files are then attached to final answer calls in every context mode and to LLM grader calls; `tool_explore`, `tool_search`, and `tool_native` attach only evidence the model opened. `max_images` defaults to 5, `max_image_bytes` defaults to 20 MiB per file, and `image_detail=low|high|auto` is optional. Text-only fallbacks are explicit in `model_raw.image_input` and `grader_raw.model_raw.image_input`, so a visual run cannot silently become a caption-only run.
`--ready-only` prunes blocked retrievers and models after preflight; it still fails if the dataset, context modes, or grader are blocked.
To generate a matrix from provider, size, role, and tag metadata instead of hand-editing long lists:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli model-matrix \
  configs/model-catalog.example.json \
  --providers openai,anthropic,xai,qwen,local_openai \
  --sizes tiny,small,medium \
  --output data/working/model-matrices/provider-tiny-small-medium.txt
```

The catalog defaults merge shared options like `temperature=0`, provider options like OpenAI `api=responses`, cloud-model `vision=true`, a local OpenAI-compatible `base_url`, and per-model overrides. Local aliases remain `vision=false` until the served checkpoint is known to accept image input. Its OpenAI tiers follow the current [GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model): Luna for efficient high-volume runs, Terra for smaller balanced runs, and Sol for frontier and grading work. Catalog entries may also include non-runtime `pricing` metadata such as `input_per_1m` and `output_per_1m` in USD; keep those values account-current before paid sweeps. Use the generated file with `--models-file`, or pass `--roles grader --format json` to inspect the current final-grader entry before selecting or editing it.
`prepare-ablation --grader-from-catalog --grader-providers openai --grader-sizes judge` selects exactly one enabled `roles=["grader"]` entry from the same catalog and persists it into the materialized config; add `--grader-tags final` or similar when the catalog has multiple judge candidates, or `--include-disabled-graders` when testing a disabled backup judge.
External retriever mode matrices can be generated the same way:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli retriever-matrix \
  configs/retriever-catalog.example.json \
  --families graphrag,lightrag,raganything \
  --modes local,hybrid \
  --output data/working/retriever-matrices/external-local-hybrid.json
```

Use the generated JSON with `--retrievers-file` on `materialize`, `plan`, or `sweep`. The catalog includes local baselines, every manuscript method, Self-RAG/CRAG policy variants, the MRAG reference wrapper, and external mode variants for GraphRAG, LightRAG, MegaRAG, RAG-Anything, HippoRAG, VisRAG, and PaperQA2.
To write the QA split, QA coverage report, model matrix, retriever matrix, materialized config, and plan in one ignored bundle:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli prepare-ablation configs/ablation.template.json \
  --name local-policy-small-medium \
  --qa-size 12 \
  --qa-seed 20260708 \
  --model-providers openai,anthropic,xai,qwen,local_openai \
  --model-sizes small,medium \
  --retriever-families local,self_rag_policy,crag_policy \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --grader-from-catalog \
  --grader-providers openai \
  --grader-sizes judge \
  --min-qa-per-stratum 1 \
  --max-total-cost-usd 5 \
  --dry-run \
  --output-dir data/working/ablation-bundles/local-policy-small-medium
```

The bundle report includes exact follow-up commands for external index setup when command-backed retrievers are selected, upstream Self-RAG/CRAG input exports when policy retrievers are selected, preflight, sweep, resume, retrying error rows, strict validation, and context-mode analysis. It snapshots the source model catalog as `model_catalog.json`; generated sweep, validation, and analysis commands use that snapshot so observed-cost calculations remain tied to the pricing metadata used when the bundle was prepared. `--max-total-cost-usd 5` propagates a post-run ceiling to those sweep and validation commands. The bundle also writes `qa_coverage.json` and `qa_coverage.csv`, comparing selected QA IDs against the full gold set across refusal, figure, and reference strata before a paid run starts. `--min-qa-per-stratum 1` makes that report a launch gate in `prepare-ablation`, `plan`, and `sweep`; every observed refusal x figure x reference stratum must be represented, and `sweep` exits before preflight or model calls when it is not. `--dry-run` preserves the intended model and grader labels but forces dry-run answer generation and skips non-heuristic grader calls; plans still show logical model calls and report `paid_model_calls: 0`.
Plan the exact row matrix and model-call count before launching a sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli plan configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --model-catalog configs/model-catalog.example.json \
  --max-total-cost-usd 5 \
  --ready-only \
  --output runs/local-tool-explore/plan.json \
  --csv runs/local-tool-explore/plan.csv
```

Use `--retrievers-file data/working/retriever-matrices/external-local-hybrid.json` in place of `--retrievers ...` when planning generated external mode matrices.

`tool_explore` rows estimate two logical answer-model calls per row: one selection call plus one answer call. `tool_search` rows estimate three logical answer-model calls per row: one search-query call, one open-selection call, and one answer call. `tool_native` reserves the configured `tool_max_rounds` plus one forced final-answer call (five calls by default); observed-cost validation uses the actual provider-call count recorded on each row. Non-heuristic graders add one logical judge-model call per row. `paid_model_calls` excludes `dry_run` model rows, heuristic grading, and full-config `dry_run: true`.
Use `--max-rows`, `--max-total-model-calls`, or `--max-paid-model-calls` on `plan`, `prepare-ablation`, or `sweep` to make oversized matrices fail before a paid run starts.
Run the same materialization as an end-to-end sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli sweep configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only \
  --overwrite
```

`sweep` writes `materialized_config.json`, `preflight.json`, `runs.jsonl`, `summary.*`, `leaderboard.*`, and context comparison artifacts under `runs/<experiment-name>/` when `injected` is paired with any tool mode.
It also writes `validation.json`, which checks expected row completeness, duplicate rows, unexpected rows, invalid JSON lines, retrieval/model/judge error counts, incomplete judge-score rubrics, stale grader labels, token ceilings, and observed USD cost ceilings. A cost ceiling passes only when every expected paid answer and judge call has complete usage plus catalog pricing; missing or partial usage fails closed instead of undercounting. `tool_explore` aggregates selection and answer usage, `tool_search` aggregates search-plan, selection, and answer usage, and `tool_native` aggregates every provider continuation in its actual function-call loop. Each mode preserves raw calls for audit. Explicit zero-priced local catalog entries remain valid without provider usage metadata. Retriever build failures, retrieval exceptions, model build/generation exceptions, and grader exceptions are recorded on individual rows so a broken external adapter does not abort the whole sweep. Use `--allow-run-errors` only for best-effort sweeps where failed rows should not make the command exit non-zero.
After fixing a broken index, credential, adapter command, stale grader label, or incomplete judge-score row, rerun only repairable rows while keeping clean rows:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli sweep configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only \
  --retry-errors
```

For larger matrices, run `analyze` over the finished `runs.jsonl` to emit a reusable report directory, ranked leaderboard, and repeated matched-pair comparisons across any axis:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/local-tool-explore/runs.jsonl \
  --output-dir runs/local-tool-explore/analysis \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
  --model-catalog configs/model-catalog.example.json \
  --axis context_mode \
  --baseline injected
```

Summary, leaderboard, and comparison metrics include grader scores, row error rates, answer/judge token usage when providers return it, observed answer/judge cost when `--model-catalog` supplies pricing, and tool-use diagnostics such as selected hits, opened hits, search-query count, unique search results, search errors, and parse failures. The leaderboard ranks condition groups by mean judge score, then row error rate, then observed cost/tokens.

When the final judge model changes, regrade an existing run without rerunning retrieval or answer generation:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli regrade configs/ablation.template.json \
  --runs runs/local-tool-explore/runs.jsonl \
  --output runs/local-tool-explore/regraded-final-judge.jsonl \
  --grader openai:<final-judge-model> \
  --strict
```

Use `--only-missing` to fill only rows with missing `judge_scores` or an existing `judge_error`. The command refuses in-place output so the original `runs.jsonl` remains intact.

Run outputs can also be stored in a redacted ZIP, including a self-contained workflow for grading through a GPT Pro subscription instead of the API:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli export-bundle \
  runs/local-tool-explore/runs.jsonl \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
  --mode gpt_pro \
  --output data/working/bundles/local-tool-explore-gpt-pro.zip

PYTHONPATH=src .venv/bin/python -m gems_rag.cli import-pro-grades \
  runs/local-tool-explore/runs.jsonl \
  path/to/grades.jsonl \
  --output runs/local-tool-explore/gpt-pro-graded-runs.jsonl \
  --strict
```

The ZIP contains deduplicated source records in `qa_pairs.jsonl`; every row-specific `grading_tasks.jsonl` object carries the question, `has_gold_answer`, available gold answer/references/figures, RAG answer, and retrieved evidence. Question-only bundles also include `source/mutcd-manual.pdf`; upstream generated answers are excluded from the gold fields. The ZIP also includes visual evidence when present, `GRADING.md`, `grades.template.jsonl`, source and manual checksums in `manifest.json`, and sanitized run artifacts. API keys and authorization fields are redacted. The importer accepts either `grades.jsonl` directly or a ZIP containing it and preserves the original answers and evidence.

Use the one-question external smoke config to verify command-backed adapter failure/success reporting without running the full external matrix:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/external-rag.smoke.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli run configs/external-rag.smoke.json --overwrite
```

The cloned MRAG reference implementation can be checked with:

```bash
.venv/bin/python scripts/query_mrag_reference.py check --mode full
.venv/bin/python scripts/query_mrag_reference.py retrieve --mode full --question "What does Section 2A.04 require?"
.venv/bin/python scripts/query_mrag_reference.py stop
```

MRAG retrieval calls reuse an ignored local worker by default, so model weights stay warm across questions and resumed runs. Pass `--no-persistent` to `retrieve` for a one-shot process; the default worker shuts down after 30 idle minutes.

Grading behavior:

- Smoke configs use the deterministic `heuristic` grader for cheap regression checks.
- Ablation configs use an LLM grader through `openai_compatible` or `litellm`.
- The LLM grader uses gold answers/references when present; question-only rows are marked explicitly and graded from authoritative evidence instead of receiving false zero scores. Every rubric key is normalized in `judge_scores`.

Context modes:

- `injected`: the runner directly places retrieved evidence into the answer prompt.
- `tool_explore`: a structured multi-prompt simulation where the runner asks the model to choose hit IDs from a catalog, opens only those selected hits, and then asks the model to answer from the opened results.
- `tool_search`: a structured multi-prompt simulation where the model first emits search-query JSON, the harness runs those searches, the model emits hit IDs to open, and a final prompt contains only opened results.
- `tool_native`: the model receives no automatic context and explores the same retriever through real provider `search` and `open` function calls. Search returns bounded metadata and short previews; only open returns bounded evidence text. Set per-model `tool_max_rounds` in model options to change the default four tool rounds.

Model provider aliases:

- `openai`: OpenAI-compatible client, `OPENAI_API_KEY`.
- `anthropic`: LiteLLM client, `ANTHROPIC_API_KEY`.
- `xai` / `grok`: OpenAI-compatible client, `XAI_API_KEY`, default base URL `https://api.x.ai/v1`.
- `qwen`: OpenAI-compatible DashScope endpoint, `DASHSCOPE_API_KEY`; override the default endpoint with `DASHSCOPE_BASE_URL` or a per-model `base_url`.
- `local_openai`: local OpenAI-compatible endpoint, defaults to `http://localhost:8000/v1` and uses a dummy local key unless overridden.

All command-backed adapters, including the local vector DB command wrapper and cloned external RAGs, can be checked with:

```bash
.venv/bin/python scripts/check_external_adapters.py
.venv/bin/python scripts/check_external_adapters.py --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The checker separates query-ready adapters from environment-ready adapters that still need provider credentials or a local index. The current GraphRAG shim uses an ignored Python 3.13 environment at `data/working/venvs/graphrag/` when it exists because upstream GraphRAG requires Python `<3.14`.
For external adapters pointed at a local OpenAI-compatible server, GraphRAG, HippoRAG, LightRAG, MegaRAG, RAG-Anything, and PaperQA2 checks accept `--allow-missing-api-key` and use a dummy `local` key for clients that require an API-key field. The configured endpoint must still be reachable and authorize the probe.
Use `configs/external-rag.local-openai.smoke.json` to preflight those local-compatible command adapters with matching `check_command` settings.
Command-backed adapters may emit JSON `evidence`, `chunks`, `figures`, `pages`, or `contexts`; the harness preserves visual/page metadata such as image paths, figure IDs, and PDF/printed page numbers. The GraphRAG, HippoRAG, LightRAG, RAG-Anything, and PaperQA2 configs pass `{top_k}` through to upstream retrieval budgets or structured context caps.
External command templates can use `{question}`, `{qa_id}`, `{mrag_dir}`, and `{top_k}` placeholders; in `tool_search` and `tool_native`, `{top_k}` follows the model-requested search budget for that query.

Summarize an ablation run with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/smoke-local/runs.jsonl \
  --output-dir runs/smoke-local/analysis \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
  --axis context_mode \
  --baseline injected
```

Search the local Qdrant vector DB baseline directly with:

```bash
.venv/bin/python scripts/query_vector_db.py check
.venv/bin/python scripts/query_vector_db.py search --question "What does Section 2A.04 require?"
```

The retriever catalog also includes `qdrant_hash_vector_command`, which runs the same vector DB through the `external_command` boundary and emits harness-native `evidence` rows.
