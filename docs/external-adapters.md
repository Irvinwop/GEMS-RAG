# External RAG Adapter Plan

The cloned upstream implementations live under `external/rag-implementations/` and remain gitignored. The harness should interact with them through adapters, not by copying their source into this repo.

## Shared Corpus Exports

Run this directly when inspecting or debugging adapter input files:

```bash
python3 scripts/export_mrag_corpus.py
```

Outputs under `data/working/mrag_corpus/`:

- `chunks.jsonl`: one canonical text document per repaired MRAG chunk.
- `lightrag_corpus.txt`: concatenated text corpus for LightRAG-style bulk insertion.
- `raganything_content_list.json`: mixed text/image/table content list for RAG-Anything.
- `manifest.json`: counts and paths.

These exports are ignored because they are derived from ignored data. `gem-rags external-indexes` also runs this exporter automatically before corpus-backed adapters index.

## Current Harness Adapters

- `bm25`: local lexical retrieval over repaired MRAG chunks.
- `hash_vector`: dependency-free local vector search using hashed token-count vectors. This is a plain vector-control baseline, not a semantic embedding model.
- `qdrant_hash_vector`: embedded Qdrant vector database baseline using the same deterministic hashed vectors, persisted under `data/working/qdrant_hash_vector/`.
- `qdrant_hash_vector_command`: command-backed wrapper over the same embedded Qdrant baseline, useful when the vector database should be exercised through the `external_command` adapter boundary.
- `bm25_graph`: local lexical retrieval plus repaired NetworkX graph expansion.
- `oracle`: upper-bound retrieval using gold reference chunks from the QA file.
- `self_rag_policy`: Self-RAG-style retrieval-control policy with `no_retrieval`, `always_retrieve`, and `adaptive_retrieval` modes over an existing retriever.
- `crag_policy`: CRAG-style corrective policy that evaluates primary retrieval quality and chooses accept, fallback, or merge/refine.
- `external_placeholder`: keeps external systems visible in experiment matrices before their indexes exist.
- `external_command`: runs a preexisting indexed RAG system through a command template and captures stdout as tool evidence. JSON stdout can include `evidence`, `chunks`, `figures`, `pages`, or `contexts`; visual/page metadata such as image paths and page numbers is preserved for multimodal adapters. Commands run from the harness repository root by default; set `options.cwd` when an upstream wrapper needs a different working directory.

## Context Modes

- `injected`: the selected retriever's evidence text is placed directly into the answer prompt.
- `tool_explore`: the selected retriever first produces a hit catalog. The model gets only that catalog, returns JSON `open_hit_ids`, and the runner opens only those IDs for the final answer prompt. Runs record `retrieval_debug.context_debug.selected_ids` and `opened_ids`.
- `tool_search`: the model gets no automatic evidence. It first returns JSON search queries, the harness runs those queries against the selected retriever, then the model chooses which returned hit IDs to open before answering. This is the plain tool-call exploration ablation against the same retriever/index used by `injected`.

## Implemented External Shims

MRAG reference implementation:

```bash
.venv/bin/python scripts/query_mrag_reference.py check
.venv/bin/python scripts/query_mrag_reference.py retrieve --question "What does Section 2A.04 require?"
```

This wraps the cloned `hannanazad/MRAG_stp2` retriever and points it at the repaired extracted MRAG directory with `MRAG_BASE_DIR`. It needs the heavy retrieval stack from `external/MRAG_stp2/requirements.txt` (`torch`, `FlagEmbedding` or `sentence-transformers`, and a reranker) before it can run.

GraphRAG:

```bash
.venv/bin/python scripts/query_graphrag_index.py check
.venv/bin/python scripts/query_graphrag_index.py prepare --force
.venv/bin/python scripts/query_graphrag_index.py init
.venv/bin/python scripts/query_graphrag_index.py index --method standard
.venv/bin/python scripts/query_graphrag_index.py query --method local --top-k 6 --json --question "What does Section 2A.04 require?"
```

This uses Microsoft GraphRAG's cloned Typer CLI through the source tree. `prepare` writes exported MRAG chunks to `data/working/graphrag_index/input/mutcd_chunks.txt`; GraphRAG still owns its normal `init`, `index`, and `query` phases. When `data/working/venvs/graphrag/bin/python` exists, the shim uses it automatically. Override with `GRAPHRAG_PYTHON=/path/to/python` or `--python /path/to/python`. The generated GraphRAG config expects `GRAPHRAG_API_KEY` by default.
Use `--allow-missing-api-key` when the generated GraphRAG settings point at a local OpenAI-compatible endpoint that accepts a dummy key.
In JSON mode, the shim calls GraphRAG's upstream query helpers inside the isolated GraphRAG interpreter, captures `context_data`, and emits harness `contexts` rows capped by `--top-k`.

LightRAG:

```bash
.venv/bin/python scripts/query_lightrag_index.py index
.venv/bin/python scripts/query_lightrag_index.py query --mode hybrid --top-k 6 --chunk-top-k 6 --only-need-context --question "What does Section 2A.04 require?"
```

The tracked LightRAG retriever configs pass the harness `{top_k}` budget to both `--top-k` and `--chunk-top-k` so entity/relationship and chunk retrieval budgets move together during ablations.

For a local OpenAI-compatible server:

```bash
.venv/bin/python scripts/query_lightrag_index.py check \
  --base-url http://localhost:8000/v1 \
  --allow-missing-api-key
```

RAG-Anything:

```bash
.venv/bin/python scripts/query_raganything_index.py index
.venv/bin/python scripts/query_raganything_index.py query --mode hybrid --top-k 6 --chunk-top-k 6 --only-need-context --json --question "What does Section 2A.04 require?"
```

The tracked RAG-Anything retriever configs pass the harness `{top_k}` budget to both `--top-k` and `--chunk-top-k`. With `--only-need-context --json`, the shim emits the retrieved LightRAG context as a `contexts` evidence row instead of asking RAG-Anything to generate a final answer before the harness model sees it.

For a local OpenAI-compatible server:

```bash
.venv/bin/python scripts/query_raganything_index.py check \
  --base-url http://localhost:8000/v1 \
  --allow-missing-api-key
```

HippoRAG:

```bash
.venv/bin/python scripts/query_hipporag_index.py check
.venv/bin/python scripts/query_hipporag_index.py index
.venv/bin/python scripts/query_hipporag_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

This wraps HippoRAG 2's `HippoRAG.index(...)` and `HippoRAG.retrieve(...)` methods over `chunks.jsonl`. It requires the HippoRAG dependency stack (`torch`, `transformers`, `python_igraph`, OpenAI/LiteLLM clients, and an embedding model endpoint).
Indexing also writes an ignored `mrag_chunk_manifest.jsonl` sidecar under the HippoRAG save directory. Query uses that sidecar, or falls back to exported `chunks.jsonl`, to emit chunk contexts with MRAG `doc_id`, section, page, title, and content-type metadata instead of anonymous text hits.

VisRAG:

```bash
.venv/bin/python scripts/query_visrag_index.py check
.venv/bin/python scripts/query_visrag_index.py prepare --scope pages
.venv/bin/python scripts/query_visrag_index.py index
.venv/bin/python scripts/query_visrag_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

This wraps the cloned OpenBMB VisRAG repository at the `VisRAG-Ret` retrieval boundary. `prepare` builds an ignored manifest over MRAG page images, or figure/table crops with `--scope figures`/`--scope both`. `index` follows the upstream `AutoModel`/`AutoTokenizer` weighted-mean-pooling recipe for `openbmb/VisRAG-Ret` and saves embeddings under `data/working/visrag_index/`. It requires the VisRAG visual model dependency stack (`torch`, `transformers`, `Pillow`, `numpy`) and local or downloadable model weights.

PaperQA2:

```bash
.venv/bin/python scripts/query_paperqa_index.py check
.venv/bin/python scripts/query_paperqa_index.py index --defer-embedding
.venv/bin/python scripts/query_paperqa_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

The tracked PaperQA2 retriever configs pass the harness `{top_k}` budget into `Settings.answer.evidence_k` and `answer_max_sources`, and the shim emits JSON-safe `contexts` rows with PaperQA summary text plus source chunk metadata.

Use `--allow-missing-api-key` before the subcommand when PaperQA2 is configured to use a local OpenAI-compatible endpoint that accepts a dummy key:

```bash
.venv/bin/python scripts/query_paperqa_index.py --allow-missing-api-key check
```

Self-RAG and CRAG upstream input exports:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli upstream-inputs \
  --retriever-kind bm25_graph \
  --top-k 10 \
  --out-dir data/working/upstream_eval_inputs
PYTHONPATH=src .venv/bin/python -m gem_rags.cli upstream-inputs \
  --format selfrag \
  --retriever-kind qdrant_hash_vector \
  --retriever-option dims=512
PYTHONPATH=src .venv/bin/python -m gem_rags.cli upstream-inputs \
  --config data/working/ablation-bundles/local-policy-small-medium/materialized_config.json \
  --retriever self_rag_adaptive_bm25_graph \
  --format selfrag
```

This bridge exports the harness QA set and retrieved evidence into the file shapes expected by the cloned upstream projects without taking over their heavyweight generation stacks. `selfrag_input.jsonl` includes `question`, `answers`, `ctxs`, and `top_contexts` fields for `external/rag-implementations/self-rag/retrieval_lm/run_short_form.py`. CRAG exports `crag_test_mutcd.txt` as repeated `question [SEP] passage` rows with `--crag-ndocs` rows per question, plus `crag_sources`, `crag_retrieved_psgs`, and `crag_answers.jsonl` sidecars for bookkeeping. The command always writes `manifest.json` with the retriever config, output paths, row count, evidence counts, upstream repo entrypoint checks, and ready-to-run/template command arrays under `upstream_commands`; `scripts/export_upstream_eval_inputs.py` remains a compatibility wrapper.

These scripts use the cloned repositories under `external/`, OpenAI-compatible model settings where applicable, and ignored working directories under `data/working/`. They are designed as harness boundaries; install upstream dependencies and configure model/embedding endpoints before indexing.

Check all command-backed adapter readiness, including the local vector DB command wrapper and cloned external RAGs, with:

```bash
.venv/bin/python scripts/check_external_adapters.py
.venv/bin/python scripts/check_external_adapters.py --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The aggregate report has three useful top-level lists:

- `ready`: adapter can answer queries with the default command, credentials, and local index artifacts in the current environment.
- `environment_ready`: the cloned package imports or CLI starts, but credentials or indexes may still be missing.
- `blocked_by_credentials`: the environment is usable, but the default command still needs provider API keys.

For local OpenAI-compatible endpoints, the GraphRAG, LightRAG, RAG-Anything, and PaperQA2 shims support `--allow-missing-api-key`; this makes `check` treat the adapter as credential-ready and uses the dummy key `local` for calls that still require an API-key field.
The aggregate checker applies the correct argument ordering for each adapter when `--allow-missing-api-key` is set.

Build query indexes for all environment-ready adapters with:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --dry-run
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes \
  --config data/working/ablation-bundles/local-policy-small-medium/materialized_config.json \
  --dry-run
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The builder runs each adapter's check command first, skips adapters whose cloned package or isolated environment is not usable, skips adapters that are already query-ready unless `--force` is passed, and writes structured JSON for automation. The top-level `query_ready`, `needs_index`, `needs_environment`, and `check_only_not_ready` lists separate adapters that can run now, adapters whose build commands should run, adapters that need heavy dependency environments, and check-only adapters such as the MRAG reference that still need dependencies or credentials. `setup_plan` records a per-adapter action and command list so a setup job can decide what to do next without parsing nested check output. Corpus-backed adapters automatically run `scripts/export_mrag_corpus.py` before indexing. Use `--config path/to/materialized_config.json` to target the command-backed retrievers referenced by a prepared sweep; config-derived setup also inherits local OpenAI-compatible `--allow-missing-api-key` and `--base-url` flags from retriever commands/checks. Use `--only graphrag,lightrag,paperqa2` to target a manual subset, `--visrag-limit N` or `--hipporag-limit N` for smoke indexes, and `--strict-skips` when a skipped adapter should fail the setup job. The legacy `scripts/build_external_indexes.py` wrapper is kept for existing shell workflows.

Bootstrap the currently supported upstream environments with:

```bash
scripts/bootstrap_external_envs.sh
```

This installs LightRAG and PaperQA2 editable into the main ignored `.venv`, installs GraphRAG editable into `data/working/venvs/graphrag/` with Python 3.13, prepares GraphRAG input/settings, prepares the VisRAG page-image manifest, and builds PaperQA2's deferred-embedding chunk index. GraphRAG is isolated because the current project `.venv` is Python 3.14 while upstream GraphRAG declares `>=3.11,<3.14`.
Set `BOOTSTRAP_HEAVY_RAGS=1` to also create ignored envs for MRAG reference (`data/working/venvs/mrag-reference/`), HippoRAG (`data/working/venvs/hipporag/`), and VisRAG (`data/working/venvs/visrag/`). Their wrapper scripts automatically re-run under those interpreters when present, so existing `external_command` configs can keep invoking `.venv/bin/python scripts/query_*.py ...`.

## Ablation Summaries

Raw experiment rows stay in `runs/<experiment>/runs.jsonl`. Aggregate by retriever, context mode, and model with:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/ablation.template.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli plan configs/ablation.template.json \
  --name local-plan-sample \
  --limit 2 \
  --retrievers bm25,visrag_pages \
  --context-modes injected,tool_explore,tool_search \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --no-external-checks \
  --preflight \
  --output runs/local-plan-sample/plan.json \
  --csv runs/local-plan-sample/plan.csv
PYTHONPATH=src .venv/bin/python -m gem_rags.cli validate configs/smoke.local.json --strict
PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze runs/smoke-local/runs.jsonl \
  --output-dir runs/smoke-local/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis context_mode \
  --baseline injected
```

Downloaded MRAG prior runs can be normalized into the same row schema:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli import-mrag-eval --overwrite --strict
PYTHONPATH=src .venv/bin/python -m gem_rags.cli validate configs/mrag-prior-eval.json \
  --runs runs/mrag-prior-eval/runs.jsonl \
  --strict
PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze runs/mrag-prior-eval/runs.jsonl \
  --output-dir runs/mrag-prior-eval/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis model \
  --baseline qwen3-vl-flash
```

The importer joins `eval/runs.jsonl` with `eval/scored.jsonl`, maps the prior Qwen VLM configurations to harness model fields, and reconstructs evidence from the extracted chunk, figure, and page-image caches. Those imported rows can be summarized, compared, or regraded like newly generated ablation rows.
`configs/mrag-prior-eval.json` validates the imported row matrix. It intentionally retains the original Qwen provider/model names, so preflight still checks Qwen credentials if you try to use it as a fresh run config.

`gem-rags preflight` validates the question/answer file, MRAG cache, retriever kinds, known external adapter checks, model/grader provider packages, API-key env vars, and the estimated row count before a run starts.
`gem-rags plan` enumerates concrete QA/retriever/context/model conditions and estimates answer-model and judge-model calls. `tool_explore` counts as two answer-model calls per row because the model first chooses hits to open and then answers from the opened evidence. `tool_search` counts as three answer-model calls per row because the model chooses search queries, chooses returned hits to open, and then answers. `paid_model_calls` excludes `dry_run` model rows, heuristic grading, and full-config `dry_run: true`.
Use `--max-rows`, `--max-total-model-calls`, or `--max-paid-model-calls` on `plan`, `prepare-ablation`, or `sweep` as a launch gate; `sweep` writes the materialized config and plan, then exits before retrieval/model calls if the budget is exceeded.
Use `gem-rags validate --max-total-tokens N --strict` after a run to gate on observed answer plus judge token usage from provider metadata.
Use `--models-file` with a line-oriented matrix like `configs/model-matrix.example.txt` when comparing many Anthropic, Grok, OpenAI, Qwen, and local OpenAI-compatible models. The tracked matrix contains current API model examples and local aliases; edit local aliases to match your server before running non-smoke calls. OpenAI entries can use `api=responses` and `reasoning_effort=low|medium|high|xhigh`, and preflight blocks unresolved placeholders such as `replace-with-*`, `*-placeholder`, and `*-or-successor`.
For repeatable large matrices, generate that file from `configs/model-catalog.example.json`:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli model-matrix \
  configs/model-catalog.example.json \
  --providers openai,anthropic,xai,qwen,local_openai \
  --sizes small,medium \
  --output data/working/model-matrices/provider-small-medium.txt
```

The catalog supports provider, size, role, tag, and optional non-runtime pricing metadata, skips entries marked `enabled=false` by default, and can emit JSON with `--format json`. Add account-current USD rates as `pricing.input_per_1m` and `pricing.output_per_1m` before paid sweeps; local example aliases use zero-cost pricing as a schema example. The default catalog includes an enabled OpenAI GPT-5.5 final-grader entry configured for Responses API `reasoning_effort=xhigh`; switch the model slug if GPT-5.6 or another account-enabled judge is preferred. `prepare-ablation --grader-from-catalog` can select exactly one `roles=["grader"]` entry from the same catalog; combine it with `--grader-providers`, `--grader-sizes`, `--grader-tags`, and `--include-disabled-graders` to pin the intended judge in the generated config.
Retriever matrices can also be generated from `configs/retriever-catalog.example.json`:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli retriever-matrix \
  configs/retriever-catalog.example.json \
  --families graphrag,lightrag,raganything \
  --modes local,hybrid \
  --output data/working/retriever-matrices/external-local-hybrid.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli plan configs/ablation.template.json \
  --name external-mode-plan \
  --limit 1 \
  --retrievers-file data/working/retriever-matrices/external-local-hybrid.json \
  --context-modes injected,tool_explore,tool_search \
  --models-file data/working/model-matrices/provider-small-medium.txt \
  --grader heuristic:heuristic
```

The retriever catalog contains local baselines, Self-RAG/CRAG policy variants, MRAG reference retrieval, GraphRAG query methods, LightRAG query modes, RAG-Anything query modes, HippoRAG, VisRAG pages, and PaperQA2. Generated entries carry explicit `check_command` fields so preflight does not have to infer readiness from the adapter command. `prepare-ablation` adds `upstream_inputs_<retriever>` follow-up commands using `gem-rags upstream-inputs` for Self-RAG/CRAG policy retrievers so the same materialized config can export upstream-native files without hand-rebuilding nested retriever options.
For repeatable setup, `prepare-ablation` writes the QA split, QA coverage JSON/CSV, generated model matrix, generated retriever matrix, materialized config, plan JSON/CSV, optional preflight, and follow-up setup/run commands into one ignored directory:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli prepare-ablation configs/ablation.template.json \
  --name external-mode-small \
  --qa-size 12 \
  --qa-seed 20260708 \
  --model-providers openai,anthropic,xai,qwen,local_openai \
  --model-sizes small \
  --retriever-families graphrag,lightrag,raganything \
  --retriever-modes local,hybrid \
  --context-modes injected,tool_explore,tool_search \
  --grader-from-catalog \
  --grader-providers openai \
  --grader-sizes judge \
  --dry-run \
  --output-dir data/working/ablation-bundles/external-mode-small
```

The `qa_coverage.*` artifacts compare the selected QA IDs against the full gold set across refusal, figure, and reference strata so a prepared sweep records what it does and does not cover. Add `--preflight` to attach readiness status to the plan bundle before spending model calls. Use `--no-external-checks` with `--preflight` when the goal is to validate dataset/provider shape without probing heavyweight adapters. Use `--dry-run` for a no-paid-call execution preview: the generated run config preserves target answer and judge labels, uses dry-run answer generation, skips non-heuristic grader calls, and reports zero `paid_model_calls`.
`gem-rags analyze` writes `analysis.json`, `summary.*`, `leaderboard.*`, and repeated matched-pair comparison artifacts for every observed candidate value on a selected axis. With `--qa-path`, it also writes QA-stratified summary and comparison CSVs for refusal, figure-backed, reference-backed, reference-count, reference-content-type, and question-type slices. Add `--model-catalog configs/model-catalog.example.json` after updating its `pricing` metadata to include observed answer/judge USD costs in those artifacts. The leaderboard ranks condition groups by mean judge score, then row error rate, then observed cost/tokens. For the context-mode example, rows are matched by QA, retriever, model provider, model, and grader, then each metric reports baseline mean, candidate mean, mean delta, wins, losses, and ties. Default metrics include grader scores, answer/judge token usage when providers return it, observed answer/judge cost when pricing is supplied, and tool-use diagnostics for selected hits, opened hits, search queries, unique search results, search errors, and parse failures.
Rows include separate `retrieval_error`, `model_error`, and `judge_error` fields plus `model_raw` and `grader_raw` metadata for auditability. Retriever build failures, retrieval exceptions, model build/generation exceptions, and grader exceptions are recorded per row, allowing large external-adapter sweeps to continue after one implementation is broken. Summaries count `retrieval_errors`, and matched comparisons include `retrieval_failed` by default so command-adapter failures do not look like legitimate empty-evidence retrievals.
`gem-rags validate` compares `runs.jsonl` against the config's expected QA/retriever/context/model rows and reports missing, duplicate, unexpected, invalid, failed, incomplete judge-score, stale-grader, token-usage, and token-budget status. `gem-rags sweep` writes the same report to `runs/<experiment>/validation.json` automatically.
After fixing a broken external index, dependency, credential, command, stale grader label, or incomplete judge-score row, use `--retry-errors` with `run` or `sweep`. It keeps clean rows, removes rows with `retrieval_error`, `model_error`, `judge_error`, stale grader labels, or incomplete judge-score rubrics, and reruns only those row keys so validation does not fail on duplicates.
`gem-rags regrade` rewrites judge fields into a new JSONL so old retrieval/model outputs can be scored with a newer final grader:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli regrade configs/ablation.template.json \
  --runs runs/local-tool-explore/runs.jsonl \
  --output runs/local-tool-explore/regraded-final-judge.jsonl \
  --grader openai:<final-judge-model> \
  --strict
```

It preserves the original run file, records row-level `judge_error` failures, and supports `--only-missing` for incremental repair.

The LLM grader receives the generated answer plus the retrieved evidence payload, gold answer JSON, gold references, and gold figures. Its output is normalized so every rubric key is present in `judge_scores`, even if the judge omits a field or wraps JSON in a fenced block.

The model matrix uses provider aliases that preflight can reason about directly:

- `openai` -> `OPENAI_API_KEY`
- `anthropic` -> `ANTHROPIC_API_KEY` through LiteLLM
- `xai` / `grok` -> `XAI_API_KEY`
- `qwen` -> `DASHSCOPE_API_KEY`
- `local_openai` -> local OpenAI-compatible endpoint, no API-key env required by default

For command-adapter regression testing without running the full external matrix:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/external-rag.smoke.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/external-rag.local-openai.smoke.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli run configs/external-rag.smoke.json --overwrite
PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze runs/external-rag-smoke/runs.jsonl \
  --output-dir runs/external-rag-smoke/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis retriever \
  --baseline mrag_reference
```

## Local Vector Tool

The Qdrant-backed baseline also has a small search/open CLI that mirrors the `tool_explore` prompt contract:

```bash
.venv/bin/python scripts/query_vector_db.py check
.venv/bin/python scripts/query_vector_db.py search --question "What does Section 2A.04 require?" --top-k 6
.venv/bin/python scripts/query_vector_db.py open --chunk-id MUTCD11e_2A04_Standard_13
```

The retriever catalog exposes this as `qdrant_hash_vector_command` for command-boundary ablations. The `search` command prints harness-native `evidence` rows, so it can be used directly by `external_command`.

## Recommended External Integration Order

1. **LightRAG**: use `lightrag_corpus.txt`, index once, then wrap query modes `naive`, `local`, `global`, and `hybrid` as separate retrievers.
2. **RAG-Anything**: use `raganything_content_list.json` so text chunks and figure/table crops enter its multimodal pipeline without reparsing the PDF.
3. **PaperQA2**: use `chunks.jsonl` or the source PDF depending on whether we want chunk-controlled parity or its native PDF parsing.
4. **GraphRAG**: use `chunks.jsonl` as input documents; treat indexing as an expensive offline step.
5. **HippoRAG**: use `chunks.jsonl` text fields as docs; likely best for graph/memory comparison rather than visual evidence.
6. **VisRAG**: use page images from the MRAG extract for parsing-free visual document retrieval.
7. **Self-RAG / CRAG**: implement as retrieval-control policies layered over existing retrievers instead of full corpus reindexing first.

## External Command Contract

The `external_command` retriever accepts `options.command` as a shell-split string or list. Placeholders:

- `{question}`
- `{qa_id}`
- `{mrag_dir}`
- `{top_k}`

Only those exact placeholders are expanded. `top_k` comes from the retriever config for injected/tool-explore runs and from the model-requested search budget for each `tool_search` query. Literal braces in inline scripts or JSON snippets can be used directly; `{{` and `}}` are also accepted and normalized to single braces for compatibility with older configs.

The command should print selected evidence or final RAG output to stdout. Preferred JSON shapes are:

- `{"chunks": [{"text": "...", "section_id": "2A.04", "content_type": "Standard", "ordinal": 13, "score": 1.0}]}`
- `{"figures": [{"figure_id": "Figure 2A-1", "caption": "...", "image_path": "...", "score": 1.0}]}`
- `{"pages": [{"page_pdf": 17, "page_printed": "2A-4", "text": "...", "image_path": "...", "score": 1.0}]}`
- `{"contexts": [{"text": "...", "name": "source-id", "score": 1.0}]}`
- `{"evidence": [{"evidence_id": "source-id", "kind": "chunk", "text": "...", "metadata": {"section_id": "2A.04"}, "score": 1.0}]}`
- `{"result": "..."}` or `{"answer": "..."}` for systems that only expose a final context block or answer.

The harness converts `evidence`, `chunks`, `figures`, `pages`, and `contexts` into individual evidence rows, then falls back to a single `tool_trace` row for raw text, `result`, or `answer`. Stderr and return code are captured in retrieval debug metadata either way.

Example config sketch:

```json
{
  "name": "lightrag_hybrid_context",
  "kind": "external_command",
  "options": {
    "command": [
      ".venv/bin/python",
      "scripts/query_lightrag_index.py",
      "query",
      "--mode",
      "hybrid",
      "--only-need-context",
      "--question",
      "{question}"
    ],
    "check_command": [
      ".venv/bin/python",
      "scripts/query_lightrag_index.py",
      "check"
    ],
    "timeout_s": 300
  }
}
```

For local OpenAI-compatible external runs, include the same local credential mode in `check_command`, for example:

```json
{
  "check_command": [
    ".venv/bin/python",
    "scripts/query_lightrag_index.py",
    "check",
    "--base-url",
    "http://localhost:8000/v1",
    "--allow-missing-api-key"
  ]
}
```
