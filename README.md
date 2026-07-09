# GEM-RAGs

Harness workspace for running RAG ablation experiments across model providers, retrieval strategies, and grading configurations.

Local-only inputs are intentionally ignored:

- `data/raw/` stores downloaded datasets and archives.
- `manuscript-draft/` stores the current manuscript draft.
- `external/MRAG_stp2/` stores the cloned reference implementation.
- `external/rag-implementations/` stores cloned comparison RAG repositories.

The planned harness should make it cheap to compare:

- automatic context injection versus model-driven data exploration through a two-step search/open tool loop
- different RAG pipelines, including dependency-free in-memory and Qdrant-backed local vector baselines
- model families and sizes across Anthropic, Grok, OpenAI, Qwen, and local runners
- grader configurations, with the current expectation that grading uses a high-reasoning GPT-5.5/5.6-class model when available

See [docs/implementation-inventory.md](docs/implementation-inventory.md) for the current local data, reference implementation, and cloned external RAG inventory.

Local MRAG metadata repairs can be checked or re-applied with:

```bash
.venv/bin/python scripts/repair_mrag_metadata.py --dry-run
```

External RAG input corpora can be exported with:

```bash
python3 scripts/export_mrag_corpus.py
```

Summarize and slice the gold QA file before running a paid sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli qa-summary
PYTHONPATH=src .venv/bin/python -m gem_rags.cli qa-split \
  --size 12 \
  --seed 20260708 \
  --strategy balanced \
  --output data/working/qa-splits/balanced-12.json
```

The balanced split strategy cycles across refusal, figure-grounding, and referenced/unreferenced strata so small sweeps do not only test the first contiguous rows in `gold_qa.jsonl`.

Import the downloaded MRAG prior generated/scored runs into the harness schema:

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

This preserves the prior Qwen VLM answers and judge scores while enriching them with local chunk, figure, and page evidence from the extracted MRAG cache.
`analyze` writes `analysis.json`, `summary.*`, and one metrics/pairs comparison set for every observed non-baseline model under the selected axis. With `--qa-path`, it also writes `strata-summary.csv` and `strata-comparisons.csv` for refusal, figure-backed, reference-backed, reference-count, reference-content-type, and question-type slices.
`configs/mrag-prior-eval.json` is for structural validation and comparison of imported historical rows; preflight will still report missing Qwen credentials unless `DASHSCOPE_API_KEY` is configured for fresh model calls.

Run the local smoke matrix with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/smoke.local.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli run configs/smoke.local.json --overwrite
PYTHONPATH=src .venv/bin/python -m gem_rags.cli validate configs/smoke.local.json --strict
```

External adapter indexes and heavyweight package environments are local and ignored. After exporting corpora, bootstrap the currently supported upstream environments with:

```bash
scripts/bootstrap_external_envs.sh
```

By default this installs the lighter command-backed adapters and prepares GraphRAG, VisRAG manifests, and PaperQA deferred chunks. To also build isolated heavy dependency envs for MRAG reference, HippoRAG, and VisRAG, run:

```bash
BOOTSTRAP_HEAVY_RAGS=1 scripts/bootstrap_external_envs.sh
```

The heavy wrappers automatically re-run themselves under `data/working/venvs/mrag-reference/bin/python`, `data/working/venvs/hipporag/bin/python`, or `data/working/venvs/visrag/bin/python` when those ignored envs exist.

Then build whatever command-backed external indexes are possible in the current environment:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --dry-run
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes \
  --config data/working/ablation-bundles/local-policy-small-medium/materialized_config.json \
  --dry-run
PYTHONPATH=src .venv/bin/python -m gem_rags.cli external-indexes --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The builder runs adapter readiness checks, skips missing heavy environments instead of failing the whole setup, and emits JSON with `query_ready`, `needs_index`, `needs_environment`, `check_only_not_ready`, and a per-adapter `setup_plan` in addition to the lower-level `built`, `already_ready`, `would_run`, `skipped`, and `failed` lists. Use `--config path/to/materialized_config.json` to derive the subset from command-backed retrievers in a prepared sweep, `--only graphrag,lightrag` for a manual subset, `--force` to rebuild ready adapters, and `--strict-skips` when skipped adapters should fail CI. The legacy `scripts/build_external_indexes.py` entrypoint delegates to the same package code.

Self-RAG and CRAG can consume harness retrieval results through upstream-compatible eval input exports:

```bash
.venv/bin/python scripts/export_upstream_eval_inputs.py \
  --retriever-kind bm25_graph \
  --top-k 10 \
  --out-dir data/working/upstream_eval_inputs
```

This writes ignored `selfrag_input.jsonl` and CRAG `question [SEP] passage` files under `data/working/upstream_eval_inputs/`, plus a manifest recording the retriever and row counts.

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
```

Then use `configs/external-rag.template.json` as the starting point for command-backed external runs.
Preflight an ablation config before spending model calls:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/ablation.template.json
```

The preflight report estimates run rows and lists dataset, retriever, model, grader, credential, and external-adapter blockers.
Materialize a smaller concrete ablation config without editing JSON by hand:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli materialize configs/ablation.template.json \
  --output configs/generated/local-tool-explore.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only
```

Model and grader specs use `provider:model[,key=value...]`. For large provider sweeps, put one model spec per line in a file like `configs/model-matrix.example.txt` and pass `--models-file path/to/models.txt` to `materialize`, `plan`, or `sweep`; edit any local endpoint aliases to match your server before running paid calls. OpenAI entries can set `api=responses` and `reasoning_effort=low|medium|high|xhigh`; unresolved model placeholders such as `replace-with-*`, `*-placeholder`, and `*-or-successor` are blocked by preflight.
`--ready-only` prunes blocked retrievers and models after preflight; it still fails if the dataset, context modes, or grader are blocked.
To generate a matrix from provider, size, role, and tag metadata instead of hand-editing long lists:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli model-matrix \
  configs/model-catalog.example.json \
  --providers openai,anthropic,xai,qwen,local_openai \
  --sizes small,medium \
  --output data/working/model-matrices/provider-small-medium.txt
```

The catalog defaults merge shared options like `temperature=0`, provider options like OpenAI `api=responses`, a local OpenAI-compatible `base_url`, and per-model overrides. Use the generated file with `--models-file`, or pass `--roles grader --format json` to inspect the current final-grader entry before selecting or editing it.
`prepare-ablation --grader-from-catalog --grader-providers openai --grader-sizes judge` selects exactly one enabled `roles=["grader"]` entry from the same catalog and persists it into the materialized config; add `--grader-tags final` or similar when the catalog has multiple judge candidates, or `--include-disabled-graders` when testing a disabled backup judge.
External retriever mode matrices can be generated the same way:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli retriever-matrix \
  configs/retriever-catalog.example.json \
  --families graphrag,lightrag,raganything \
  --modes local,hybrid \
  --output data/working/retriever-matrices/external-local-hybrid.json
```

Use the generated JSON with `--retrievers-file` on `materialize`, `plan`, or `sweep`. The catalog includes local baselines, Self-RAG/CRAG policy variants, the MRAG reference wrapper, and external mode variants for GraphRAG, LightRAG, RAG-Anything, HippoRAG, VisRAG, and PaperQA2.
To write the QA split, model matrix, retriever matrix, materialized config, and plan in one ignored bundle:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli prepare-ablation configs/ablation.template.json \
  --name local-policy-small-medium \
  --qa-size 12 \
  --qa-seed 20260708 \
  --model-providers openai,anthropic,xai,qwen,local_openai \
  --model-sizes small,medium \
  --retriever-families local,self_rag_policy,crag_policy \
  --context-modes injected,tool_explore,tool_search \
  --grader-from-catalog \
  --grader-providers openai \
  --grader-sizes judge \
  --dry-run \
  --output-dir data/working/ablation-bundles/local-policy-small-medium
```

The bundle report includes exact follow-up commands for preflight, sweep, resume, retrying error rows, and context-mode analysis. `--dry-run` preserves the intended model and grader labels but forces dry-run answer generation and skips non-heuristic grader calls; plans still show logical model calls and report `paid_model_calls: 0`.
Plan the exact row matrix and model-call count before launching a sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli plan configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only \
  --output runs/local-tool-explore/plan.json \
  --csv runs/local-tool-explore/plan.csv
```

Use `--retrievers-file data/working/retriever-matrices/external-local-hybrid.json` in place of `--retrievers ...` when planning generated external mode matrices.

`tool_explore` rows estimate two logical answer-model calls per row: one selection call plus one answer call. `tool_search` rows estimate three logical answer-model calls per row: one search-query call, one open-selection call, and one answer call. Non-heuristic graders add one logical judge-model call per row. `paid_model_calls` excludes `dry_run` model rows, heuristic grading, and full-config `dry_run: true`.
Use `--max-rows`, `--max-total-model-calls`, or `--max-paid-model-calls` on `plan`, `prepare-ablation`, or `sweep` to make oversized matrices fail before a paid run starts.
Run the same materialization as an end-to-end sweep:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli sweep configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only \
  --overwrite
```

`sweep` writes `materialized_config.json`, `preflight.json`, `runs.jsonl`, `summary.json`, `summary.csv`, and context comparison artifacts under `runs/<experiment-name>/` when `injected` is paired with `tool_explore` or `tool_search`.
It also writes `validation.json`, which checks expected row completeness, duplicate rows, unexpected rows, invalid JSON lines, retrieval/model/judge error counts, incomplete judge-score rubrics, and stale grader labels. Retriever build failures, retrieval exceptions, model build/generation exceptions, and grader exceptions are recorded on individual rows so a broken external adapter does not abort the whole sweep. Use `--allow-run-errors` only for best-effort sweeps where failed rows should not make the command exit non-zero.
After fixing a broken index, credential, adapter command, stale grader label, or incomplete judge-score row, rerun only repairable rows while keeping clean rows:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli sweep configs/ablation.template.json \
  --name local-tool-explore \
  --qa-ids-file data/working/qa-splits/balanced-12.json \
  --retrievers bm25,qdrant_hash_vector,bm25_graph,oracle_gold_refs \
  --context-modes injected,tool_explore,tool_search \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --ready-only \
  --retry-errors
```

For larger matrices, run `analyze` over the finished `runs.jsonl` to emit a reusable report directory and repeated matched-pair comparisons across any axis:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze runs/local-tool-explore/runs.jsonl \
  --output-dir runs/local-tool-explore/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis context_mode \
  --baseline injected
```

When the final judge model changes, regrade an existing run without rerunning retrieval or answer generation:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli regrade configs/ablation.template.json \
  --runs runs/local-tool-explore/runs.jsonl \
  --output runs/local-tool-explore/regraded-final-judge.jsonl \
  --grader openai:<final-judge-model> \
  --strict
```

Use `--only-missing` to fill only rows with missing `judge_scores` or an existing `judge_error`. The command refuses in-place output so the original `runs.jsonl` remains intact.
Use the one-question external smoke config to verify command-backed adapter failure/success reporting without running the full external matrix:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli preflight configs/external-rag.smoke.json
PYTHONPATH=src .venv/bin/python -m gem_rags.cli run configs/external-rag.smoke.json --overwrite
```

The cloned MRAG reference implementation can be checked with:

```bash
.venv/bin/python scripts/query_mrag_reference.py check
```

Grading behavior:

- Smoke configs use the deterministic `heuristic` grader for cheap regression checks.
- Ablation configs use an LLM grader through `openai_compatible` or `litellm`.
- The LLM grader prompt includes the question, gold answer, gold references, retrieved evidence, and generated answer, then normalizes all rubric keys in `judge_scores`.

Context modes:

- `injected`: the runner directly places retrieved evidence into the answer prompt.
- `tool_explore`: the runner first asks the model to choose hit IDs from a catalog, opens only those selected hits, and then asks the model to answer from the opened tool results.
- `tool_search`: the runner gives the model no retrieved context up front; the model first chooses search queries, the harness runs those searches against the configured retriever, the model chooses hits to open, and the final answer is generated from only those opened results.

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
For external adapters pointed at a local OpenAI-compatible server, GraphRAG, LightRAG, RAG-Anything, and PaperQA2 checks accept `--allow-missing-api-key` and use a dummy `local` key for clients that require an API-key field.
Use `configs/external-rag.local-openai.smoke.json` to preflight those local-compatible command adapters with matching `check_command` settings.
Command-backed adapters may emit JSON `evidence`, `chunks`, `figures`, `pages`, or `contexts`; the harness preserves visual/page metadata such as image paths, figure IDs, and PDF/printed page numbers.

Summarize an ablation run with:

```bash
PYTHONPATH=src .venv/bin/python -m gem_rags.cli analyze runs/smoke-local/runs.jsonl \
  --output-dir runs/smoke-local/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis context_mode \
  --baseline injected
```

Search the local Qdrant vector DB baseline directly with:

```bash
.venv/bin/python scripts/query_vector_db.py check
.venv/bin/python scripts/query_vector_db.py search --question "What does Section 2A.04 require?"
```

The retriever catalog also includes `qdrant_hash_vector_command`, which runs the same vector DB through the `external_command` boundary and emits harness-native `evidence` rows.
