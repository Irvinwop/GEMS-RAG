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

These exports are ignored because they are derived from ignored data. `gems-rag external-indexes` also runs this exporter automatically before corpus-backed adapters index.

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
- `external_command`: runs a preexisting indexed RAG system through a command template and captures stdout as tool evidence. JSON stdout can include `evidence`, `chunks`, `figures`, `pages`, or `contexts`; visual/page metadata such as image paths and page numbers is preserved for multimodal adapters and eligible images are passed to vision-enabled answer models. Commands run from the harness repository root by default; set `options.cwd` when an upstream wrapper needs a different working directory.

## Context Modes

- `injected`: the selected retriever's evidence text is placed directly into the answer prompt. Retrieved images are attached to that answer call when `vision=true`.
- `tool_explore`: a structured multi-prompt simulation. The selected retriever first produces a hit catalog; the model returns JSON `open_hit_ids`, and the runner opens only those IDs for the final answer prompt. Only opened images are attached. Runs record `retrieval_debug.context_debug.selected_ids` and `opened_ids`.
- `tool_search`: a structured multi-prompt simulation. The model returns JSON search queries, the harness runs them, then the model returns hit IDs to open before a final answer prompt is built. Only images from the selected open set are attached.
- `tool_native`: real provider function calls. The model receives `search(query, top_k)` and `open(hit_ids)` tools, explores the same configured retriever without automatic context, and answers after opening evidence. Search returns metadata and short previews only; open returns bounded text and triggers separate image blocks for opened visual evidence. Local image paths are redacted from provider-visible tool JSON. Runs preserve provider continuations, normalized tool traces, searches, opened IDs, and image-transport status.

## Visual Evidence Transport

Set `vision=true` on answer and grader model options to transmit retrieved page/figure files as base64 data URLs. The harness uses the documented `image_url` content format for Chat Completions and LiteLLM, and `input_image` for Responses. The tracked cloud-model examples enable it; local aliases default to `vision=false` because OpenAI-compatible local servers may expose text-only checkpoints. See the primary provider formats for [OpenAI](https://platform.openai.com/docs/guides/images-vision), [LiteLLM](https://docs.litellm.ai/docs/completion/vision), [xAI](https://docs.x.ai/developers/model-capabilities/images/understanding), and [Qwen](https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai).

`max_images` defaults to 5, matching the default open limit; `max_image_bytes` defaults to 20 MiB per image; and `image_detail` may be `low`, `high`, or `auto`. Unsupported, missing, oversized, capped, disabled, and dry-run image paths remain text-only and are reported under `model_raw.image_input`. The LLM grader receives only images present in retrieved/opened evidence, never unretrieved gold images, and records the same status under `grader_raw.model_raw.image_input`.
At corpus load and external-command ingestion, image paths are resolved by basename against the configured MRAG `figures/` and `page_images/` directories. This repairs the extracted reference cache's original Colab paths without mutating ignored source data; unresolved paths remain unchanged so `image_input.skipped_images` records the failure instead of silently dropping evidence.

## Manuscript Paper Algorithms

Five manuscript methods run directly over the shared repaired corpus and graph without a separate index:

- `sam_rag_adaptive_multimodal` follows the official SAM-RAG retrieval flow: rank shared text/figure candidates, verify `isRel` in batches, and stop at the first relevant batch. The harness grader reports answer usefulness/support separately so those calls remain visible in experiment accounting.
- `lpkg_planned_retrieval` parses the official LPKG `generated_predictions.jsonl` plan syntax without executing generated code, runs every `Sub_Question_n` against the shared corpus, resolves `{Ans_n}` dependencies from prior evidence labels, and merges evidence with step-weighted reciprocal rank. It deliberately leaves final answer generation to the harness model matrix.
- `kg2rag_graph_guided` adapts the official KG2RAG seed, graph expansion, and context organization stages to MUTCD chunk/section edges.
- `m3kg_rag_paper_spec` implements modality-wise text/figure seeding, multi-hop graph lifting, and a deterministic GRASP relevance proxy. The cited paper has no public code and its audio branch is inapplicable to the image/text MUTCD corpus; both limitations are recorded in retrieval debug output.
- `okh_rag_paper_spec` treats section membership as a higher-order hyperedge and returns document-order evidence trajectories. The cited paper has no public code/checkpoint, so the adapter records that it uses observable document precedence rather than a learned transition model.

These names are explicit about whether they are an official-algorithm adaptation or a paper-spec implementation. They do not claim to reproduce unreleased training artifacts.

Normalize plans produced by the official LPKG fine-tuning/inference scripts before selecting its retriever:

```bash
.venv/bin/python scripts/prepare_lpkg_plans.py normalize \
  --predictions /path/to/generated_predictions.jsonl
.venv/bin/python scripts/prepare_lpkg_plans.py check
```

The normalizer aligns predictions to `qa_id` by row order and refuses count mismatches or plans without parseable subquestions. LPKG did not publish a trained planner checkpoint. For harness availability and adapter smoke tests, generate deterministic one-step plans in the same official syntax:

```bash
.venv/bin/python scripts/prepare_lpkg_plans.py atomic \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl
.venv/bin/python scripts/prepare_lpkg_plans.py check \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl
```

Atomic rows and retrieval results are marked `official_lpkg_atomic_fallback` with checkpoint `unavailable_upstream`. This validates official plan parsing and iterative retrieval plumbing, but it is not a learned-planner reproduction and must not be compared as that condition. Scientific LPKG planner runs still require externally generated official-format predictions or a separately trained checkpoint.

## Implemented External Shims

DPR and canonical RAG retrieval:

```bash
.venv/bin/python scripts/query_dpr_index.py check
.venv/bin/python scripts/query_dpr_index.py index
.venv/bin/python scripts/query_dpr_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

This adapter uses the original `facebook/dpr-ctx_encoder-single-nq-base` and `facebook/dpr-question_encoder-single-nq-base` checkpoints over the shared MUTCD chunks. Inputs are truncated to the upstream DPR `hf_bert` configuration's 256-token sequence length. `dpr_dense` exposes the cited DPR method. `canonical_rag_dpr` uses the same non-parametric memory while keeping generation in the harness model matrix, which preserves the manuscript's requirement that all retrieval methods use the same answer model.

GFM-RAG:

```bash
.venv/bin/python scripts/query_gfmrag_index.py prepare --force
.venv/bin/python scripts/query_gfmrag_index.py index
.venv/bin/python scripts/query_gfmrag_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

`prepare` converts the repaired NetworkX graph to the official `nodes.csv` / `relations.csv` / `edges.csv` stage-one interface, retaining exactly 5,705 MUTCD chunks as document nodes with full metadata. It mirrors document-to-entity citations into the entity-to-document mapping required by GFM-RAG's document ranker. `index` and `query` run the official `GFMRetriever` and the pinned `rmanluo/GFM-RAG-8M` revision `4da9e4655d12`. A deterministic BM25 section-alias NER/entity-linking boundary replaces the upstream API-backed NER so retrieval does not require a hidden answer-model call; the pretrained graph foundation model remains the document ranker, and no fallback retriever rewrites its scores.

The upstream package declares CUDA-only `vllm` and `faiss-gpu-cu12` dependencies and eagerly imports optional training/embedding backends. `patches/gfmrag-retrieval-only.patch` lazy-loads those unused paths and fixes the documented bring-your-own-graph flow so `raw/documents.json` is not required when stage-one CSVs already exist. The bootstrap script applies this patch idempotently and installs the smaller CPU-compatible retrieval dependency set. Ready markers include the model revision, a SHA-256 fingerprint of all three stage-one files, and the generated stage-two graph paths; interrupted or stale builds therefore cannot pass `check`.

MegaRAG:

```bash
.venv/bin/python scripts/query_megarag_index.py prepare
.venv/bin/python scripts/query_megarag_index.py check
.venv/bin/python scripts/query_megarag_index.py index
.venv/bin/python scripts/query_megarag_index.py query --top-k 6 --question "What does Section 2A.04 require?"
```

`prepare` converts the existing MRAG extract directly to MegaRAG's native per-page JSON schema, avoiding a second lossy MinerU parse. The full prepared input contains 1,162 page images, 5,707 canonical text chunks, and 299 local figure/table crops. Each page receives a stable page marker because the upstream chunker hashes text alone and otherwise collapses multiple blank/image-only pages onto one chunk ID. Indexing and retrieval use the official `MegaRAG` class, GME-Qwen2-VL embedder, MMKG construction/refinement, and the exact LightRAG `v1.4.3` dependency expected upstream.
For a substantive smoke test, pass the same scope to every phase, for example `prepare --start-page 42 --limit 1`, then `check`, `index`, and `query` with `--start-page 42 --limit 1`. The setup builder exposes the same controls as `--megarag-start-page` and `--megarag-limit`. Scope-bound completion markers prevent a smoke index from satisfying a full-corpus check, and prepared input plus completion markers are published atomically. A manifest-specific document ID and atomic attempt record allow same-input retries while rejecting a changed corpus/backend in an existing working directory unless `--force` is explicit.

MegaRAG's published query helper performs MMKG retrieval and page-image retrieval, then invokes an internal two-stage answer synthesizer. Its `mix_two_step` code does not propagate `only_need_context`. The harness shim runs the same official `hybrid` MMKG and `naive` page branches concurrently with `only_need_context=True`, emits recovered chunk/page evidence plus both raw contexts, and bypasses only that internal final synthesis. This keeps retrieval multimodal while ensuring the selected harness model is the sole final generator. MegaRAG has a custom upstream license; review it before non-research use.
Local profiles cap both MegaRAG text and vision completions at 2,048 tokens. Readiness is published only after every embedded LightRAG page document reaches `processed`, so an interrupted build can be rerun without exposing a partial index.
The bootstrap applies `patches/megarag-empty-graph.patch`, which makes the official merge and refinement stages accept pages with no extracted graph records instead of indexing an empty mapping or calling `asyncio.wait()` on an empty set. It also propagates extraction/refinement failures after upstream records the failed status, preventing a later success update from hiding the error. This is required for cover and image-only pages in a complete manual.

MRAG reference implementation:

```bash
.venv/bin/python scripts/query_mrag_reference.py check --mode full
.venv/bin/python scripts/query_mrag_reference.py retrieve --mode full --question "What does Section 2A.04 require?"
.venv/bin/python scripts/query_mrag_reference.py stop
```

This wraps the cloned `hannanazad/MRAG_stp2` index and points it at the repaired extracted MRAG directory with `MRAG_BASE_DIR`. Its explicit modes are `dense`, `hybrid`, `multimodal`, `full`, `no_graph`, `no_visual`, `no_rule`, and `no_hierarchy`. Dense and hybrid checks do not require the visual or reranker dependencies; each enhanced mode fails closed when one of its required components is unavailable. Retrieval defaults to an auto-started Unix-socket worker under ignored `data/working/mrag-reference-server/`, which keeps the official models and Qdrant store loaded between calls, caches identical mode/query requests, survives an interrupted harness run, and exits after 30 idle minutes. Use `--no-persistent` for a one-shot diagnostic process or `stop` to release the warm worker immediately.

The tracked wrapper translates FlagEmbedding 1.4's `dtype` argument for the upstream-pinned Transformers 4.54 runtime so hybrid retrieval retains real BGE-M3 lexical weights instead of silently falling back to dense-only retrieval. It also deduplicates colliding Qdrant chunk IDs, localizes Colab image paths to the imported bundle, and repairs the upstream graph-expansion placeholder by adding two-hop graph-neighbor chunks to the candidate set before scoring and reranking.

GraphRAG:

```bash
.venv/bin/python scripts/query_graphrag_index.py check
.venv/bin/python scripts/query_graphrag_index.py prepare --force
.venv/bin/python scripts/query_graphrag_index.py init
.venv/bin/python scripts/query_graphrag_index.py index --method fast --community-levels 2
.venv/bin/python scripts/query_graphrag_index.py query --method local --top-k 6 --json --question "What does Section 2A.04 require?"
```

This uses Microsoft GraphRAG's cloned Typer CLI through the source tree. `prepare` writes exported MRAG chunks to `data/working/graphrag_index/input/mutcd_chunks.txt`; GraphRAG still owns its normal `init`, `index`, and `query` phases. When `data/working/venvs/graphrag/bin/python` exists, the shim uses it automatically. Override with `GRAPHRAG_PYTHON=/path/to/python` or `--python /path/to/python`. GraphRAG has no separate API service: its generated config calls the selected model provider through an environment variable named `GRAPHRAG_API_KEY`. The harness fills that variable from `OPENAI_API_KEY` by default, while `GRAPHRAG_API_KEY` or `--api-key-env` can override the provider credential.
For a bounded validation build, pass the same `--limit N` to `prepare`, `index`, `check`, and `query`, or use `external-indexes --graphrag-limit N`. Preparation atomically replaces its input file, and GraphRAG's upstream cache remains available on rerun after an interrupted index. The completion marker records the limit, so a smoke index cannot satisfy a full-corpus check or query.
Indexing summarizes community level 2 by default because all cataloged local, global, and DRIFT query profiles request level 2. The wrapper filters only the input seen by Microsoft's registered community-report workflow: `communities.parquet` retains the complete hierarchy, while `community_reports.parquet` and its embeddings contain the requested levels. Use `index --community-levels 0,1,2` for several query levels or `index --community-levels all` to reproduce the much slower all-level upstream build. The completion marker records the available report levels, and structured queries reject an unbuilt level instead of silently returning incomplete context. `check` likewise requires level 2 unless `check --community-level N` selects another profile. Completed model calls remain in GraphRAG's cache after interruption.
`init` expands GraphRAG's generic organization/person/place/event extraction schema with MUTCD-specific device, facility, road-user, regulation, standard, and concept types. Override the comma-separated schema with `init --entity-types ...` for a different corpus.
The setup builder defaults to Microsoft's official `fast` indexing method, which constructs the graph with deterministic NLP and avoids requiring a large schema-following extraction model. Use `external-indexes --graphrag-method standard` when deliberately testing standard LLM graph extraction with a model that has first passed an output-schema smoke test. Initialization replaces the synthetic few-shot blocks in the upstream entity and claim prompts with compact MUTCD-grounded format examples, and removes synthetic community examples, so small models retain the required output grammar without copying unrelated entities into MUTCD indexes. The local prompt emits each unique record once, caps each extraction at 40 records, and forbids treating source IDs as entities. Use `init --keep-index-prompt-examples` (or the legacy `--keep-community-prompt-examples` alias) only to reproduce the exact upstream prompts. Standard entity and claim extraction use a deterministic profile capped at 4,096 generated tokens with a 0.2 frequency penalty; override it with `init --entity-extraction-max-tokens N --entity-extraction-temperature N --entity-extraction-frequency-penalty N`. `init --max-gleanings N` controls optional follow-up extraction passes for that standard path and defaults to zero for small local models. Before each retry, the wrapper removes active-profile cache responses whose provider finish reason is `length`; a newly truncated response prevents the completion sentinel from being published even if upstream GraphRAG parsed partial records and returned success.
Use `--allow-missing-api-key` when the generated GraphRAG settings point at a local OpenAI-compatible endpoint that accepts a dummy key. Set `--embedding-base-url URL` when completion and embedding models run on separate endpoints; otherwise embeddings inherit `--base-url`.
Local backend profiles cap each GraphRAG query completion at 2,048 tokens through the generated model `call_args`; direct adapter calls can override this with `--llm-max-tokens`. Initialization assigns community reports a separate deterministic completion profile capped at 768 generated tokens and requests 300-word reports by default, preventing thousands of oversized summaries without truncating later queries. It asks for 2-4 distinct concise findings to avoid repetition loops caused by the upstream prompt's conflicting 5-10 multi-paragraph and 300-word requirements. For local endpoints selected with `--allow-missing-api-key`, uncached community-report calls receive a provider-side 4,096-token floor and a bounded JSON schema while retaining the configured 768-token cache key and Microsoft's original response parser. The provider schema limits reports to four findings and bounds individual text fields, preventing a constrained local decoder from producing unbounded arrays or paragraphs. This lets the few deterministic structured outputs that need more room finish without invalidating hundreds of successful cached reports; override the ceiling with `--community-report-token-floor N`. Before retry and before publishing a completion marker, the wrapper validates the active report cache as JSON with at least one non-empty finding; malformed or empty cached reports are removed and regenerated. The 2-4 count is a generation target rather than a completion requirement because Microsoft's GraphRAG response schema accepts one finding, and deterministic local models can otherwise reproduce the same valid response indefinitely. Each indexing workflow's cache partition includes its model profile, so changing a model or call setting preserves resumability without reusing stale completions. Settings, the effective local token floor, and all indexing-prompt checksums are included in the completion marker, so changing a prompt or runtime budget invalidates stale readiness. Override the report bounds and sampling with `init --community-report-max-tokens N --community-report-max-length N --community-report-temperature N`.
In JSON mode, the shim calls GraphRAG's upstream query helpers inside the isolated GraphRAG interpreter, captures `context_data`, and emits harness `contexts` rows capped by `--top-k`.
Drift search performs substantially more dependent model calls than local, global, or basic search. Its catalog timeout is 1,200 seconds so a slow local structured query and the upstream CLI fallback can finish without being killed at the ten-minute boundary.

LightRAG:

```bash
.venv/bin/python scripts/query_lightrag_index.py index
.venv/bin/python scripts/query_lightrag_index.py query --mode hybrid --top-k 6 --chunk-top-k 6 --only-need-context --question "What does Section 2A.04 require?"
```

The tracked LightRAG retriever configs pass the harness `{top_k}` budget to both `--top-k` and `--chunk-top-k` so entity/relationship and chunk retrieval budgets move together during ablations.
Local backend profiles also enable structured entity extraction and cap each internal text completion at 2,048 tokens. Direct adapter calls can set the same ceiling with `--llm-max-tokens`; failed or interrupted document states never produce a ready marker and can be retried without `--force`.
LightRAG completion checks target the deterministic document ID for the selected corpus rather than unrelated status rows, including synthetic `dup-*` failures left by retries. Direct queries fail closed unless the corpus/backend completion marker matches and all recorded index artifacts remain present.
RAG-Anything queries fail closed unless the completion marker matches the exact content-list checksum, ingestion mode, optional `--limit`, model settings, and endpoint. Use the same `--limit N` on `index`, `check`, and `query` for a deliberately scoped smoke index; omitting it always means the full corpus.

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
The adapter applies the same local structured-extraction and 2,048-token ceiling policy to its embedded LightRAG instance, and rejects embedded text failures before RAG-Anything can mark the outer document complete.
The official initializer normally validates MinerU before it initializes LightRAG, even when no parsing is requested. The shim suppresses that check only for the pre-parsed `shared_corpus` path and retrieval-only queries; `native_pdf` indexing still requires the configured upstream parser.

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

This wraps HippoRAG 2's `HippoRAG.index(...)` and `HippoRAG.retrieve(...)` methods over `chunks.jsonl`. The default path requires Torch, Transformers, python-igraph, and OpenAI-compatible chat and embedding services.
The upstream package eagerly imports CUDA-only VLLM/GritLM and optional Bedrock/Transformers backends even when the configured clients are OpenAI-compatible. It also pins the unpublished `openai==1.91.1`, omits the Parquet engine required by its default vector store, and recognizes API embedding models only when their IDs contain `text-embedding`. `patches/hipporag-lazy-optional-backends.patch` makes optional backends lazy, adds `pyarrow`, routes otherwise unrecognized embedding IDs such as `nomic-embed-text` through the configured OpenAI-compatible endpoint, uses the published `openai==1.91.0`, and atomically publishes Parquet, OpenIE, and graph files. The retrieval-only Python 3.12 bootstrap therefore installs no unused CUDA stack and cannot expose a file interrupted mid-write.

Indexing writes an ignored `mrag_chunk_manifest.jsonl` sidecar and a completion sentinel under the HippoRAG save directory. The sentinel is published atomically only after `HippoRAG.index(...)` returns and is bound to the corpus checksum, optional document limit, model names, and endpoint settings, so interrupted cache files and smoke indexes cannot pass a full-corpus `check`. Pass the same `--chunks` and `--limit` to `check` or `query` when intentionally validating a limited index. Per-completion SQLite caching, persistent embedding stores, and atomic Parquet/OpenIE/graph replacement make a retry continuable without marking partial work ready. Query uses the sidecar, or falls back to exported `chunks.jsonl`, to emit chunk contexts with MRAG `doc_id`, section, page, title, and content-type metadata instead of anonymous text hits.

For one OpenAI-compatible endpoint serving both chat and embeddings:

```bash
.venv/bin/python scripts/query_hipporag_index.py \
  --base-url http://localhost:8000/v1 \
  --allow-missing-api-key \
  check
```

Use `--llm-base-url` and `--embedding-base-url` when those services are separate.

VisRAG:

```bash
.venv/bin/python scripts/query_visrag_index.py check
.venv/bin/python scripts/query_visrag_index.py prepare --scope pages
.venv/bin/python scripts/query_visrag_index.py index
.venv/bin/python scripts/query_visrag_index.py query --top-k 6 --question "What does Section 2A.04 require?"
.venv/bin/python scripts/query_visrag_index.py stop
```

This wraps the cloned OpenBMB VisRAG repository at the `VisRAG-Ret` retrieval boundary. `prepare` builds an ignored manifest over MRAG page images, or figure/table crops with `--scope figures`/`--scope both`. `index` follows the upstream `AutoModel`/`AutoTokenizer` weighted-mean-pooling recipe for `openbmb/VisRAG-Ret`, pinned to revision `95ef596df871b606167cb7e4b7215caf1bfdf761`, and saves embeddings under `data/working/visrag_index/`. Each completed batch is flushed to `embeddings.partial.npy` with an atomic progress marker. Re-running the same command resumes at the first unfinished row; `--force` is required when the manifest, model, device, or dtype no longer matches that partial state. The final matrix is published only after every row finishes, and `check` verifies its manifest checksum, model revision, shape, and dtype through `embeddings.ready.json`.

The isolated retrieval environment uses native Python 3.12 with Torch 2.6, torchvision 0.21, Transformers 4.40.2, Accelerate 0.34, Pillow, NumPy 1.26, and SentencePiece. It deliberately omits the upstream training and VLM-generation dependency set because this adapter stops at VisRAG-Ret retrieval. Query commands auto-start an idle-expiring local worker so the 3.4B retriever and embedding matrix are loaded once per ablation run rather than once per question. The worker is bound to the ready-marker and runtime fingerprint, caches exact repeated queries, and can be shut down explicitly with `stop`.

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
PYTHONPATH=src .venv/bin/python -m gems_rag.cli upstream-inputs \
  --retriever-kind bm25_graph \
  --top-k 10 \
  --out-dir data/working/upstream_eval_inputs
PYTHONPATH=src .venv/bin/python -m gems_rag.cli upstream-inputs \
  --format selfrag \
  --retriever-kind qdrant_hash_vector \
  --retriever-option dims=512
PYTHONPATH=src .venv/bin/python -m gems_rag.cli upstream-inputs \
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

The aggregate report has four useful top-level lists:

- `ready`: adapter can answer queries with the default command, credentials, and local index artifacts in the current environment.
- `environment_ready`: the cloned package imports or CLI starts, but credentials or indexes may still be missing.
- `blocked_by_credentials`: the environment is usable, but the default command still needs provider API keys.
- `blocked_by_model_service`: credentials or dummy-key mode are configured, but the selected OpenAI-compatible endpoint is unavailable or rejects authorization.

For local OpenAI-compatible endpoints, the GraphRAG, HippoRAG, LightRAG, MegaRAG, RAG-Anything, and PaperQA2 shims support `--allow-missing-api-key`; this uses the dummy key `local` for clients that require an API-key field, then probes `<base-url>/models` before reporting the model service ready. GraphRAG also persists that URL as `api_base` for both generated completion and embedding model settings.
GUI-generated configs persist these choices under `rag_backend`, separate from `models` (answer generation) and `grader`. The profile controls internal chat, text-embedding, embedding-dimension, and vision model identifiers for the adapters that use each capability. MegaRAG retains its native GME multimodal embedding model while using the profile's chat model; VisRAG retains its native pinned visual retriever and does not use this profile.
The aggregate checker applies the correct argument ordering for each adapter when `--allow-missing-api-key` is set.

Build query indexes for all environment-ready adapters with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes --dry-run
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes \
  --config data/working/ablation-bundles/local-policy-small-medium/materialized_config.json \
  --dry-run
PYTHONPATH=src .venv/bin/python -m gems_rag.cli external-indexes --allow-missing-api-key --local-openai-base-url http://localhost:8000/v1
```

The builder runs each adapter's check command first, skips adapters whose cloned package, isolated environment, credentials, or model service is not usable, skips adapters that are already query-ready unless `--force` is passed, and writes structured JSON for automation. The top-level `query_ready`, `needs_index`, `needs_environment`, `needs_model_service`, and `check_only_not_ready` lists separate adapters that can run now, adapters whose build commands should run, adapters that need heavy dependency environments, adapters waiting for a model endpoint, and check-only adapters such as the MRAG reference that still need dependencies or credentials. `setup_plan` records a per-adapter action and command list so a setup job can decide what to do next without parsing nested check output. Corpus-backed adapters automatically run `scripts/export_mrag_corpus.py` before indexing. GraphRAG, LightRAG, RAG-Anything, HippoRAG, MegaRAG, VisRAG, and PaperQA2 require a matching completion marker before checks report an index ready; partial files left by an interrupted build are not query-ready, and rerunning the builder lets the upstream implementation resume or safely rebuild them. PaperQA2 also fsyncs a temporary pickle and atomically replaces the prior index. Use `--config path/to/materialized_config.json` to target the command-backed retrievers referenced by a prepared sweep; config-derived setup inherits the complete persisted RAG backend profile. Use `--only graphrag,lightrag,paperqa2` to target a manual subset, `--graphrag-limit N`, `--visrag-limit N`, `--hipporag-limit N`, or `--megarag-limit N` for smoke indexes, and `--strict-skips` when a skipped adapter should fail CI. The legacy `scripts/build_external_indexes.py` wrapper is kept for existing shell workflows.

Bootstrap the currently supported upstream environments with:

```bash
scripts/bootstrap_external_envs.sh
```

This installs LightRAG and PaperQA2 editable into the main ignored `.venv`, installs GraphRAG editable into `data/working/venvs/graphrag/` with Python 3.13, prepares GraphRAG input/settings, prepares the VisRAG page-image manifest, and builds PaperQA2's deferred-embedding chunk index. GraphRAG is isolated because the current project `.venv` is Python 3.14 while upstream GraphRAG declares `>=3.11,<3.14`.
Set `BOOTSTRAP_HEAVY_RAGS=1` to also create ignored envs for MegaRAG (`data/working/venvs/megarag/`), GFM-RAG (`data/working/venvs/gfmrag/`), DPR (`data/working/venvs/dpr/`), MRAG reference (`data/working/venvs/mrag-reference/`), HippoRAG (`data/working/venvs/hipporag/`), and VisRAG (`data/working/venvs/visrag/`). These heavy retrieval environments use native Python 3.12 on Apple Silicon; MegaRAG retains its pinned LightRAG `v1.4.3`, while HippoRAG uses an explicit retrieval-only dependency set instead of its CUDA-only optional packages. Their wrapper scripts automatically re-run under those interpreters when present, so existing `external_command` configs can keep invoking `.venv/bin/python scripts/query_*.py ...`.

## Ablation Summaries

Raw experiment rows stay in `runs/<experiment>/runs.jsonl`. Aggregate by retriever, context mode, and model with:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/ablation.template.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli plan configs/ablation.template.json \
  --name local-plan-sample \
  --limit 2 \
  --retrievers bm25,visrag_pages \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file configs/model-matrix.example.txt \
  --grader heuristic:heuristic \
  --no-external-checks \
  --preflight \
  --output runs/local-plan-sample/plan.json \
  --csv runs/local-plan-sample/plan.csv
PYTHONPATH=src .venv/bin/python -m gems_rag.cli validate configs/smoke.local.json --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/smoke-local/runs.jsonl \
  --output-dir runs/smoke-local/analysis \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
  --axis context_mode \
  --baseline injected
```

Downloaded MRAG prior runs can be normalized into the same row schema:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli import-mrag-eval --overwrite --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli validate configs/mrag-prior-eval.json \
  --runs runs/mrag-prior-eval/runs.jsonl \
  --strict
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/mrag-prior-eval/runs.jsonl \
  --output-dir runs/mrag-prior-eval/analysis \
  --qa-path data/extracted/MRAG-20260708T114057Z-3/MRAG/eval/gold_qa.jsonl \
  --axis model \
  --baseline qwen3-vl-flash
```

The importer joins `eval/runs.jsonl` with `eval/scored.jsonl`, maps the prior Qwen VLM configurations to harness model fields, and reconstructs evidence from the extracted chunk, figure, and page-image caches. Those imported rows can be summarized, compared, or regraded like newly generated ablation rows.
`configs/mrag-prior-eval.json` validates the imported row matrix. It intentionally retains the original Qwen provider/model names, so preflight still checks Qwen credentials if you try to use it as a fresh run config.

`gems-rag preflight` validates the question/answer file, MRAG cache, retriever kinds, known external adapter checks, model/grader provider packages, API-key env vars, and the estimated row count before a run starts.
`gems-rag plan` enumerates concrete QA/retriever/context/model conditions and estimates answer-model and judge-model calls. `tool_explore` counts as two answer-model calls per row because the model first chooses hits to open and then answers from the opened evidence. `tool_search` counts as three answer-model calls per row because the model chooses search queries, chooses returned hits to open, and then answers. `tool_native` reserves `tool_max_rounds + 1` calls per row (five by default) for bounded tool continuations plus a forced final answer. `paid_model_calls` excludes `dry_run` model rows, heuristic grading, and full-config `dry_run: true`.
Use `--max-rows`, `--max-total-model-calls`, or `--max-paid-model-calls` on `plan`, `prepare-ablation`, or `sweep` as a launch gate; `sweep` writes the materialized config and plan, then exits before retrieval/model calls if the budget is exceeded.
Use `gems-rag validate --max-total-tokens N --strict` after a run to gate on observed answer plus judge token usage from provider metadata.
Use `--models-file` with a line-oriented matrix like `configs/model-matrix.example.txt` when comparing many Anthropic, Grok, OpenAI, Qwen, and local OpenAI-compatible models. The tracked matrix contains current API model examples and local aliases; edit local aliases to match your server before running non-smoke calls. OpenAI entries can use `api=responses` and `reasoning_effort=low|medium|high|xhigh`; multimodal entries use `vision=true`; and preflight blocks unresolved placeholders such as `replace-with-*`, `*-placeholder`, and `*-or-successor`.
For repeatable large matrices, generate that file from `configs/model-catalog.example.json`:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli model-matrix \
  configs/model-catalog.example.json \
  --providers openai,anthropic,xai,qwen,local_openai \
  --sizes tiny,small,medium \
  --output data/working/model-matrices/provider-tiny-small-medium.txt
```

The catalog supports provider, size, role, tag, and optional non-runtime pricing metadata, skips entries marked `enabled=false` by default, and can emit JSON with `--format json`. Add account-current USD rates as `pricing.input_per_1m` and `pricing.output_per_1m` before paid sweeps; local example aliases use zero-cost pricing as a schema example. The OpenAI entries follow the current GPT-5.6 Luna/Terra/Sol tiers, and the enabled final grader uses GPT-5.6 Sol through Responses API with `reasoning_effort=xhigh`. `prepare-ablation --grader-from-catalog` can select exactly one `roles=["grader"]` entry from the same catalog; combine it with `--grader-providers`, `--grader-sizes`, `--grader-tags`, and `--include-disabled-graders` to pin the intended judge in the generated config.
Retriever matrices can also be generated from `configs/retriever-catalog.example.json`:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli retriever-matrix \
  configs/retriever-catalog.example.json \
  --families graphrag,lightrag,raganything \
  --modes local,hybrid \
  --output data/working/retriever-matrices/external-local-hybrid.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli plan configs/ablation.template.json \
  --name external-mode-plan \
  --limit 1 \
  --retrievers-file data/working/retriever-matrices/external-local-hybrid.json \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --models-file data/working/model-matrices/provider-tiny-small-medium.txt \
  --grader heuristic:heuristic
```

The retriever catalog contains local baselines, every manuscript method, Self-RAG/CRAG policy variants, MRAG reference retrieval, GraphRAG query methods, LightRAG query modes, MegaRAG dual retrieval, RAG-Anything query modes, HippoRAG, VisRAG pages, and PaperQA2. Generated entries carry explicit `check_command` fields so preflight does not have to infer readiness from the adapter command. `prepare-ablation` adds `upstream_inputs_<retriever>` follow-up commands using `gems-rag upstream-inputs` for Self-RAG/CRAG policy retrievers so the same materialized config can export upstream-native files without hand-rebuilding nested retriever options.
For repeatable setup, `prepare-ablation` writes the QA split, QA coverage JSON/CSV, a source model-catalog snapshot, generated model matrix, generated retriever matrix, materialized config, plan JSON/CSV, optional preflight, and follow-up setup/run commands into one ignored directory. Its generated analysis command points to the catalog snapshot, preserving the pricing metadata needed for observed-cost reports even if the source catalog later changes:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli prepare-ablation configs/ablation.template.json \
  --name external-mode-small \
  --qa-size 12 \
  --qa-seed 20260708 \
  --model-providers openai,anthropic,xai,qwen,local_openai \
  --model-sizes small \
  --retriever-families graphrag,lightrag,raganything \
  --retriever-modes local,hybrid \
  --context-modes injected,tool_explore,tool_search,tool_native \
  --grader-from-catalog \
  --grader-providers openai \
  --grader-sizes judge \
  --min-qa-per-stratum 1 \
  --max-total-cost-usd 5 \
  --dry-run \
  --output-dir data/working/ablation-bundles/external-mode-small
```

The `qa_coverage.*` artifacts compare the selected QA IDs against the full gold set across refusal, figure, and reference strata so a prepared sweep records what it does and does not cover. Use `--min-qa-per-stratum 1` on `prepare-ablation`, `plan`, or `sweep` to require every observed refusal x figure x reference stratum; generated sweep commands preserve the gate, and direct sweeps write coverage JSON/CSV before stopping on a failure. `--max-total-cost-usd 5` propagates an observed post-run ceiling to the generated sweep and strict validation commands, both of which use the bundle's model-catalog snapshot. Add `--preflight` to attach readiness status to the plan bundle before spending model calls. Use `--no-external-checks` with `--preflight` when the goal is to validate dataset/provider shape without probing heavyweight adapters. Use `--dry-run` for a no-paid-call execution preview: the generated run config preserves target answer and judge labels, uses dry-run answer generation, skips non-heuristic grader calls, and reports zero `paid_model_calls`.
`gems-rag analyze` writes `analysis.json`, `summary.*`, `leaderboard.*`, and repeated matched-pair comparison artifacts for every observed candidate value on a selected axis. With `--qa-path`, it also writes QA-stratified summary and comparison CSVs for refusal, figure-backed, reference-backed, reference-count, reference-content-type, and question-type slices. Add `--model-catalog configs/model-catalog.example.json` after updating its `pricing` metadata to include observed answer/judge USD costs in those artifacts. The leaderboard ranks condition groups by mean judge score, then row error rate, then observed cost/tokens. For the context-mode example, rows are matched by QA, retriever, model provider, model, and grader, then each metric reports baseline mean, candidate mean, mean delta, wins, losses, and ties. Default metrics include grader scores, answer/judge token usage when providers return it, observed answer/judge cost when pricing is supplied, and tool-use diagnostics for provider tool calls, selected hits, opened hits, search queries, unique search results, search errors, and parse failures.
Rows include separate `retrieval_error`, `model_error`, and `judge_error` fields plus `model_raw` and `grader_raw` metadata for auditability. Retriever build failures, retrieval exceptions, model build/generation exceptions, and grader exceptions are recorded per row, allowing large external-adapter sweeps to continue after one implementation is broken. Summaries count `retrieval_errors`, and matched comparisons include `retrieval_failed` by default so command-adapter failures do not look like legitimate empty-evidence retrievals.
`gems-rag validate` compares `runs.jsonl` against the config's expected QA/retriever/context/model rows and reports missing, duplicate, unexpected, invalid, failed, incomplete judge-score, stale-grader, token-usage, token-budget, cost-coverage, and observed-cost status. Pass `--model-catalog <catalog> --max-total-cost-usd <limit> --strict` to enforce a USD ceiling. Paid calls with missing pricing, missing usage, or incomplete multi-call usage fail cost coverage rather than being counted as zero; explicit zero-priced local models are accepted. `tool_explore` rows aggregate both answer-model calls, `tool_search` rows aggregate all three, and `tool_native` rows aggregate their actual provider continuation count. Individual raw calls remain under `model_raw.model_calls`. `gems-rag sweep` writes the same report to `runs/<experiment>/validation.json` automatically and accepts the same model catalog and cost limit.
After fixing a broken external index, dependency, credential, command, stale grader label, or incomplete judge-score row, use `--retry-errors` with `run` or `sweep`. It keeps clean rows, removes rows with `retrieval_error`, `model_error`, `judge_error`, stale grader labels, or incomplete judge-score rubrics, and reruns only those row keys so validation does not fail on duplicates.
`gems-rag regrade` rewrites judge fields into a new JSONL so old retrieval/model outputs can be scored with a newer final grader:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli regrade configs/ablation.template.json \
  --runs runs/local-tool-explore/runs.jsonl \
  --output runs/local-tool-explore/regraded-final-judge.jsonl \
  --grader openai:<final-judge-model> \
  --strict
```

It preserves the original run file, records row-level `judge_error` failures, and supports `--only-missing` for incremental repair.

The LLM grader receives the generated answer, retrieved evidence, and any available gold answer, references, and figures. Question-only rows are explicitly marked and do not receive false zero heuristic scores. Its output is normalized so every rubric key is present in `judge_scores`, even if the judge omits a field or wraps JSON in a fenced block.

The model matrix uses provider aliases that preflight can reason about directly:

- `openai` -> `OPENAI_API_KEY`
- `anthropic` -> `ANTHROPIC_API_KEY` through LiteLLM
- `xai` / `grok` -> `XAI_API_KEY`
- `qwen` -> `DASHSCOPE_API_KEY`
- `local_openai` -> local OpenAI-compatible endpoint, no API-key env required by default

For command-adapter regression testing without running the full external matrix:

```bash
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/external-rag.smoke.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli preflight configs/external-rag.local-openai.smoke.json
PYTHONPATH=src .venv/bin/python -m gems_rag.cli run configs/external-rag.smoke.json --overwrite
PYTHONPATH=src .venv/bin/python -m gems_rag.cli analyze runs/external-rag-smoke/runs.jsonl \
  --output-dir runs/external-rag-smoke/analysis \
  --qa-path data/extracted/MRAG-20260715T174043Z-1/MRAG/eval/gold_qa.jsonl \
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
7. **MegaRAG**: use the existing page render, canonical chunk, and figure-crop assets to build its native MMKG without rerunning document parsing.
8. **Self-RAG / CRAG**: implement as retrieval-control policies layered over existing retrievers instead of full corpus reindexing first.

## External Command Contract

The `external_command` retriever accepts `options.command` as a shell-split string or list. Placeholders:

- `{question}`
- `{qa_id}`
- `{mrag_dir}`
- `{top_k}`

Only those exact placeholders are expanded. `top_k` comes from the retriever config for injected/tool-explore runs and from the model-requested search budget for each `tool_search` or `tool_native` query. Literal braces in inline scripts or JSON snippets can be used directly; `{{` and `}}` are also accepted and normalized to single braces for compatibility with older configs.

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
