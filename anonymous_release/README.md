# MUTCD RAG Comparison Study

This folder is a self-contained, anonymous code-and-index release for the
MUTCD-150 retrieval comparison. It includes the exact query-time artifacts for
BM25, GraphRAG, PaperQA, and GEMS-RAG, plus the benchmark, gold annotations,
resumable runners, retrieval scoring code, and the latest curated GEMS-RAG
source snapshot.

The comparison-study methods are named exactly:

- `bm25`
- `graphrag`
- `paperqa`
- `gems-rag`

There are no public method variants with hardware, endpoint, ingestion, or
query-strategy suffixes. GraphRAG internally invokes its upstream `local`
search algorithm because that is the algorithm used by the study; the method
label remains `graphrag`.

## Release identity

- Release date: `{{RELEASE_DATE}}`
- Benchmark: MUTCD-150 v1.0
- Benchmark questions: `{{QUESTION_COUNT}}`
- Canonical text chunks: `{{CORPUS_COUNT}}`
- GraphRAG source revision: `{{GRAPHRAG_REVISION}}`
- PaperQA source revision: `{{PAPERQA_REVISION}}`
- GEMS-RAG source state: latest source snapshot available at packaging time,
  with the MUTCD part-hierarchy parser correction described below
- Release size: `{{RELEASE_SIZE}}`

The original Git histories and remote URLs are intentionally absent. The
GraphRAG and PaperQA revisions are retained because they identify external,
licensed research software and are needed to reproduce the environment. A
source revision for GEMS-RAG is intentionally not embedded in this anonymous
release.

## Folder layout

```text
.
├── README.md
├── RELEASE_MANIFEST.json
├── CHECKSUMS.sha256
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── .env.example
├── benchmark/
│   ├── questions.jsonl
│   ├── gold.jsonl
│   └── MUTCD_RAG_EVALUATION_SPECIFICATION.md
├── configs/
│   └── comparison.json
├── indexes/
│   ├── corpus/
│   │   ├── chunks.jsonl
│   │   └── manifest.json
│   ├── graphrag/
│   │   ├── input/
│   │   ├── output/
│   │   ├── prompts/
│   │   ├── settings.yaml
│   │   └── .graphrag_index.json
│   ├── paperqa/
│   │   ├── docs.pkl
│   │   └── docs.pkl.ready.json
│   └── gems-rag/
│       ├── mutcd11theditionr1hl.pdf
│       ├── figures/
│       ├── page_images/
│       ├── mmrag_cache_v3/
│       └── qdrant_db/
├── gems-rag/
│   ├── mrag/
│   │   └── ... internal GEMS-RAG Python package ...
│   ├── scripts/
│   ├── docs/
│   └── requirements.txt
├── scripts/
│   └── setup_environments.sh
├── pipelines/
│   ├── query_bm25.py
│   ├── query_graphrag.py
│   ├── query_paperqa.py
│   ├── query_gems_rag.py
│   ├── run_comparison.py
│   └── score_retrieval.py
├── src/comparison_support/
│   └── ... shared adapter and metric helpers ...
└── third_party/
    ├── graphrag/
    └── paperqa/
```

The source repository is published under `gems-rag/`. Its internal Python
import package remains `mrag` for compatibility with the research
implementation. Public source, method, environment, command, and index names
use `gems-rag`; only the internal import namespace and upstream `MRAG_*`
configuration variables retain their original spelling.

## What is included

### BM25

`pipelines/query_bm25.py` is the dependency-free BM25 baseline used in the
study. It builds an in-memory index over the shared 5,705-chunk corpus. Its
tokenization, `k1=1.5`, `b=0.75`, IDF formula, and tie-breaking behavior match
the comparison implementation.

BM25 has no remote API, model-server, or GPU requirement.

### GraphRAG

`third_party/graphrag/` contains the external GraphRAG source snapshot used by
the adapter. `indexes/graphrag/` contains the completed MUTCD index: parquet
tables, LanceDB vectors, prompts, input text, settings, and a completion
marker. Index-time caches and logs are omitted because they are not needed for
queries.

The supplied index was built over the shared canonical corpus. Query-time
semantic search must use an embedding model compatible with the index; the
study used `nomic-embed-text`. The adapter's context-only mode constructs
GraphRAG context without asking GraphRAG to write the final answer.

### PaperQA

`third_party/paperqa/` contains the external PaperQA source snapshot used by
the adapter. `indexes/paperqa/docs.pkl` is the completed shared-corpus PaperQA
document index. It contains one PaperQA text source per canonical MUTCD chunk.

PaperQA still uses configured embedding and language-model endpoints while
selecting and summarizing evidence. Context-only mode suppresses the final
PaperQA answer, not the retrieval-time operations required by PaperQA.

### GEMS-RAG

The release root replaces the old GEMS-RAG repository README. The latest
curated implementation is under `gems-rag/`; its ingestion entry points are
under `gems-rag/scripts/`. `indexes/gems-rag/` contains the query-time assets:

- local Qdrant collections for chunks, figure captions, visual figures, and
  rendered pages;
- the NetworkX knowledge graph;
- canonical chunk and figure metadata;
- extracted figure/table crops and rendered page images; and
- the source MUTCD PDF for audit or index rebuilding.

Large intermediate embedding arrays and model caches are omitted. They
duplicate data already stored in Qdrant and are not needed to query the built
index. Model weights are also omitted and are downloaded by the configured
libraries. Query-time model downloads and caches are written under the ignored
`runs/model-cache/` directory by default. Set `HF_HOME` or `MRAG_HF_HOME` to
use another cache location.

The packaged parser includes a hierarchy correction that derives each part
from the section/chapter identifier. This prevents later MUTCD parts from
inheriting a stale outline part heading.

## What is not included

The folder intentionally excludes:

- manuscript drafts;
- evaluation outputs, model answers, grading results, and cost records;
- prior comparison or ablation run directories;
- notebooks, backup trees, old deployment bundles, and obsolete releases;
- Git metadata, remotes, commit authors, and local machine paths;
- `.env` files, API tokens, provider credentials, and production keys;
- Python virtual environments and downloaded model caches; and
- GraphRAG index-time response caches and GEMS-RAG duplicate embedding
  intermediates.

## Benchmark and gold data

`benchmark/questions.jsonl` contains all MUTCD-150 questions.
`benchmark/gold.jsonl` contains the corresponding locked answerability and
evidence annotations. The gold row is joined by `question_id`, not by line
position.

Some questions are intentionally unanswerable from the MUTCD. They must not
be forced to return a supported answer. Retrieval scoring constructs finite
canonical-chunk qrels from the gold pages, sections, tables, and figures.
Questions with no text qrels are marked `evaluable: false`, reported
explicitly, and excluded from macro text-retrieval means; they are not silently
assigned zero retrieval credit.

The grader instructions used for answer evaluation are included at
`benchmark/MUTCD_RAG_EVALUATION_SPECIFICATION.md`.

## Requirements

- Python 3.11, 3.12, or 3.13
- Python 3.13 for the supplied GraphRAG snapshot
- approximately `{{RELEASE_SIZE}}` of disk for this release, plus virtual
  environments and model weights
- an OpenAI-compatible embedding endpoint for GraphRAG and PaperQA queries
- an OpenAI-compatible language-model endpoint for PaperQA retrieval
- local model weights for full GEMS-RAG retrieval
- a CUDA GPU with adequate memory is strongly recommended for GEMS-RAG

BM25 runs on any ordinary CPU. GraphRAG and PaperQA can use either hosted APIs
or a local OpenAI-compatible server. Full GEMS-RAG retrieval can run without a
remote API key, but loading its text encoder, reranker, and visual encoder on
CPU is substantially slower than GPU execution.

## Environment setup

Create separate environments because the three research implementations have
different dependency constraints:

```bash
bash scripts/setup_environments.sh all
```

Or install one method:

```bash
bash scripts/setup_environments.sh graphrag
bash scripts/setup_environments.sh paperqa
bash scripts/setup_environments.sh gems-rag
```

BM25 uses the standard library and needs no environment installation.

The setup script defaults to `python3.13` for GraphRAG and `python3.12` for
PaperQA and GEMS-RAG. Override them when necessary:

```bash
PYTHON_GRAPHRAG=/path/to/python3.13 \
PYTHON=/path/to/python3.12 \
bash scripts/setup_environments.sh all
```

## Credentials and endpoints

No RAG method in this folder contains or receives an embedded API key. Copy
the variable names from `.env.example` into an ignored `.env`, then export
them into the shell:

```bash
set -a
source .env
set +a
```

Credential requirements are:

| Method | Remote key required | Model endpoint required |
|---|---:|---:|
| BM25 | No | No |
| GraphRAG | Only if endpoint enforces it | Yes, query embedding |
| PaperQA | Only if endpoint enforces it | Yes, embedding and LLM |
| GEMS-RAG retrieval | No | No |
| GEMS-RAG answer generation | Provider-dependent | Provider-dependent |

For a local server that accepts any non-empty bearer token, pass
`--allow-missing-api-key`. The adapters then use a fixed placeholder string;
they do not create or persist a credential.

## Validate the supplied indexes

Run these checks from the release root:

```bash
python pipelines/query_bm25.py check

python pipelines/query_graphrag.py \
  --python .venv-graphrag/bin/python \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}" \
  --embedding-base-url "${OPENAI_BASE_URL}" \
  --query-embedding-model nomic-embed-text \
  check

.venv-paperqa/bin/python pipelines/query_paperqa.py \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}" \
  check --embedding nomic-embed-text

python pipelines/query_gems_rag.py \
  --python .venv-gems-rag/bin/python \
  check --mode full
```

The GraphRAG and PaperQA completion markers hash their source/index files.
Moving the release does not invalidate them because the markers contain file
identities, not absolute paths.

## Query one method

BM25:

```bash
python pipelines/query_bm25.py query \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

GraphRAG context:

```bash
python pipelines/query_graphrag.py \
  --python .venv-graphrag/bin/python \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}" \
  --embedding-base-url "${OPENAI_BASE_URL}" \
  --query-embedding-model nomic-embed-text \
  query \
  --method local \
  --context-only \
  --json \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

PaperQA context:

```bash
.venv-paperqa/bin/python pipelines/query_paperqa.py \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}" \
  query \
  --context-only \
  --embedding nomic-embed-text \
  --llm qwen2.5:3b \
  --summary-llm qwen2.5:3b \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

Full GEMS-RAG retrieval:

```bash
python pipelines/query_gems_rag.py \
  --python .venv-gems-rag/bin/python \
  retrieve \
  --mode full \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

All query adapters emit one JSON object to stdout.

## Run the resumable comparison

The default run covers the three comparison systems:

```bash
python pipelines/run_comparison.py \
  --methods bm25,graphrag,paperqa \
  --output runs/comparison \
  --top-k 10 \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}"
```

Add the study's primary method when needed:

```bash
python pipelines/run_comparison.py \
  --methods bm25,graphrag,paperqa,gems-rag \
  --output runs/comparison-with-gems-rag \
  --top-k 10 \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}"
```

Every completed `(question_id, method)` pair is written atomically to its own
file under `runs/.../rows/`. `results.jsonl` and `state.json` are regenerated
after each pair. If the process, network, machine, or terminal is interrupted,
rerun the same command and completed rows are skipped.

Failed rows are retained for diagnosis. Retry only failures with:

```bash
python pipelines/run_comparison.py \
  --methods bm25,graphrag,paperqa \
  --output runs/comparison \
  --top-k 10 \
  --allow-missing-api-key \
  --base-url "${OPENAI_BASE_URL}" \
  --retry-errors
```

The run manifest hashes every retrieval-affecting setting. Reusing an output
directory with a different corpus, method list, model, endpoint, or depth is
rejected; use a new output directory instead.

## Score retrieval

After a completed run:

```bash
python pipelines/score_retrieval.py \
  --results runs/comparison/results.jsonl \
  --output runs/comparison/retrieval_metrics
```

The scorer writes:

- `normalized_rankings.jsonl`
- `qrels.jsonl`
- `per_question_metrics.jsonl`
- `native_context_sensitivity.jsonl`
- `summary.json`
- `summary.csv`
- `qrels_report.json`

The primary metrics are canonical-chunk Recall@10, MRR@10, and binary nDCG@10.
GraphRAG source contexts are mapped back to canonical chunks using exact spans
in the packaged GraphRAG input. PaperQA, BM25, and GEMS-RAG map by stable chunk
identifier.

## Reusing retrieval across answer models

The comparison is an injected-context study. Retrieval output is computed once
per retrieval configuration and may be reused across every downstream answer
model. Do not rerun retrieval merely because the answer model changes.

Reuse is valid only while all retrieval-affecting inputs remain fixed:

- corpus and built index;
- method and method configuration;
- top-k depth;
- embedding model and endpoint behavior; and
- any retrieval-time language model used internally by the RAG.

BM25 is deterministic for a fixed corpus and query. The supplied runners
persist the exact GraphRAG and PaperQA context so later answer-model sweeps use
the same retrieval output even if an endpoint itself is nondeterministic.

## Rebuilding indexes

The package is query-ready; rebuilding is optional.

### BM25

No persistent build step is needed. The index is constructed from
`indexes/corpus/chunks.jsonl` when the adapter starts.

### GraphRAG

The adapter supports `prepare`, `init`, and `index` subcommands. Rebuilding can
be expensive because graph extraction and community reports call a language
model. Use a new index directory rather than overwriting the supplied index
until the rebuilt index passes `check`.

### PaperQA

Use `pipelines/query_paperqa.py index` with `--ingestion-mode shared_corpus`.
The embedding model must be recorded and reused at query time.

### GEMS-RAG

Set `MRAG_BASE_DIR` to an asset directory containing the PDF, then run:

```bash
MRAG_BASE_DIR=/path/to/gems-rag-assets \
.venv-gems-rag/bin/python gems-rag/scripts/ingest_v4.py
```

`ingest_v4.py` is resumable through intermediate files and imports unchanged
embedding/Qdrant stages from `ingest_v3.py`. Do not point a rebuild at the
supplied query-ready assets unless replacement is intentional.

## Index portability

Absolute Colab-era media paths were removed from the packaged GEMS-RAG figure
metadata, graph, and Qdrant payloads. The release adapter resolves media by
filename under `indexes/gems-rag/figures` and
`indexes/gems-rag/page_images`.

Qdrant local storage must be opened by one process at a time. The persistent
GEMS-RAG worker in `query_gems_rag.py` owns the database and reuses loaded
models across questions.

## Integrity

`RELEASE_MANIFEST.json` records every included file, byte size, and SHA-256
digest. `CHECKSUMS.sha256` provides the same hashes in standard checksum-file
format.

Verify the release:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

## Licensing

GraphRAG's MIT license and PaperQA's Apache-2.0 license are preserved inside
their respective `third_party/` directories. See
`THIRD_PARTY_NOTICES.md`.

The GEMS-RAG source snapshot did not include a project license. This release
does not invent one. Add the intended license before public redistribution.
The MUTCD manual and its derived data are also separate from the software
licenses; verify the applicable federal publication terms for the intended
distribution channel.

## Anonymous-release hygiene

The release builder verifies that the assembled folder contains no:

- Git directories or remotes;
- local usernames or home-directory paths;
- API-key values or copied `.env` files;
- historical method IDs;
- manuscript drafts, run outputs, or grading artifacts; or
- stale absolute media paths in GEMS-RAG index payloads.

Third-party author and project attributions required by their licenses remain
intact.
