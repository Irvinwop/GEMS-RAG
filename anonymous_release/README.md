# MUTCD RAG Comparison Study

This folder is a self-contained, anonymous code-and-index release for the
MUTCD-150 retrieval comparison. It contains the complete BM25, GraphRAG, and
PaperQA comparison artifacts; a compact GEMS-RAG text-and-graph index and its
parent source; the benchmark and gold annotations; resumable runners;
retrieval scoring; and the grader specification supplied with the study.

Public method IDs are exactly:

- `bm25`
- `graphrag`
- `paperqa`
- `gems-rag`

There are no public method variants with endpoint, hardware, ingestion, or
query-strategy suffixes.

## Release identity

- Release date: `{{RELEASE_DATE}}`
- Benchmark: MUTCD-150 v1.0
- Benchmark questions: `{{QUESTION_COUNT}}`
- Canonical text chunks: `{{CORPUS_COUNT}}`
- GraphRAG source revision: `{{GRAPHRAG_REVISION}}`
- PaperQA source revision: `{{PAPERQA_REVISION}}`
- GEMS-RAG source state: latest source snapshot available at packaging time,
  with the MUTCD hierarchy correction described below
- Release size: `{{RELEASE_SIZE}}`

Git histories and remote URLs are intentionally absent. GraphRAG and PaperQA
revision identifiers are retained because they identify licensed research
software required for reproduction. A GEMS-RAG revision identifier is not
embedded in the anonymous release.

## Folder layout

```text
.
|-- README.md
|-- RELEASE_MANIFEST.json
|-- CHECKSUMS.sha256
|-- THIRD_PARTY_NOTICES.md
|-- .env.example
|-- benchmark/
|   |-- questions.jsonl
|   |-- gold.jsonl
|   `-- MUTCD_RAG_EVALUATION_SPECIFICATION.md
|-- configs/
|   `-- comparison.json
|-- indexes/
|   |-- corpus/
|   |-- graphrag/
|   |-- paperqa/
|   `-- gems-rag/
|-- gems-rag/
|   |-- mrag/
|   |-- scripts/
|   |-- docs/
|   `-- requirements.txt
|-- pipelines/
|   |-- build_indexes_openai.py
|   |-- query_bm25.py
|   |-- query_graphrag.py
|   |-- query_paperqa.py
|   |-- query_gems_rag.py
|   |-- run_comparison.py
|   `-- score_retrieval.py
|-- scripts/
|   |-- package_upload.py
|   `-- setup_environments.sh
|-- src/comparison_support/
`-- third_party/
    |-- graphrag/
    `-- paperqa/
```

The GEMS-RAG repository is published under `gems-rag/`. Its internal Python
import package remains `mrag` for compatibility with the parent research
implementation. The parent source and its provider catalogs are retained.

## Included methods

### BM25

`query_bm25.py` implements the dependency-free baseline over the shared
5,705-chunk corpus. It uses `k1=1.5`, `b=0.75`, stable chunk identifiers, and
deterministic tie breaking. It requires no persistent build step.

### GraphRAG

The release includes the research-team source required by the adapter and a
completed MUTCD study index containing parquet tables, vector storage,
prompts, input text, settings, and a completion marker.

The adapter exposes preparation, initialization, indexing, validation, and
context retrieval. Public output always uses the method ID `graphrag`.

### PaperQA

The release includes the research-team source required by the adapter and the
completed shared-corpus document index. Each PaperQA source maps to one
canonical MUTCD chunk, allowing retrieved contexts to be scored by stable
chunk identifier.

Context-only retrieval suppresses PaperQA's final answer while retaining the
evidence-selection operations used by the comparison.

### GEMS-RAG

The GEMS-RAG source is under `gems-rag/`. The compact query-time profile
includes the text and caption Qdrant collections, knowledge graph, canonical
chunk and figure metadata, and source MUTCD PDF. Its packaged query mode is
`no_visual`.

Derived page images, figure crops, and their visual-vector collections are
excluded so the entire release fits in one upload file. They are not used by
the three-method BM25, GraphRAG, and PaperQA comparison. The source PDF and
parent ingestion code remain available for auditing those derivatives.

The packaged parser derives every MUTCD part from the section identifier. This
prevents later parts from inheriting an earlier outline heading. Colab-era
media paths are rewritten to release-relative paths in the graph, metadata,
and Qdrant payloads.

## Benchmark, gold data, and grader

`benchmark/questions.jsonl` contains all MUTCD-150 questions.
`benchmark/gold.jsonl` contains the locked answerability and evidence
annotations. Rows are joined by `question_id`, not line position.

Thirty questions are intentionally unanswerable from the MUTCD. They must not
be forced to return a supported answer. Questions without finite text qrels
are reported as unevaluable and excluded from macro text-retrieval means.

The supplied grading instructions are included unchanged at
`benchmark/MUTCD_RAG_EVALUATION_SPECIFICATION.md`.

## Environment setup

Use separate environments because the research implementations have different
dependency constraints:

```bash
bash scripts/setup_environments.sh all
```

Install one method when preferred:

```bash
bash scripts/setup_environments.sh graphrag
bash scripts/setup_environments.sh paperqa
bash scripts/setup_environments.sh gems-rag
```

BM25 uses the Python standard library.

The setup script defaults to Python 3.13 for GraphRAG and Python 3.12 for
PaperQA and GEMS-RAG. Override the interpreter paths when necessary:

```bash
PYTHON_GRAPHRAG=/path/to/python3.13 \
PYTHON=/path/to/python3.12 \
bash scripts/setup_environments.sh all
```

## API configuration

No credential is embedded in this release. Create an ignored `.env` from the
variable names in `.env.example`, then export it:

```bash
set -a
source .env
set +a
```

The documented index-construction path uses the official OpenAI API:

```text
OPENAI_BASE_URL=https://api.openai.com/v1
```

Set `OPENAI_API_KEY` and the OpenAI IDs selected for GraphRAG and PaperQA in
the corresponding environment variables. Provider credentials used by the
answer configurations remain separate.

## Build indexes with the OpenAI API

The supplied study indexes are preserved for audit. To build fresh BM25,
GraphRAG, and PaperQA artifacts through the OpenAI API, run:

```bash
python pipelines/build_indexes_openai.py \
  --methods bm25,graphrag,paperqa
```

The command requires:

```text
OPENAI_API_KEY
GRAPHRAG_LLM_MODEL
GRAPHRAG_EMBEDDING_MODEL
PAPERQA_EMBEDDING_MODEL
```

Fresh artifacts are written under `rebuilt_indexes/`, leaving the supplied
study artifacts untouched. Each completed stage is persisted to
`rebuilt_indexes/state.json`. Rerunning the same command skips completed
stages. A failed stage remains recorded and can be retried with
`--retry-failed` after its cause is corrected.

BM25 records readiness immediately because it is constructed in memory from
the canonical corpus.

## Validate indexes

Run from the release root:

```bash
python pipelines/query_bm25.py check

python pipelines/query_graphrag.py \
  --python .venv-graphrag/bin/python \
  --working-dir rebuilt_indexes/graphrag \
  --base-url https://api.openai.com/v1 \
  --embedding-base-url https://api.openai.com/v1 \
  check

.venv-paperqa/bin/python pipelines/query_paperqa.py \
  --index rebuilt_indexes/paperqa/docs.pkl \
  --base-url https://api.openai.com/v1 \
  check --embedding "${PAPERQA_EMBEDDING_MODEL}"

python pipelines/query_gems_rag.py \
  --python .venv-gems-rag/bin/python \
  check --mode no_visual
```

The supplied completion markers hash their source and index inputs. Moving the
release does not invalidate them because they contain file identities rather
than machine-specific paths.

## Query one method

BM25:

```bash
python pipelines/query_bm25.py query \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

GraphRAG context from an OpenAI-built index:

```bash
python pipelines/query_graphrag.py \
  --python .venv-graphrag/bin/python \
  --working-dir rebuilt_indexes/graphrag \
  --base-url https://api.openai.com/v1 \
  --embedding-base-url https://api.openai.com/v1 \
  --query-llm-model "${GRAPHRAG_QUERY_LLM_MODEL}" \
  --query-embedding-model "${GRAPHRAG_QUERY_EMBEDDING_MODEL}" \
  query \
  --method local \
  --context-only \
  --json \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

Here `--method local` is GraphRAG's upstream entity-centric query algorithm;
the public comparison method remains `graphrag`.

PaperQA context from an OpenAI-built index:

```bash
.venv-paperqa/bin/python pipelines/query_paperqa.py \
  --index rebuilt_indexes/paperqa/docs.pkl \
  --base-url https://api.openai.com/v1 \
  query \
  --context-only \
  --embedding "${PAPERQA_EMBEDDING_MODEL}" \
  --llm "${PAPERQA_LLM_MODEL}" \
  --summary-llm "${PAPERQA_SUMMARY_MODEL}" \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

GEMS-RAG:

```bash
python pipelines/query_gems_rag.py \
  --python .venv-gems-rag/bin/python \
  retrieve \
  --mode no_visual \
  --question "What shape is a STOP sign?" \
  --top-k 10
```

Every adapter emits one JSON object to stdout.

## Run and resume the comparison

Run the three comparison systems against OpenAI-built indexes:

```bash
python pipelines/run_comparison.py \
  --methods bm25,graphrag,paperqa \
  --graphrag-working-dir rebuilt_indexes/graphrag \
  --paperqa-index rebuilt_indexes/paperqa/docs.pkl \
  --output runs/comparison \
  --top-k 10 \
  --base-url https://api.openai.com/v1
```

Add GEMS-RAG when that comparison is required:

```bash
python pipelines/run_comparison.py \
  --methods bm25,graphrag,paperqa,gems-rag \
  --graphrag-working-dir rebuilt_indexes/graphrag \
  --paperqa-index rebuilt_indexes/paperqa/docs.pkl \
  --gems-rag-mode no_visual \
  --output runs/comparison-with-gems-rag \
  --top-k 10 \
  --base-url https://api.openai.com/v1
```

Every completed `(question_id, method)` pair is written atomically to its own
file under `runs/.../rows/`. `results.jsonl` and `state.json` are regenerated
after each pair. Rerunning the same command skips completed pairs.

Failed rows remain available for diagnosis. Retry them with `--retry-errors`.
The run manifest hashes every retrieval-affecting setting and rejects reuse of
an output directory with a different corpus, method list, API configuration,
or retrieval depth.

## Score retrieval

After a run:

```bash
python pipelines/score_retrieval.py \
  --results runs/comparison/results.jsonl \
  --output runs/comparison/retrieval_metrics
```

The scorer writes normalized rankings, qrels, per-question metrics, a summary,
and qrel coverage diagnostics. Primary metrics are:

- Recall@10
- MRR@10
- binary nDCG@10

GraphRAG contexts are mapped back to canonical chunks through exact spans in
the packaged GraphRAG input. BM25, PaperQA, and GEMS-RAG use stable chunk
identifiers.

## Reuse retrieval across answer configurations

This is an injected-context comparison. Retrieval output is computed once per
retrieval configuration and can be reused across downstream answer
configurations.

Reuse is valid only while these inputs remain fixed:

- corpus and built index;
- method configuration;
- retrieval depth;
- API configuration used by retrieval; and
- any retrieval-time provider calls performed by the method.

The runner persists the exact retrieved evidence so resumed or subsequent
answer runs use the same context.

## Package for a 512 MB upload limit

Create the single standard ZIP file. The packager verifies the release
checksums, ZIP inventory, ZIP CRCs, and a hard 500,000,000-byte limit before
atomically publishing the result:

```bash
python scripts/package_upload.py --force
```

The output is:

```text
mutcd-rag-anonymous-release.zip
```

Extract and verify it with:

```bash
mkdir assembled
unzip -q mutcd-rag-anonymous-release.zip -d assembled
cd assembled/mutcd-rag-anonymous-release
shasum -a 256 -c CHECKSUMS.sha256
```

## Integrity

`RELEASE_MANIFEST.json` records every included file, byte size, and SHA-256
digest. `CHECKSUMS.sha256` provides standard checksum-file verification:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

## Licensing

GraphRAG's MIT license and PaperQA's Apache-2.0 license are retained in their
respective source directories. See `THIRD_PARTY_NOTICES.md`.

The GEMS-RAG source snapshot did not contain a project license. This release
does not infer one. Add the intended license before public redistribution.
The MUTCD manual and derived artifacts are separate from these software
licenses; verify the applicable publication terms for the intended channel.

## Anonymous-release hygiene

The release builder rejects:

- Git metadata and remotes;
- machine-specific home paths;
- copied credential files and recognized key patterns;
- historical public method IDs;
- manuscript drafts, evaluation outputs, and grading runs;
- stale absolute media paths; and
- release-authored instructions for non-API execution.

Third-party attribution and the parent GEMS-RAG source are retained.
