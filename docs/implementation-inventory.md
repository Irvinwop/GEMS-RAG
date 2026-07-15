# Implementation Inventory

This note records the local state after the initial project import and external RAG acquisition. The source/data folders named here are gitignored; this file is the tracked pointer to what is present locally.

## Local MRAG Reference

Path: `/Users/irvin/Documents/GEMS-RAG/external/MRAG_stp2`

Repository: `https://github.com/hannanazad/MRAG_stp2.git`  
Commit: `1d534ea93ee6` (latest upstream v4/MUTCD-150 bundle as of 2026-07-15)

Implemented pieces:

- `mrag/parsing.py`: outline-driven section parsing into typed MUTCD chunks.
- `mrag/figures.py`: page rendering plus caption-anchored figure/table crops.
- `mrag/sign_codes.py`: sign-code mining and categorization.
- `mrag/kg.py`: NetworkX multigraph over parts, chapters, sections, chunks, figures, tables, sign codes, and categories.
- `mrag/vector_store.py`: embedded Qdrant collections for text chunks, figure captions, visual figure crops, and page images.
- `mrag/embeddings.py`: BGE-M3 dense+sparse text embeddings, ColQwen/ColPali visual embeddings, and mxbai reranking.
- `mrag/retrieval.py`: hybrid retrieval, graph/metadata scoring, reranking, figure retrieval, and page retrieval.
- `mrag/question_router.py`: query-time routing that suppresses visual retrieval for questions that do not need figures.
- `mrag/vlm.py` and `mrag/ask.py`: answer generation with local or OpenAI-compatible VLM APIs.
- `benchmarks/mutcd150/v1/`: 150 immutable benchmark questions plus the resumable upstream runner; the public repository intentionally excludes gold answers and evaluator annotations.

Current retrieval design:

- Text retrieval: BGE-M3 dense + sparse vectors with reciprocal-rank fusion in Qdrant.
- Graph expansion/scoring: explicit IDs/sign codes, graph proximity, hierarchy prior, and rule-type weighting.
- Reranking: `mixedbread-ai/mxbai-rerank-large-v2`.
- Visual retrieval: ColQwen/ColPali over figure crops and full page renders.
- Generation: defaults to an OpenAI-compatible DashScope/Qwen VLM endpoint, with local Qwen2.5-VL fallback paths in code.

Harness correction and ablation boundary:

- `scripts/query_mrag_reference.py` exposes isolated `dense`, `hybrid`, `multimodal`, and `full` modes plus the manuscript's four component-removal variants.
- `scripts/query_dpr_index.py` builds a shared-corpus index with the original DPR context/question checkpoints for both the cited DPR retriever and canonical RAG retrieval condition.
- `scripts/query_gfmrag_index.py` exports all 8,248 repaired graph nodes and 18,245 edges through GFM-RAG's official bring-your-own-graph schema, then queries the official pretrained graph retriever.
- `scripts/query_megarag_index.py` exports all 1,162 page renders, 5,705 canonical chunks, and the local figure/table records to MegaRAG's page schema, then exposes its official MMKG and page-image retrieval branches without an internal final answer call.
- `gems-rag manual-status` verifies the source MUTCD PDF and writes a checksum-bearing lineage manifest for all 19 required manuscript systems. `shared_corpus` is the controlled default; `native_pdf` invokes the official PaperQA2 and RAG-Anything parsers, while MegaRAG and VisRAG retain their native PDF/page-render paths.
- The tracked full-mode implementation adds graph-neighbor chunks to the candidate set before scoring. This repairs the upstream `pass` placeholder that previously labeled retrieval as graph expansion without adding graph candidates.
- Mode-specific checks require only the dependencies used by that mode, preventing dense/hybrid rows from being blocked by or silently conflated with missing visual components.

Local patch:

- `external/MRAG_stp2/mrag/parsing.py` was patched locally so future parsing derives MUTCD part membership from section/chapter IDs instead of trusting the PDF outline traversal state.

## Extracted MRAG Data

Active path: `/Users/irvin/Documents/GEMS-RAG/data/extracted/MRAG-20260715T174043Z-1/MRAG`

Key artifacts:

- `mmrag_cache_v3/chunks.jsonl`: 5,812 raw rows, canonicalized by the harness to 5,705 unique chunk IDs.
- `mmrag_cache_v3/figures.jsonl`: 727 figure/table image records; Qdrant contains 553 canonical figure-caption and visual points.
- `mmrag_cache_v3/sign_codes.json`: 9,270 lines of sign-code dictionary JSON.
- `mmrag_cache_v3/chunks_dense.npy`: BGE-M3 dense chunk vectors.
- `mmrag_cache_v3/chunks_sparse.json`: sparse chunk vectors.
- `mmrag_cache_v3/figures_dense.npy`: dense figure-caption vectors.
- `mmrag_cache_v3/colqwen_pages/`: ColQwen page vectors.
- `mmrag_cache_v3/colqwen_figures/`: ColQwen figure-crop vectors.
- `qdrant_db/`: restored embedded Qdrant v4 database snapshot.
- `hf_cache/`: restored BGE-M3, mxbai, and ColQwen model cache with six checksum-verified detached weight blobs.
- `eval/gold_qa.jsonl`: 49 gold questions.
- `eval/runs.jsonl` and `eval/scored.jsonl`: 147 prior generated/scored runs.
- `eval/summary_by_config.csv` and `.xlsx`: aggregate prior eval summaries.

The prior generated/scored runs can be imported into the harness row schema with `gems-rag import-mrag-eval`. The current local import target is `runs/mrag-prior-eval/runs.jsonl`, with reconstructed chunk, figure, and page evidence so the rows can be summarized, compared, validated against `configs/mrag-prior-eval.json`, or regraded beside new ablation runs.

Qdrant collections:

- `mutcd_chunks`: 5,705 points with 1024-dim dense vectors plus sparse vectors.
- `mutcd_figures`: 553 points with 1024-dim dense caption vectors.
- `mutcd_pages`: 1,162 points with 128-dim ColPali multivectors and binary quantization.
- `mutcd_figures_visual`: 553 points with 128-dim ColPali multivectors over figure crops.

Data-quality notes to check before publication-grade ablations:

- The imported cache originally assigned every chunk to `Part 9 Traffic Control For Bicycle Facilities`. This has been repaired in `chunks.jsonl`, `graph.gpickle`, and Qdrant chunk payloads using `scripts/repair_mrag_metadata.py`.
- Only 1 of 727 figure/table records has non-empty `sign_codes_depicted`; figure captions are often minimal, so figure-to-sign grounding may need stronger extraction.
- The raw chunk cache contains 107 collision rows across 36 repeated chunk IDs. `gems_rag.data.load_chunks` and `scripts/export_mrag_corpus.py` deterministically retain the most information-rich row for each ID and report 5,705 unique chunks. This prevents duplicate IDs and noisy later table fragments from entering local or exported indexes.
- The gold set is small: 49 questions, with 12 expected refusals and 9 questions with gold figures.
- The updated upstream clone also contains 150 immutable MUTCD-150 questions. They are the GUI default but remain explicitly question-only: generated upstream answers are not used as gold, and grading ZIPs include the source MUTCD manual as authority. The curated 49-record gold set remains selectable for answer- and reference-paired experiments.

## External RAG Implementations

All repos are cloned shallowly under `/Users/irvin/Documents/GEMS-RAG/external/rag-implementations`.

| Name | Local path | Repository | Commit | Harness role |
| --- | --- | --- | --- | --- |
| GraphRAG | `external/rag-implementations/graphrag` | `https://github.com/microsoft/graphrag.git` | `6d02c2355c3f` | Canonical graph-RAG baseline; expensive indexing, useful as a high-end graph baseline. |
| LightRAG | `external/rag-implementations/lightrag` | `https://github.com/HKUDS/LightRAG.git` | `fedd95ce7db0` | Lightweight graph+vector RAG with API server, multiple storage backends, and multimodal integration hooks. |
| HippoRAG | `external/rag-implementations/hipporag` | `https://github.com/OSU-NLP-Group/HippoRAG.git` | `ef2f14c4f254` | Memory/graph retrieval baseline using OpenIE, dense retrieval, and Personalized PageRank. |
| RAG-Anything | `external/rag-implementations/rag-anything` | `https://github.com/HKUDS/RAG-Anything.git` | `32eef6ecc2cc` | Multimodal document RAG over text, images, tables, and equations; closest external match to mixed-content standards. |
| VisRAG | `external/rag-implementations/visrag` | `https://github.com/OpenBMB/VisRAG.git` | `f35d232d4c6c` | Parsing-free visual document RAG and multi-image VLM reasoning baseline. |
| Self-RAG | `external/rag-implementations/self-rag` | `https://github.com/akariasai/self-rag.git` | `1fcdc420e48f` | Retrieval-control pattern implemented locally as `self_rag_policy`; upstream eval input can be exported with `gems-rag upstream-inputs`. |
| CRAG | `external/rag-implementations/crag` | `https://github.com/HuskyInSalt/CRAG.git` | `de7c2961ae62` | Corrective retrieval pattern implemented locally as `crag_policy`; upstream `question [SEP] passage` eval input can be exported with `gems-rag upstream-inputs`. |
| PaperQA2 | `external/rag-implementations/paper-qa` | `https://github.com/Future-House/paper-qa.git` | `d7675d7b7edd` | Agentic PDF/document RAG with citation-focused answering and LiteLLM model support. |
| DPR | `external/rag-implementations/dpr` | `https://github.com/facebookresearch/DPR.git` | `a31212dc0a54` | Original-team dense retriever used by canonical RAG; archived upstream. |
| SAM-RAG | `external/rag-implementations/sam-rag` | `https://github.com/SAM-RAG/SAM_RAG.git` | `5fdb1c656b09` | Original self-adaptive multimodal retrieval flow; upstream warns that the code is not ready for use. |
| LPKG | `external/rag-implementations/lpkg` | `https://github.com/zjukg/LPKG.git` | `8379a2e362f8` | Learned planning model and planning-output parser for iterative KG-backed retrieval. |
| KG2RAG | `external/rag-implementations/kg2rag` | `https://github.com/nju-websoft/KG2RAG.git` | `7d626c77b7af` | Knowledge-graph-guided seed expansion and evidence organization. |
| GFM-RAG | `external/rag-implementations/gfm-rag` | `https://github.com/RManLuo/gfm-rag.git` | `57e3e28045ff` | Graph foundation-model retriever with a bring-your-own-graph interface. |
| MegaRAG | `external/rag-implementations/megarag` | `https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git` | `ca7c627c1e88` | Multimodal knowledge-graph retrieval over document text and page imagery; custom upstream license. |
| MegaRAG LightRAG dependency | `external/rag-implementations/megarag-lightrag-v1.4.3` | `https://github.com/HKUDS/LightRAG.git` | `0171e0ce20e7` (`v1.4.3`) | Exact dependency revision required by the official MegaRAG installation instructions; isolated from the standalone newer LightRAG baseline. |

`configs/manuscript-rags.json` is the source-of-truth crosswalk from every RAG system, explicit baseline, and survey citation in the manuscript to its upstream provenance and harness retriever names. Every coverage-required entry now has a concrete retriever integration. Readiness remains explicit: heavy methods can still require ignored environments, credentials, model downloads, indexes, or normalized planner output before a given sweep is runnable. The LPKG entry uses the original generated-plan syntax and requires normalized per-question planner output because its authors released training data and scripts but no trained planner checkpoint.

`gems-rag manuscript-coverage` enforces this crosswalk against the retriever catalog. It fails when the audited 19-method set changes unexpectedly, a required entry is not marked integrated, a named retriever is missing or disabled, upstream provenance is incomplete, or a `manuscript-system` retriever is orphaned.

## Baseline Shape For The Harness

The manuscript baseline names map cleanly to two classes:

- In-house retrieval baselines over the same extracted corpus: BM25, dense vector, hybrid dense+sparse, Qdrant vector DB with model tool calls, and direct context injection.
- External-system baselines via adapters: GraphRAG, LightRAG, HippoRAG, RAG-Anything, VisRAG, PaperQA2, plus Self-RAG/CRAG-style retrieval-control variants.

The most important ablation axis should be explicit in the harness API:

- `context_mode = "injected"`: harness retrieves evidence and shoves it into the model.
- `context_mode = "tool_explore"`: model gets a hit catalog from the selected retriever and chooses what to inspect.
- `context_mode = "tool_search"`: model chooses retrieval/search queries first, then chooses which returned hits to inspect.
- `context_mode = "tool_native"`: model chooses searches and opens evidence through actual provider function calls.

That split should be independent of model provider, model size, RAG implementation, retrieval budget, prompt style, and grader model.

See `docs/external-adapters.md` for the adapter boundary and the exported corpus formats used to wire the cloned upstream RAG implementations into the harness.
