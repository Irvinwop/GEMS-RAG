# Implementation Inventory

This note records the local state after the initial project import and external RAG acquisition. The source/data folders named here are gitignored; this file is the tracked pointer to what is present locally.

## Local MRAG Reference

Path: `/Users/irvin/Documents/GEM-RAGs/external/MRAG_stp2`

Repository: `https://github.com/hannanazad/MRAG_stp2.git`  
Commit: `c282bf72df0c` (`stp2_v2: restore v6 VLM model switcher; revert notebook`)

Implemented pieces:

- `mrag/parsing.py`: outline-driven section parsing into typed MUTCD chunks.
- `mrag/figures.py`: page rendering plus caption-anchored figure/table crops.
- `mrag/sign_codes.py`: sign-code mining and categorization.
- `mrag/kg.py`: NetworkX multigraph over parts, chapters, sections, chunks, figures, tables, sign codes, and categories.
- `mrag/vector_store.py`: embedded Qdrant collections for text chunks, figure captions, visual figure crops, and page images.
- `mrag/embeddings.py`: BGE-M3 dense+sparse text embeddings, ColQwen/ColPali visual embeddings, and mxbai reranking.
- `mrag/retrieval.py`: hybrid retrieval, graph/metadata scoring, reranking, figure retrieval, and page retrieval.
- `mrag/vlm.py` and `mrag/ask.py`: answer generation with local or OpenAI-compatible VLM APIs.

Current retrieval design:

- Text retrieval: BGE-M3 dense + sparse vectors with reciprocal-rank fusion in Qdrant.
- Graph expansion/scoring: explicit IDs/sign codes, graph proximity, hierarchy prior, and rule-type weighting.
- Reranking: `mixedbread-ai/mxbai-rerank-large-v2`.
- Visual retrieval: ColQwen/ColPali over figure crops and full page renders.
- Generation: defaults to an OpenAI-compatible DashScope/Qwen VLM endpoint, with local Qwen2.5-VL fallback paths in code.

Local patch:

- `external/MRAG_stp2/mrag/parsing.py` was patched locally so future parsing derives MUTCD part membership from section/chapter IDs instead of trusting the PDF outline traversal state.

## Extracted MRAG Data

Path: `/Users/irvin/Documents/GEM-RAGs/data/extracted/MRAG-20260708T114057Z-3/MRAG`

Key artifacts:

- `mmrag_cache_v3/chunks.jsonl`: 5,821 chunks.
- `mmrag_cache_v3/figures.jsonl`: 314 figure/table records.
- `mmrag_cache_v3/sign_codes.json`: 9,270 lines of sign-code dictionary JSON.
- `mmrag_cache_v3/chunks_dense.npy`: BGE-M3 dense chunk vectors.
- `mmrag_cache_v3/chunks_sparse.json`: sparse chunk vectors.
- `mmrag_cache_v3/figures_dense.npy`: dense figure-caption vectors.
- `mmrag_cache_v3/colqwen_pages/`: ColQwen page vectors.
- `mmrag_cache_v3/colqwen_figures/`: ColQwen figure-crop vectors.
- `qdrant_db/`: extracted embedded Qdrant database snapshot.
- `eval/gold_qa.jsonl`: 49 gold questions.
- `eval/runs.jsonl` and `eval/scored.jsonl`: 147 prior generated/scored runs.
- `eval/summary_by_config.csv` and `.xlsx`: aggregate prior eval summaries.

The prior generated/scored runs can be imported into the harness row schema with `gem-rags import-mrag-eval`. The current local import target is `runs/mrag-prior-eval/runs.jsonl`, with reconstructed chunk, figure, and page evidence so the rows can be summarized, compared, validated against `configs/mrag-prior-eval.json`, or regraded beside new ablation runs.

Qdrant collections:

- `mutcd_chunks`: 1024-dim dense vectors plus sparse vectors.
- `mutcd_figures`: 1024-dim dense caption vectors.
- `mutcd_pages`: 128-dim ColPali multivectors with binary quantization.
- `mutcd_figures_visual`: 128-dim ColPali multivectors over figure crops.

Data-quality notes to check before publication-grade ablations:

- The imported cache originally assigned every chunk to `Part 9 Traffic Control For Bicycle Facilities`. This has been repaired in `chunks.jsonl`, `graph.gpickle`, and Qdrant chunk payloads using `scripts/repair_mrag_metadata.py`.
- Only 2 of 314 figure/table records have non-empty `sign_codes_depicted`; figure captions are often minimal, so figure-to-sign grounding may need stronger extraction.
- The gold set is small: 49 questions, with 12 expected refusals and 9 questions with gold figures.

## External RAG Implementations

All repos are cloned shallowly under `/Users/irvin/Documents/GEM-RAGs/external/rag-implementations`.

| Name | Local path | Repository | Commit | Harness role |
| --- | --- | --- | --- | --- |
| GraphRAG | `external/rag-implementations/graphrag` | `https://github.com/microsoft/graphrag.git` | `6d02c2355c3f` | Canonical graph-RAG baseline; expensive indexing, useful as a high-end graph baseline. |
| LightRAG | `external/rag-implementations/lightrag` | `https://github.com/HKUDS/LightRAG.git` | `fedd95ce7db0` | Lightweight graph+vector RAG with API server, multiple storage backends, and multimodal integration hooks. |
| HippoRAG | `external/rag-implementations/hipporag` | `https://github.com/OSU-NLP-Group/HippoRAG.git` | `ef2f14c4f254` | Memory/graph retrieval baseline using OpenIE, dense retrieval, and Personalized PageRank. |
| RAG-Anything | `external/rag-implementations/rag-anything` | `https://github.com/HKUDS/RAG-Anything.git` | `32eef6ecc2cc` | Multimodal document RAG over text, images, tables, and equations; closest external match to mixed-content standards. |
| VisRAG | `external/rag-implementations/visrag` | `https://github.com/OpenBMB/VisRAG.git` | `f35d232d4c6c` | Parsing-free visual document RAG and multi-image VLM reasoning baseline. |
| Self-RAG | `external/rag-implementations/self-rag` | `https://github.com/akariasai/self-rag.git` | `1fcdc420e48f` | Retrieval-control pattern implemented locally as `self_rag_policy`; upstream eval input can be exported with `scripts/export_upstream_eval_inputs.py`. |
| CRAG | `external/rag-implementations/crag` | `https://github.com/HuskyInSalt/CRAG.git` | `de7c2961ae62` | Corrective retrieval pattern implemented locally as `crag_policy`; upstream `question [SEP] passage` eval input can be exported with `scripts/export_upstream_eval_inputs.py`. |
| PaperQA2 | `external/rag-implementations/paper-qa` | `https://github.com/Future-House/paper-qa.git` | `d7675d7b7edd` | Agentic PDF/document RAG with citation-focused answering and LiteLLM model support. |

## Baseline Shape For The Harness

The manuscript baseline names map cleanly to two classes:

- In-house retrieval baselines over the same extracted corpus: BM25, dense vector, hybrid dense+sparse, Qdrant vector DB with model tool calls, and direct context injection.
- External-system baselines via adapters: GraphRAG, LightRAG, HippoRAG, RAG-Anything, VisRAG, PaperQA2, plus Self-RAG/CRAG-style retrieval-control variants.

The most important ablation axis should be explicit in the harness API:

- `context_mode = "injected"`: harness retrieves evidence and shoves it into the model.
- `context_mode = "tool_explore"`: model gets a hit catalog from the selected retriever and chooses what to inspect.
- `context_mode = "tool_search"`: model chooses retrieval/search queries first, then chooses which returned hits to inspect.

That split should be independent of model provider, model size, RAG implementation, retrieval budget, prompt style, and grader model.

See `docs/external-adapters.md` for the adapter boundary and the exported corpus formats used to wire the cloned upstream RAG implementations into the harness.
