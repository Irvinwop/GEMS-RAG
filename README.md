# GEM-RAGs

Harness workspace for running RAG ablation experiments across model providers, retrieval strategies, and grading configurations.

Local-only inputs are intentionally ignored:

- `data/raw/` stores downloaded datasets and archives.
- `manuscript-draft/` stores the current manuscript draft.
- `external/MRAG_stp2/` stores the cloned reference implementation.

The planned harness should make it cheap to compare:

- automatic context injection versus model-driven data exploration through tool calls
- different RAG pipelines, including vector database baselines
- model families and sizes across Anthropic, Grok, OpenAI, Qwen, and local runners
- grader configurations, with the current expectation that grading uses a high-reasoning GPT-5.5/5.6-class model when available
