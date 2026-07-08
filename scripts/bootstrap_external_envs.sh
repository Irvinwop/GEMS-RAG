#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HARNESS_PYTHON="${HARNESS_PYTHON:-.venv/bin/python}"
GRAPHRAG_BASE_PYTHON="${GRAPHRAG_BASE_PYTHON:-python3.13}"
GRAPHRAG_ENV_PYTHON="data/working/venvs/graphrag/bin/python"

"$HARNESS_PYTHON" -m pip install -e external/rag-implementations/lightrag
"$HARNESS_PYTHON" -m pip install -e external/rag-implementations/paper-qa

"$GRAPHRAG_BASE_PYTHON" -m venv data/working/venvs/graphrag
"$GRAPHRAG_ENV_PYTHON" -m pip install --upgrade pip
"$GRAPHRAG_ENV_PYTHON" -m pip install \
  -e external/rag-implementations/graphrag/packages/graphrag-common \
  -e external/rag-implementations/graphrag/packages/graphrag-storage \
  -e external/rag-implementations/graphrag/packages/graphrag-cache \
  -e external/rag-implementations/graphrag/packages/graphrag-chunking \
  -e external/rag-implementations/graphrag/packages/graphrag-input \
  -e external/rag-implementations/graphrag/packages/graphrag-vectors \
  -e external/rag-implementations/graphrag/packages/graphrag-llm \
  -e external/rag-implementations/graphrag/packages/graphrag

"$HARNESS_PYTHON" scripts/query_graphrag_index.py prepare --force
"$HARNESS_PYTHON" scripts/query_graphrag_index.py init
"$HARNESS_PYTHON" scripts/query_visrag_index.py prepare --scope pages
"$HARNESS_PYTHON" scripts/query_paperqa_index.py index --defer-embedding
