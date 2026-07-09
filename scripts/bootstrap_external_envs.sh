#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HARNESS_PYTHON="${HARNESS_PYTHON:-.venv/bin/python}"
GRAPHRAG_BASE_PYTHON="${GRAPHRAG_BASE_PYTHON:-python3.13}"
HEAVY_BASE_PYTHON="${HEAVY_BASE_PYTHON:-python3}"
BOOTSTRAP_HEAVY_RAGS="${BOOTSTRAP_HEAVY_RAGS:-0}"
GRAPHRAG_ENV_PYTHON="data/working/venvs/graphrag/bin/python"
MRAG_REFERENCE_ENV_PYTHON="data/working/venvs/mrag-reference/bin/python"
HIPPORAG_ENV_PYTHON="data/working/venvs/hipporag/bin/python"
VISRAG_ENV_PYTHON="data/working/venvs/visrag/bin/python"

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

if [[ "$BOOTSTRAP_HEAVY_RAGS" == "1" ]]; then
  "$HEAVY_BASE_PYTHON" -m venv data/working/venvs/mrag-reference
  "$MRAG_REFERENCE_ENV_PYTHON" -m pip install --upgrade pip
  "$MRAG_REFERENCE_ENV_PYTHON" -m pip install -r external/MRAG_stp2/requirements.txt

  "$HEAVY_BASE_PYTHON" -m venv data/working/venvs/hipporag
  "$HIPPORAG_ENV_PYTHON" -m pip install --upgrade pip
  "$HIPPORAG_ENV_PYTHON" -m pip install -r external/rag-implementations/hipporag/requirements.txt
  "$HIPPORAG_ENV_PYTHON" -m pip install -e external/rag-implementations/hipporag

  "$HEAVY_BASE_PYTHON" -m venv data/working/venvs/visrag
  "$VISRAG_ENV_PYTHON" -m pip install --upgrade pip
  "$VISRAG_ENV_PYTHON" -m pip install -r external/rag-implementations/visrag/requirements.txt
  "$VISRAG_ENV_PYTHON" -m pip install -e external/rag-implementations/visrag
fi
