#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HARNESS_PYTHON="${HARNESS_PYTHON:-.venv/bin/python}"
GRAPHRAG_BASE_PYTHON="${GRAPHRAG_BASE_PYTHON:-python3.13}"
GFMRAG_BASE_PYTHON="${GFMRAG_BASE_PYTHON:-python3.12}"
DPR_BASE_PYTHON="${DPR_BASE_PYTHON:-python3.12}"
MRAG_REFERENCE_BASE_PYTHON="${MRAG_REFERENCE_BASE_PYTHON:-python3.12}"
MEGARAG_BASE_PYTHON="${MEGARAG_BASE_PYTHON:-python3.12}"
HIPPORAG_BASE_PYTHON="${HIPPORAG_BASE_PYTHON:-python3.12}"
VISRAG_BASE_PYTHON="${VISRAG_BASE_PYTHON:-python3.12}"
BOOTSTRAP_HEAVY_RAGS="${BOOTSTRAP_HEAVY_RAGS:-0}"
GRAPHRAG_ENV_PYTHON="data/working/venvs/graphrag/bin/python"
MRAG_REFERENCE_ENV_PYTHON="data/working/venvs/mrag-reference/bin/python"
HIPPORAG_ENV_PYTHON="data/working/venvs/hipporag/bin/python"
VISRAG_ENV_PYTHON="data/working/venvs/visrag/bin/python"
DPR_ENV_PYTHON="data/working/venvs/dpr/bin/python"
GFMRAG_ENV_PYTHON="data/working/venvs/gfmrag/bin/python"
MEGARAG_ENV_PYTHON="data/working/venvs/megarag/bin/python"
MEGARAG_LIGHTRAG_REPO="external/rag-implementations/megarag-lightrag-v1.4.3"
GFMRAG_REPO="external/rag-implementations/gfm-rag"
GFMRAG_PATCH="$ROOT/patches/gfmrag-retrieval-only.patch"
HIPPORAG_REPO="external/rag-implementations/hipporag"
HIPPORAG_PATCH="$ROOT/patches/hipporag-lazy-optional-backends.patch"

apply_external_patch() {
  local repo="$1"
  local patch="$2"
  if git -C "$repo" apply --check "$patch"; then
    git -C "$repo" apply "$patch"
  elif git -C "$repo" apply --reverse --check "$patch"; then
    return 0
  else
    printf 'Patch does not apply cleanly: %s\n' "$patch" >&2
    return 1
  fi
}

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
  if [[ ! -d "$MEGARAG_LIGHTRAG_REPO/.git" ]]; then
    git clone --depth 1 --branch v1.4.3 https://github.com/HKUDS/LightRAG.git "$MEGARAG_LIGHTRAG_REPO"
  fi

  "$MEGARAG_BASE_PYTHON" -m venv data/working/venvs/megarag
  "$MEGARAG_ENV_PYTHON" -m pip install --upgrade pip
  "$MEGARAG_ENV_PYTHON" -m pip install \
    'numpy==1.26.4' \
    'torch>=2.6,<2.7' \
    'torchvision>=0.21,<0.22' \
    'transformers==4.51.3' \
    'openai==1.97.0' \
    'accelerate==1.9.0' \
    'beautifulsoup4==4.13.4' \
    'matplotlib>=3.10,<3.12' \
    'rich>=14,<15' \
    'aiohttp>=3.11,<4' \
    'configparser>=7,<8' \
    'dotenv>=0.9,<1' \
    'future>=1,<2' \
    'nano-vectordb==0.0.4.3' \
    'pandas>=2.2,<3' \
    'Pillow>=11,<12' \
    'pipmaster>=0.9,<1' \
    'pydantic>=2.10,<3' \
    'python-dotenv>=1.1,<2' \
    'pyuca>=1.2,<2' \
    'PyYAML>=6,<7' \
    'tenacity>=9,<10' \
    'tiktoken>=0.9,<0.14' \
    'xlsxwriter>=3.1,<4'
  "$MEGARAG_ENV_PYTHON" -m pip install --no-deps -e "$MEGARAG_LIGHTRAG_REPO"
  "$MEGARAG_ENV_PYTHON" -m pip install --no-deps -e external/rag-implementations/megarag

  apply_external_patch "$GFMRAG_REPO" "$GFMRAG_PATCH"
  "$GFMRAG_BASE_PYTHON" -m venv data/working/venvs/gfmrag
  "$GFMRAG_ENV_PYTHON" -m pip install --upgrade pip
  "$GFMRAG_ENV_PYTHON" -m pip install \
    'numpy>=1.26,<2.3' \
    'torch>=2.6,<2.7' \
    'transformers>=4.52.4,<4.55' \
    'sentence-transformers==3.4.1' \
    'torch-geometric>=2.4,<2.7' \
    'datasets>=3,<4' \
    'pandas>=2.2,<3' \
    'hydra-core==1.3.2' \
    'ninja>=1.11,<2' \
    'easydict>=1.13,<2'
  "$GFMRAG_ENV_PYTHON" -m pip install --no-deps -e "$GFMRAG_REPO"

  "$DPR_BASE_PYTHON" -m venv data/working/venvs/dpr
  "$DPR_ENV_PYTHON" -m pip install --upgrade pip
  "$DPR_ENV_PYTHON" -m pip install \
    'numpy>=1.26,<2.3' \
    'torch>=2.6,<2.7' \
    'transformers>=4.49,<4.55'

  "$MRAG_REFERENCE_BASE_PYTHON" -m venv data/working/venvs/mrag-reference
  "$MRAG_REFERENCE_ENV_PYTHON" -m pip install --upgrade pip
  "$MRAG_REFERENCE_ENV_PYTHON" -m pip install -r external/MRAG_stp2/requirements.txt

  apply_external_patch "$HIPPORAG_REPO" "$HIPPORAG_PATCH"
  "$HIPPORAG_BASE_PYTHON" -m venv data/working/venvs/hipporag
  "$HIPPORAG_ENV_PYTHON" -m pip install --upgrade pip
  "$HIPPORAG_ENV_PYTHON" -m pip install \
    'numpy==1.26.4' \
    'torch>=2.6,<2.7' \
    'transformers==4.45.2' \
    'openai==1.91.0' \
    'networkx==3.4.2' \
    'pydantic==2.10.4' \
    'python-igraph==0.11.8' \
    'tenacity==8.5.0' \
    'tiktoken==0.7.0' \
    'tqdm>=4.66,<5' \
    'einops>=0.8,<1' \
    'scipy>=1.14,<2' \
    'pandas>=2.2,<3' \
    'filelock>=3.16,<4' \
    'packaging>=24,<27'
  "$HIPPORAG_ENV_PYTHON" -m pip install --no-deps -e "$HIPPORAG_REPO"

  "$VISRAG_BASE_PYTHON" -m venv data/working/venvs/visrag
  "$VISRAG_ENV_PYTHON" -m pip install --upgrade pip
  "$VISRAG_ENV_PYTHON" -m pip install \
    'numpy==1.26.4' \
    'torch>=2.6,<2.7' \
    'torchvision>=0.21,<0.22' \
    'transformers==4.40.2' \
    'accelerate>=0.34,<0.35' \
    'Pillow>=11,<12' \
    'sentencepiece>=0.2,<0.3'
fi
