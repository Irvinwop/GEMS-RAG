#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
PYTHON_DEFAULT="${PYTHON:-python3.12}"
PYTHON_GRAPHRAG="${PYTHON_GRAPHRAG:-python3.13}"

initialize_indexes() {
  python3 "${ROOT}/scripts/initialize_indexes.py"
}

setup_graphrag() {
  "${PYTHON_GRAPHRAG}" -m venv "${ROOT}/.venv-graphrag"
  "${ROOT}/.venv-graphrag/bin/python" -m pip install --upgrade pip
  local packages=()
  local package
  for package in "${ROOT}"/third_party/graphrag/packages/*; do
    if [[ -f "${package}/pyproject.toml" ]]; then
      packages+=("-e" "${package}")
    fi
  done
  "${ROOT}/.venv-graphrag/bin/python" -m pip install "${packages[@]}"
}

setup_paperqa() {
  "${PYTHON_DEFAULT}" -m venv "${ROOT}/.venv-paperqa"
  "${ROOT}/.venv-paperqa/bin/python" -m pip install --upgrade pip
  "${ROOT}/.venv-paperqa/bin/python" -m pip install \
    -e "${ROOT}/third_party/paperqa"
}

setup_gems_rag() {
  "${PYTHON_DEFAULT}" -m venv "${ROOT}/.venv-gems-rag"
  "${ROOT}/.venv-gems-rag/bin/python" -m pip install --upgrade pip
  "${ROOT}/.venv-gems-rag/bin/python" -m pip install \
    -r "${ROOT}/gems-rag/requirements.txt"
  "${ROOT}/.venv-gems-rag/bin/python" -m pip install \
    "zstandard>=0.23,<1" \
    "pymupdf==1.28.0"
  "${ROOT}/.venv-gems-rag/bin/python" \
    "${ROOT}/scripts/materialize_visual_assets.py"
}

case "${TARGET}" in
  all)
    initialize_indexes
    setup_graphrag
    setup_paperqa
    setup_gems_rag
    ;;
  bm25)
    echo "BM25 uses only the Python standard library."
    ;;
  graphrag)
    initialize_indexes
    setup_graphrag
    ;;
  paperqa)
    initialize_indexes
    setup_paperqa
    ;;
  gems-rag)
    initialize_indexes
    setup_gems_rag
    ;;
  indexes)
    initialize_indexes
    ;;
  *)
    echo "usage: $0 [all|indexes|bm25|graphrag|paperqa|gems-rag]" >&2
    exit 2
    ;;
esac
