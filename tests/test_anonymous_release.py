from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_method_ids_are_normalized() -> None:
    config = json.loads(
        (ROOT / "anonymous_release" / "configs" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(config["available_methods"]) == {
        "bm25",
        "graphrag",
        "paperqa",
        "gems-rag",
    }
    assert config["default_methods"] == ["bm25", "graphrag", "paperqa"]


def test_release_templates_do_not_contain_historical_method_ids() -> None:
    forbidden = ("graphrag_local", "graphrag-local", "paperqa2_chunks", "gems_full")
    for path in (ROOT / "anonymous_release").rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not any(value in text for value in forbidden), path


def test_public_gems_rag_source_has_a_named_repository_folder() -> None:
    readme = (ROOT / "anonymous_release" / "README.md").read_text(encoding="utf-8")
    setup = (
        ROOT / "anonymous_release" / "scripts" / "setup_environments.sh"
    ).read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_anonymous_release.py").read_text(
        encoding="utf-8"
    )

    assert "├── gems-rag/" in readme
    assert "\n├── mrag/" not in readme
    assert "│   ├── mrag/" in readme
    assert '"${ROOT}/gems-rag/requirements.txt"' in setup
    assert 'source_root = stage / "gems-rag"' in builder
    assert '"mrag_reference", "gems_rag_reference"' in builder


def test_media_paths_are_made_release_relative() -> None:
    builder = load_module(
        "build_anonymous_release",
        ROOT / "scripts" / "build_anonymous_release.py",
    )
    assert (
        builder.relative_media_path(
            "/content/drive/MyDrive/MRAG/figures/figure_2B-1.png"
        )
        == "figures/figure_2B-1.png"
    )
    assert (
        builder.relative_media_path(
            "/content/drive/MyDrive/MRAG/page_images/page_0042.png"
        )
        == "page_images/page_0042.png"
    )


def test_bm25_emits_stable_chunk_identifiers() -> None:
    bm25 = load_module(
        "release_query_bm25",
        ROOT / "anonymous_release" / "pipelines" / "query_bm25.py",
    )
    documents = [
        {
            "doc_id": "chunk-1",
            "title": "STOP sign",
            "text": "A STOP sign is octagonal.",
            "metadata": {"section_id": "2B.04"},
        },
        {
            "doc_id": "chunk-2",
            "title": "YIELD sign",
            "text": "A YIELD sign is triangular.",
            "metadata": {"section_id": "2B.05"},
        },
    ]
    contexts = bm25.BM25Index(documents).query("STOP octagonal", top_k=1)
    assert contexts[0]["name"] == "chunk-1"
    assert contexts[0]["metadata"]["chunk_id"] == "chunk-1"
