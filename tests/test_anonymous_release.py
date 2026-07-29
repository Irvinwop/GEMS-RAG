from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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

    assert "|-- gems-rag/" in readme
    assert "\n|-- mrag/" not in readme
    assert "|   |-- mrag/" in readme
    assert '"${ROOT}/gems-rag/requirements.txt"' in setup
    assert 'source_root = stage / "gems-rag"' in builder
    assert '"mrag_reference", "gems_rag_reference"' in builder


def test_release_authored_surfaces_do_not_advertise_local_execution() -> None:
    forbidden = (
        "ollama",
        "nomic",
        "qwen2.5:",
        "127.0.0.1",
        "localhost",
        "local model",
        "model weights",
        "local openai-compatible",
        "huggingface/",
    )
    roots = (
        ROOT / "anonymous_release" / "README.md",
        ROOT / "anonymous_release" / ".env.example",
        ROOT / "anonymous_release" / "configs",
        ROOT / "anonymous_release" / "pipelines",
        ROOT / "anonymous_release" / "scripts",
    )
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            assert not any(value in text for value in forbidden), path


def test_upload_packager_creates_disjoint_standard_zip_parts(tmp_path: Path) -> None:
    packager = load_module(
        "release_package_upload",
        ROOT / "anonymous_release" / "scripts" / "package_upload.py",
    )
    release = tmp_path / "mutcd-rag-anonymous-release"
    files = {
        "RELEASE_MANIFEST.json": "{}\n",
        "CHECKSUMS.sha256": "checksums\n",
        (
            "indexes/gems-rag/qdrant_db/collection/"
            "mutcd_pages/storage.sqlite"
        ): "page vectors\n",
        (
            "indexes/gems-rag/qdrant_db/collection/"
            "mutcd_figures_visual/storage.sqlite"
        ): "figure vectors\n",
        "indexes/gems-rag/page_images/page_0001.png": "page image\n",
        "indexes/gems-rag/figures/figure_1.png": "figure image\n",
    }
    for relative, content in files.items():
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    output = tmp_path / "upload"
    records = packager.build_upload_parts(
        release=release,
        output=output,
        max_part_bytes=100_000,
        force=False,
    )

    assert len(records) == 4
    observed: set[str] = set()
    for record in records:
        archive = output / record["filename"]
        assert zipfile.is_zipfile(archive)
        assert archive.stat().st_size <= 100_000
        with zipfile.ZipFile(archive) as bundle:
            entries = {
                name
                for name in bundle.namelist()
                if name != packager.UPLOAD_README_NAME
            }
        assert not observed.intersection(entries)
        observed.update(entries)
    assert len(observed) == len(files)


def test_upload_packager_rejects_release_ancestor_as_output(
    tmp_path: Path,
) -> None:
    packager = load_module(
        "release_package_upload_path_guard",
        ROOT / "anonymous_release" / "scripts" / "package_upload.py",
    )
    release = tmp_path / "mutcd-rag-anonymous-release"
    release.mkdir()
    (release / "RELEASE_MANIFEST.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        packager.build_upload_parts(
            release=release,
            output=tmp_path,
            max_part_bytes=500_000_000,
            force=True,
        )

    assert release.is_dir()


def test_index_builder_routes_graph_and_paperqa_through_openai_api(
    tmp_path: Path,
) -> None:
    builder = load_module(
        "release_build_indexes_openai",
        ROOT / "anonymous_release" / "pipelines" / "build_indexes_openai.py",
    )
    args = SimpleNamespace(
        methods=("graphrag", "paperqa"),
        output=tmp_path / "indexes",
        corpus=tmp_path / "chunks.jsonl",
        graphrag_python=Path("/python-graphrag"),
        paperqa_python=Path("/python-paperqa"),
        graphrag_completion_id="completion-id",
        graphrag_embedding_id="graph-embedding-id",
        paperqa_embedding_id="paper-embedding-id",
    )

    commands = builder.commands(args)
    flattened = [token for _, command in commands for token in command]
    assert builder.OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert flattened.count(builder.OPENAI_BASE_URL) >= 4
    assert "GRAPHRAG_API_KEY" not in flattened
    assert "OPENAI_API_KEY" in flattened


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
