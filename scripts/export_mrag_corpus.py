#!/usr/bin/env python3
"""Export repaired MRAG artifacts into common external-RAG input formats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunk_doc(chunk: dict) -> dict:
    doc_id = chunk["chunk_id"]
    title = f"Section {chunk.get('section_id')} {chunk.get('content_type')} {chunk.get('ordinal')} - {chunk.get('section_title')}"
    text = (
        f"{title}\n"
        f"Part: {chunk.get('part')}\n"
        f"Chapter: {chunk.get('chapter')}\n"
        f"Page: {chunk.get('page_printed')}\n\n"
        f"{chunk.get('text', '')}"
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "metadata": {
            "section_id": chunk.get("section_id"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "ordinal": chunk.get("ordinal"),
            "page_pdf": chunk.get("page_pdf"),
            "page_printed": chunk.get("page_printed"),
            "part": chunk.get("part"),
            "chapter": chunk.get("chapter"),
            "figure_refs": chunk.get("figure_refs", []),
            "table_refs": chunk.get("table_refs", []),
            "section_refs": chunk.get("section_refs", []),
            "sign_codes": chunk.get("sign_codes", []),
        },
    }


def raganything_text_item(doc: dict, page_idx: int) -> dict:
    return {
        "type": "text",
        "text": doc["text"],
        "page_idx": page_idx,
        "metadata": {"doc_id": doc["doc_id"], **doc["metadata"]},
    }


def raganything_figure_item(figure: dict) -> dict:
    kind = "table" if str(figure.get("kind")).lower() == "table" else "image"
    base = {
        "type": kind,
        "page_idx": int(figure.get("page_pdf") or 0) - 1,
        "metadata": {
            "figure_id": figure.get("figure_id"),
            "page_pdf": figure.get("page_pdf"),
            "page_printed": figure.get("page_printed"),
            "sign_codes": figure.get("sign_codes_depicted", []),
        },
    }
    caption = figure.get("caption") or figure.get("title") or figure.get("figure_id")
    if kind == "table":
        base["table_body"] = caption
        base["table_caption"] = [caption]
    else:
        base["img_path"] = str(Path(figure.get("image_path", "")).resolve())
        base["image_caption"] = [caption]
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrag-dir", type=Path, default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/working/mrag_corpus"))
    parser.add_argument("--max-lightrag-chunks", type=int, default=0, help="0 exports all chunks.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [chunk_doc(row) for row in read_jsonl(args.mrag_dir / "mmrag_cache_v3" / "chunks.jsonl")]
    figures = list(read_jsonl(args.mrag_dir / "mmrag_cache_v3" / "figures.jsonl"))

    with (args.out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for doc in chunks:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    light_docs = chunks[: args.max_lightrag_chunks] if args.max_lightrag_chunks else chunks
    with (args.out_dir / "lightrag_corpus.txt").open("w", encoding="utf-8") as handle:
        for doc in light_docs:
            handle.write(f"\n\n===== {doc['doc_id']} =====\n{doc['text']}\n")

    content_list = [raganything_text_item(doc, int(doc["metadata"].get("page_pdf") or 0) - 1) for doc in chunks]
    content_list.extend(raganything_figure_item(figure) for figure in figures)
    (args.out_dir / "raganything_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "mrag_dir": str(args.mrag_dir),
        "chunks": len(chunks),
        "figures": len(figures),
        "outputs": {
            "chunks_jsonl": str(args.out_dir / "chunks.jsonl"),
            "lightrag_corpus_txt": str(args.out_dir / "lightrag_corpus.txt"),
            "raganything_content_list_json": str(args.out_dir / "raganything_content_list.json"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
