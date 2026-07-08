#!/usr/bin/env python3
"""Repair deterministic MRAG metadata issues in local extracted artifacts.

The imported MRAG cache has correct chapter IDs but every chunk was assigned
to Part 9. MUTCD part membership is encoded in the leading digit of each
section/chapter ID, so this repair can be done without re-running extraction or
embedding jobs.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any


PART_TITLES = {
    "1": "Part 1 General",
    "2": "Part 2 Signs",
    "3": "Part 3 Markings",
    "4": "Part 4 Highway Traffic Signals",
    "5": "Part 5 Traffic Control Device Considerations For Automated Vehicles",
    "6": "Part 6 Temporary Traffic Control",
    "7": "Part 7 Traffic Control For School Areas",
    "8": "Part 8 Traffic Control For Railroad And Light Rail Transit Grade Crossings",
    "9": "Part 9 Traffic Control For Bicycle Facilities",
}


def part_for_section(section_id: str | None) -> str | None:
    if not section_id:
        return None
    match = re.match(r"^(\d+)", section_id.strip())
    if not match:
        return None
    return PART_TITLES.get(match.group(1))


def part_for_chapter(chapter: str | None) -> str | None:
    if not chapter:
        return None
    match = re.search(r"\bChapter\s+(\d+)", chapter, re.IGNORECASE)
    if not match:
        return None
    return PART_TITLES.get(match.group(1))


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".partfix.bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def repair_chunks(chunks_path: Path, dry_run: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    changed = 0
    by_part: collections.Counter[str] = collections.Counter()
    unresolved: list[str] = []

    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            desired = part_for_section(row.get("section_id")) or part_for_chapter(row.get("chapter"))
            if desired is None:
                unresolved.append(row.get("chunk_id", "<unknown>"))
            else:
                by_part[desired] += 1
                if row.get("part") != desired:
                    row["part"] = desired
                    changed += 1
            rows.append(row)

    if changed and not dry_run:
        backup_once(chunks_path)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"rows": len(rows), "changed": changed, "by_part": dict(by_part), "unresolved": unresolved}


def _has_contains_edge(graph: Any, source: str, target: str) -> bool:
    edge_data = graph.get_edge_data(source, target, default={})
    return any(data.get("label") == "contains" for data in edge_data.values())


def repair_graph(graph_path: Path, dry_run: bool) -> dict[str, Any]:
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)

    changed_attrs = 0
    removed_edges = 0
    added_edges = 0

    chapter_parts: dict[str, str] = {}

    for node, data in graph.nodes(data=True):
        if data.get("kind") == "Chapter":
            desired = part_for_chapter(data.get("title") or node)
            if desired:
                chapter_parts[node] = desired
                if data.get("part") != desired:
                    data["part"] = desired
                    changed_attrs += 1
        elif data.get("kind") == "Section":
            desired = part_for_section(data.get("id")) or part_for_chapter(data.get("chapter"))
            if desired and data.get("part") != desired:
                data["part"] = desired
                changed_attrs += 1

    for part_title in PART_TITLES.values():
        part_node = f"part:{part_title}"
        if not graph.has_node(part_node):
            graph.add_node(part_node, kind="Part", title=part_title)
            changed_attrs += 1

    to_remove: list[tuple[str, str, int]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        if data.get("label") != "contains":
            continue
        source_kind = graph.nodes[source].get("kind") if graph.has_node(source) else None
        target_kind = graph.nodes[target].get("kind") if graph.has_node(target) else None
        if source_kind == "Part" and target_kind == "Chapter":
            desired_part = chapter_parts.get(target)
            if desired_part and source != f"part:{desired_part}":
                to_remove.append((source, target, key))
    for source, target, key in to_remove:
        graph.remove_edge(source, target, key)
        removed_edges += 1

    for chapter_node, part_title in chapter_parts.items():
        part_node = f"part:{part_title}"
        if not _has_contains_edge(graph, part_node, chapter_node):
            graph.add_edge(part_node, chapter_node, label="contains")
            added_edges += 1

    for node, data in list(graph.nodes(data=True)):
        if data.get("kind") != "Part":
            continue
        if data.get("title") in set(PART_TITLES.values()):
            continue
        if graph.degree(node) == 0:
            graph.remove_node(node)
            changed_attrs += 1

    if (changed_attrs or removed_edges or added_edges) and not dry_run:
        backup_once(graph_path)
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "changed_attrs": changed_attrs,
        "removed_part_chapter_edges": removed_edges,
        "added_part_chapter_edges": added_edges,
    }


def repair_qdrant(qdrant_dir: Path, dry_run: bool) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_dir))
    try:
        changed_by_part: dict[str, list[int | str]] = collections.defaultdict(list)
        scanned = 0
        offset = None

        while True:
            points, offset = client.scroll(
                collection_name="mutcd_chunks",
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            scanned += len(points)
            for point in points:
                payload = point.payload or {}
                desired = part_for_section(payload.get("section_id")) or part_for_chapter(payload.get("chapter"))
                if desired and payload.get("part") != desired:
                    changed_by_part[desired].append(point.id)
            if offset is None:
                break

        if not dry_run:
            for part_title, point_ids in changed_by_part.items():
                for i in range(0, len(point_ids), 256):
                    client.set_payload(
                        collection_name="mutcd_chunks",
                        payload={"part": part_title},
                        points=point_ids[i : i + 256],
                        wait=True,
                    )

        return {
            "scanned": scanned,
            "changed": sum(len(ids) for ids in changed_by_part.values()),
            "changed_by_part": {part: len(ids) for part, ids in changed_by_part.items()},
        }
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrag-dir",
        type=Path,
        default=Path("data/extracted/MRAG-20260708T114057Z-3/MRAG"),
        help="Extracted MRAG data directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--skip-qdrant", action="store_true", help="Do not update embedded Qdrant payloads.")
    args = parser.parse_args()

    mrag_dir = args.mrag_dir
    chunks_path = mrag_dir / "mmrag_cache_v3" / "chunks.jsonl"
    graph_path = mrag_dir / "mmrag_cache_v3" / "graph.gpickle"
    qdrant_dir = mrag_dir / "qdrant_db"

    print("chunks", json.dumps(repair_chunks(chunks_path, args.dry_run), sort_keys=True))
    print("graph", json.dumps(repair_graph(graph_path, args.dry_run), sort_keys=True))
    if not args.skip_qdrant:
        print("qdrant", json.dumps(repair_qdrant(qdrant_dir, args.dry_run), sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
