#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "indexes" / "corpus" / "chunks.jsonl"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    """The exact dependency-free BM25 baseline used in the comparison study."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("the BM25 corpus is empty")
        self.documents = documents
        self.document_tokens = [
            tokenize(f"{row.get('title', '')}\n{row.get('text', '')}")
            for row in documents
        ]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )
        document_frequency: collections.Counter[str] = collections.Counter()
        for tokens in self.document_tokens:
            document_frequency.update(set(tokens))
        document_count = len(self.document_tokens)
        self.inverse_document_frequency = {
            term: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def query(self, question: str, *, top_k: int) -> list[dict[str, Any]]:
        query_tokens = tokenize(question)
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self.document_tokens):
            score = self._score(
                query_tokens,
                tokens,
                self.document_lengths[index],
            )
            if score > 0:
                scored.append((score, index))

        # Preserve the study implementation's score-descending/index-descending
        # tie break so a rebuilt run matches the frozen baseline.
        scored.sort(reverse=True)
        contexts = []
        for score, index in scored[:top_k]:
            document = self.documents[index]
            document_id = str(document.get("doc_id") or f"document_{index}")
            metadata = dict(document.get("metadata") or {})
            metadata["source_name"] = document_id
            metadata["chunk_id"] = document_id
            contexts.append(
                {
                    "name": document_id,
                    "kind": "chunk",
                    "text": str(document.get("text") or ""),
                    "score": score,
                    "metadata": metadata,
                }
            )
        return contexts

    def _score(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
        document_length: int,
    ) -> float:
        counts = collections.Counter(document_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_tokens:
            term_frequency = counts.get(term, 0)
            if not term_frequency:
                continue
            denominator = term_frequency + k1 * (
                1
                - b
                + b
                * document_length
                / max(self.average_document_length, 1e-9)
            )
            score += self.inverse_document_frequency.get(term, 0.0) * (
                term_frequency * (k1 + 1)
            ) / denominator
        return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the comparison BM25 index.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate and summarize the BM25 corpus.")
    query = subparsers.add_parser("query", help="Return ranked BM25 context.")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if getattr(args, "top_k", 1) < 1:
        parser.error("--top-k must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.corpus.is_file():
        print(
            json.dumps(
                {
                    "runnable": False,
                    "method": "bm25",
                    "corpus": str(args.corpus),
                    "error": "corpus_not_found",
                }
            )
        )
        return 2
    documents = list(read_jsonl(args.corpus))
    index = BM25Index(documents)
    if args.command == "check":
        print(
            json.dumps(
                {
                    "runnable": True,
                    "method": "bm25",
                    "documents": len(documents),
                    "corpus": str(args.corpus),
                },
                indent=2,
            )
        )
        return 0
    contexts = index.query(args.question, top_k=args.top_k)
    print(
        json.dumps(
            {
                "method": "bm25",
                "question": args.question,
                "top_k": args.top_k,
                "contexts": contexts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
