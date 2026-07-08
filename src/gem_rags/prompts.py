from __future__ import annotations

import json
import re

from .types import Evidence, QAItem


SYSTEM_PROMPT = """You answer questions about the MUTCD using only supplied evidence.
Classify provisions by MUTCD rule type: Standards are mandatory, Guidance is recommended, Options are permitted, Support is explanatory.
If the question asks for jurisdiction-specific or out-of-scope information not supported by the evidence, say that the MUTCD evidence does not answer it.
Include concise citations using section IDs, content type, paragraph ordinal, and page when available."""


def build_injected_prompt(item: QAItem, evidence: list[Evidence], max_evidence_chars: int) -> str:
    blocks = []
    used = 0
    for idx, ev in enumerate(evidence, 1):
        text = ev.text
        if used + len(text) > max_evidence_chars and blocks:
            break
        used += len(text)
        blocks.append(f"[Evidence {idx}: {ev.kind} score={ev.score:.3f} id={ev.evidence_id}]\n{text}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no retrieved evidence)"
    return f"""{SYSTEM_PROMPT}

Question:
{item.question}

Retrieved evidence:
{evidence_text}

Answer format:
Direct Answer:
Standards:
Guidance:
Options:
Support:
Citations:
"""


def build_tool_explore_prompt(item: QAItem, evidence: list[Evidence], max_evidence_chars: int) -> str:
    return build_tool_answer_prompt(item, evidence, max_evidence_chars)


def build_tool_selection_prompt(item: QAItem, evidence: list[Evidence], max_evidence_chars: int) -> str:
    catalog = []
    used = 0
    for idx, ev in enumerate(evidence, 1):
        meta = ev.metadata
        summary = (
            f"{idx}. {ev.evidence_id} score={ev.score:.3f} "
            f"section={meta.get('section_id')} type={meta.get('content_type')} "
            f"page={meta.get('page_printed')} title={meta.get('section_title')}"
        )
        used += len(summary)
        if used > max_evidence_chars and catalog:
            break
        catalog.append(summary)
    return f"""{SYSTEM_PROMPT}

Context mode: tool_explore.
You can inspect a retrieval tool instead of receiving full context automatically.
Available tools:
- search_mutcd(query): returns candidate hit IDs.
- open_hit(hit_id): returns the MUTCD passage, figure, or external tool trace for that hit.

The harness has already run search_mutcd for the user question and exposed the candidate hit catalog below.
Choose which hit IDs to open before answering. Return only JSON:
{{"open_hit_ids": ["hit-id-1", "hit-id-2"], "reason": "short reason"}}
Open at most 5 hit IDs. Do not invent hit IDs that are not in the catalog.

Question:
{item.question}

Available hit catalog:
{chr(10).join(catalog) if catalog else "(no hits)"}
"""


def build_tool_answer_prompt(item: QAItem, opened_evidence: list[Evidence], max_evidence_chars: int) -> str:
    blocks = []
    used = 0
    for idx, ev in enumerate(opened_evidence, 1):
        text = ev.text
        if used + len(text) > max_evidence_chars and blocks:
            break
        used += len(text)
        blocks.append(f"[Opened Evidence {idx}: {ev.kind} score={ev.score:.3f} id={ev.evidence_id}]\n{text}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no opened evidence)"
    return f"""{SYSTEM_PROMPT}

Context mode: tool_explore.
You requested tool results from open_hit(hit_id). Answer only from the opened evidence below.
If the opened evidence is insufficient, say that the opened MUTCD evidence does not answer the question.

Question:
{item.question}

Opened tool results:
{evidence_text}

Answer format:
Tool Plan:
Inspected Evidence:
Direct Answer:
Standards:
Guidance:
Options:
Support:
Citations:
"""


def parse_open_hit_ids(text: str) -> list[str]:
    parsed = _parse_json_object(text)
    if not isinstance(parsed, dict):
        return []
    raw_ids = parsed.get("open_hit_ids") or parsed.get("hit_ids") or parsed.get("ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        return []
    ids: list[str] = []
    for raw_id in raw_ids:
        hit_id = str(raw_id).strip()
        if hit_id and hit_id not in ids:
            ids.append(hit_id)
    return ids


def _parse_json_object(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for candidate in fenced:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
