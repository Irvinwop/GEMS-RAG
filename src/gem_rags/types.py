from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ContextMode = Literal["injected", "tool_explore"]


@dataclass(frozen=True)
class QAItem:
    qa_id: str
    question: str
    question_type: str | None
    expected_refusal: bool
    gold_answer: dict[str, Any]
    references: list[dict[str, Any]]
    gold_figures: list[Any] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: Literal["chunk", "figure", "page", "tool_trace"]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    adapter: str
    query: str
    evidence: list[Evidence]
    debug: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ModelResult:
    provider: str
    model: str
    output: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class GradingResult:
    grader: str
    scores: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    confidence: float | None = None
    explanation: str | None = None
    figure_metrics: dict[str, Any] = field(default_factory=dict)
    system_confidence_breakdown: dict[str, Any] = field(default_factory=dict)
