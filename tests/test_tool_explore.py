from __future__ import annotations

import unittest

from gem_rags.prompts import parse_open_hit_ids
from gem_rags.runner import _generate_tool_explore
from gem_rags.types import Evidence, ModelResult, QAItem, RetrievalResult


class FakeExploreModel:
    def __init__(self, selection_output: str) -> None:
        self.selection_output = selection_output
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ModelResult:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return ModelResult(provider="fake", model="explorer", output=self.selection_output)
        return ModelResult(provider="fake", model="explorer", output="Direct Answer: opened answer", raw={"answer": True})


def _item() -> QAItem:
    return QAItem(
        qa_id="qa_tool",
        question="What does Section 2A.04 say?",
        question_type=None,
        expected_refusal=False,
        gold_answer={},
        references=[],
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        adapter="unit",
        query="What does Section 2A.04 say?",
        evidence=[
            Evidence("hit-a", "chunk", "A full passage", {"section_id": "2A.04"}, 2.0),
            Evidence("hit-b", "chunk", "B full passage", {"section_id": "2A.04"}, 1.0),
            Evidence("hit-c", "chunk", "C full passage", {"section_id": "2A.05"}, 0.5),
        ],
    )


class TestToolExplore(unittest.TestCase):
    def test_parse_open_hit_ids_from_fenced_json(self) -> None:
        self.assertEqual(
            parse_open_hit_ids('```json\n{"open_hit_ids": ["hit-a", "hit-b", "hit-a"]}\n```'),
            ["hit-a", "hit-b"],
        )

    def test_generate_tool_explore_opens_only_selected_known_ids(self) -> None:
        model = FakeExploreModel('{"open_hit_ids": ["hit-b", "missing", "hit-a"]}')
        result, context_retrieval, debug = _generate_tool_explore(model, _item(), _retrieval(), 2000)

        self.assertEqual(result.output, "Direct Answer: opened answer")
        self.assertEqual([ev.evidence_id for ev in context_retrieval.evidence], ["hit-b", "hit-a"])
        self.assertEqual(debug["selected_ids"], ["hit-b", "missing", "hit-a"])
        self.assertEqual(debug["opened_ids"], ["hit-b", "hit-a"])
        self.assertIn("Available hit catalog", model.prompts[0])
        self.assertNotIn("A full passage", model.prompts[0])
        self.assertIn("Opened tool results", model.prompts[1])
        self.assertIn("B full passage", model.prompts[1])
        self.assertIn("A full passage", model.prompts[1])
        self.assertNotIn("C full passage", model.prompts[1])


if __name__ == "__main__":
    unittest.main()
