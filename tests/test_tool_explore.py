from __future__ import annotations

import unittest

from gem_rags.prompts import parse_open_hit_ids, parse_search_queries
from gem_rags.runner import _generate_tool_explore, _generate_tool_search
from gem_rags.types import Evidence, ModelResult, QAItem, RetrievalResult


class FakeExploreModel:
    def __init__(self, selection_output: str) -> None:
        self.selection_output = selection_output
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ModelResult:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return ModelResult(
                provider="fake",
                model="explorer",
                output=self.selection_output,
                raw={"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
            )
        return ModelResult(
            provider="fake",
            model="explorer",
            output="Direct Answer: opened answer",
            raw={"answer": True, "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24}},
        )


class FakeToolSearchModel:
    def __init__(self, search_output: str | None = None, selection_output: str | None = None) -> None:
        self.search_output = search_output or '{"search_queries": [{"query": "Section 2A.04 warning signs", "top_k": 2}]}'
        self.selection_output = selection_output or '{"open_hit_ids": ["hit-b"]}'
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ModelResult:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return ModelResult(
                provider="fake",
                model="tool-searcher",
                output=self.search_output,
                raw={"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
            )
        if len(self.prompts) == 2:
            return ModelResult(
                provider="fake",
                model="tool-searcher",
                output=self.selection_output,
                raw={"usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23}},
            )
        return ModelResult(
            provider="fake",
            model="tool-searcher",
            output="Direct Answer: searched answer",
            raw={"answer": True, "usage": {"input_tokens": 30, "output_tokens": 4, "total_tokens": 34}},
        )


class FakeSearchRetriever:
    name = "searchable"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, item: QAItem) -> RetrievalResult:
        self.queries.append(item.question)
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=[
                Evidence("hit-a", "chunk", "A searched passage", {"section_id": "2A.04"}, 2.0),
                Evidence("hit-b", "chunk", "B searched passage", {"section_id": "2A.04"}, 1.0),
            ],
        )


class TopKSearchRetriever:
    name = "top-k-searchable"

    def __init__(self) -> None:
        self.top_k = 1
        self.seen_top_k: list[int] = []

    def retrieve(self, item: QAItem) -> RetrievalResult:
        self.seen_top_k.append(self.top_k)
        evidence = [
            Evidence("hit-a", "chunk", "A searched passage", {"section_id": "2A.04"}, 4.0),
            Evidence("hit-b", "chunk", "B searched passage", {"section_id": "2A.04"}, 3.0),
            Evidence("hit-c", "chunk", "C searched passage", {"section_id": "2A.04"}, 2.0),
            Evidence("hit-d", "chunk", "D searched passage", {"section_id": "2A.04"}, 1.0),
        ]
        return RetrievalResult(adapter=self.name, query=item.question, evidence=evidence[: self.top_k])


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

    def test_parse_search_queries_accepts_strings_and_objects(self) -> None:
        self.assertEqual(
            parse_search_queries('{"search_queries": ["warning signs", {"query": "Section 2A.04", "top_k": 12}, {"query": "warning signs"}]}'),
            [
                {"query": "warning signs", "top_k": 6},
                {"query": "Section 2A.04", "top_k": 12},
            ],
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
        self.assertEqual(result.raw["usage"], {"input_tokens": 30, "output_tokens": 6, "total_tokens": 36})
        self.assertEqual(result.raw["usage_coverage"], {"expected_calls": 2, "observed_calls": 2, "complete": True})
        self.assertEqual(result.raw["model_calls"]["selection"]["usage"]["total_tokens"], 12)

    def test_generate_tool_search_runs_model_chosen_query_then_opens_selected_hit(self) -> None:
        model = FakeToolSearchModel()
        retriever = FakeSearchRetriever()
        initial = RetrievalResult(adapter="searchable", query=_item().question, evidence=[], debug={"deferred_retrieval": True})
        result, context_retrieval, debug = _generate_tool_search(model, _item(), retriever, initial, 2000)

        self.assertEqual(result.output, "Direct Answer: searched answer")
        self.assertEqual(retriever.queries, ["Section 2A.04 warning signs"])
        self.assertEqual([ev.evidence_id for ev in context_retrieval.evidence], ["hit-b"])
        self.assertEqual(context_retrieval.evidence[0].metadata["tool_search_query"], "Section 2A.04 warning signs")
        self.assertEqual(debug["search_queries"], [{"query": "Section 2A.04 warning signs", "top_k": 2}])
        self.assertEqual(debug["opened_ids"], ["hit-b"])
        self.assertIn("choose search queries", model.prompts[0])
        self.assertNotIn("A searched passage", model.prompts[0])
        self.assertIn("Search result catalog", model.prompts[1])
        self.assertNotIn("A searched passage", model.prompts[1])
        self.assertIn("Opened tool results", model.prompts[2])
        self.assertIn("B searched passage", model.prompts[2])
        self.assertNotIn("A searched passage", model.prompts[2])
        self.assertEqual(result.raw["usage"], {"input_tokens": 60, "output_tokens": 9, "total_tokens": 69})
        self.assertEqual(result.raw["usage_coverage"], {"expected_calls": 3, "observed_calls": 3, "complete": True})
        self.assertEqual(result.raw["model_calls"]["search_plan"]["usage"]["total_tokens"], 12)

    def test_generate_tool_search_applies_requested_top_k_temporarily(self) -> None:
        model = FakeToolSearchModel(
            search_output='{"search_queries": [{"query": "Section 2A.04 warning signs", "top_k": 4}]}',
            selection_output='{"open_hit_ids": ["hit-d"]}',
        )
        retriever = TopKSearchRetriever()
        initial = RetrievalResult(adapter="top-k-searchable", query=_item().question, evidence=[], debug={"deferred_retrieval": True})
        result, context_retrieval, debug = _generate_tool_search(model, _item(), retriever, initial, 2000)

        self.assertEqual(result.output, "Direct Answer: searched answer")
        self.assertEqual(retriever.seen_top_k, [4])
        self.assertEqual(retriever.top_k, 1)
        self.assertEqual([ev.evidence_id for ev in context_retrieval.evidence], ["hit-d"])
        self.assertEqual(debug["search_results"][0]["result_ids"], ["hit-a", "hit-b", "hit-c", "hit-d"])


if __name__ == "__main__":
    unittest.main()
