from __future__ import annotations

import unittest

from gems_rag.runtime_validity import operational_row_problems, provider_completion_problems


def _valid_row() -> dict:
    answer = "Direct Answer: The MUTCD requires the device.\nCitations: Section 2A.01."
    return {
        "qa_id": "T001",
        "question": "What is required?",
        "run_status": "successful",
        "answer": answer,
        "serialized_return": {"answer": answer},
        "model_raw": {"finish_reason": "stop", "status": "completed"},
        "retrieval_error": None,
        "model_error": None,
        "judge_error": None,
        "evidence": [
            {
                "evidence_id": "chunk-1",
                "kind": "chunk",
                "text": "The device shall be used.",
                "score": 1.0,
                "metadata": {"section_id": "2A.01"},
            }
        ],
        "retrieval_debug": {
            "retrieved_evidence_count": 1,
            "provided_evidence_count": 1,
        },
    }


class TestRuntimeValidity(unittest.TestCase):
    def test_complete_row_is_operationally_valid(self) -> None:
        self.assertEqual(operational_row_problems(_valid_row()), [])

    def test_empty_serialized_answer_and_heading_only_output_are_invalid(self) -> None:
        row = _valid_row()
        row["answer"] = "# Direct Answer:"
        row["serialized_return"] = {"answer": ""}

        problems = operational_row_problems(row)

        self.assertIn("saved answer is only a heading or evidence caption", problems)
        self.assertIn("serialized_return.answer is empty", problems)

    def test_nested_provider_truncation_is_invalid(self) -> None:
        row = _valid_row()
        row["model_raw"] = {
            "model_calls": [
                {"finish_reason": "stop"},
                {"status": "incomplete", "incomplete_details": {"stop_reason": "max_tokens"}},
            ]
        }

        problems = operational_row_problems(row)

        self.assertTrue(any("provider completion is invalid" in problem for problem in problems))
        self.assertEqual(
            provider_completion_problems(row["model_raw"]),
            ["status=incomplete", "stop_reason=max_tokens"],
        )

    def test_incomplete_retrieval_log_is_invalid(self) -> None:
        row = _valid_row()
        row["retrieval_debug"]["provided_evidence_count"] = 2

        problems = operational_row_problems(row)

        self.assertTrue(any("does not match evidence rows" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
