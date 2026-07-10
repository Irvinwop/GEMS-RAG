from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gem_rags.config import GraderConfig
from gem_rags.grading import RUBRIC_KEYS, build_llm_grader_prompt, grade_answer, llm_grade, normalize_judge_scores, parse_grader_output
from gem_rags.types import Evidence, ModelResult, QAItem, RetrievalResult


def _qa() -> QAItem:
    return QAItem(
        qa_id="qa_test",
        question="May an agency substitute an alternative legend for a standard sign?",
        question_type="standards",
        expected_refusal=False,
        gold_answer={"answer": "No, except as provided in the Manual."},
        references=[{"section_id": "2A.04", "content_type": "Standard", "ordinal": 13}],
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        adapter="unit",
        query="question",
        evidence=[
            Evidence(
                evidence_id="MUTCD11e_2A04_Standard_13",
                kind="chunk",
                text="Where a standard sign is applicable, an alternative legend shall not be allowed." + (" x" * 800),
                metadata={"section_id": "2A.04", "content_type": "Standard", "ordinal": 13},
                score=7.5,
            )
        ],
    )


class TestGrading(unittest.TestCase):
    def test_prompt_includes_retrieved_evidence_and_truncates_text(self) -> None:
        prompt = build_llm_grader_prompt(
            GraderConfig(provider="openai", model="judge", options={"max_evidence_text_chars": 120}),
            _qa(),
            ModelResult(provider="dry_run", model="dry-run", output="No."),
            _retrieval(),
        )
        self.assertIn("retrieved_evidence", prompt)
        self.assertIn("MUTCD11e_2A04_Standard_13", prompt)
        self.assertIn("...", prompt)
        self.assertLess(len(prompt), 5000)

    def test_parse_fenced_json_and_normalize_scores(self) -> None:
        parsed, error = parse_grader_output(
            """Some preface.
```json
{"judge_scores": {"factual_accuracy": {"score": 6, "note": "too high"}, "completeness": "4"}, "judge_confidence": "0.75"}
```
"""
        )
        self.assertIsNone(error)
        scores = normalize_judge_scores(parsed)
        self.assertEqual(set(scores), set(RUBRIC_KEYS))
        self.assertEqual(scores["factual_accuracy"]["score"], 5)
        self.assertEqual(scores["completeness"]["score"], 4)
        self.assertIsNone(scores["figure_grounding"]["score"])

    def test_parse_failure_is_reported(self) -> None:
        parsed, error = parse_grader_output("not json")
        self.assertEqual(error, "grader_json_parse_failed")
        self.assertEqual(parsed["raw_text"], "not json")

    def test_llm_grade_uses_evidence_and_returns_complete_rubric(self) -> None:
        class FakeModel:
            prompt = ""

            def generate(self, prompt: str) -> ModelResult:
                FakeModel.prompt = prompt
                return ModelResult(
                    provider="fake",
                    model="judge",
                    output='{"judge_scores": {"factual_accuracy": {"score": 5, "note": "ok"}}, "judge_confidence": 1.2, "judge_explanation": "grounded"}',
                    raw={"fake": True},
                )

        with patch("gem_rags.grading.build_model", return_value=FakeModel()):
            result = llm_grade(
                GraderConfig(provider="openai", model="judge"),
                _qa(),
                ModelResult(provider="answer", model="m", output="No, except as provided in the Manual."),
                _retrieval(),
            )

        self.assertIn("retrieved_evidence", FakeModel.prompt)
        self.assertIn("MUTCD11e_2A04_Standard_13", FakeModel.prompt)
        self.assertIsNone(result.error)
        self.assertEqual(set(result.scores), set(RUBRIC_KEYS))
        self.assertEqual(result.scores["factual_accuracy"]["score"], 5)
        self.assertEqual(result.confidence, 1.0)

    def test_llm_grade_can_reuse_prebuilt_model_client(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, _prompt: str) -> ModelResult:
                self.calls += 1
                return ModelResult(
                    provider="fake",
                    model="judge",
                    output='{"judge_scores": {"factual_accuracy": 4}, "judge_confidence": 0.8}',
                    raw={"fake": True},
                )

        fake_model = FakeModel()
        with patch("gem_rags.grading.build_model", side_effect=AssertionError("grader client should be reused")):
            result = llm_grade(
                GraderConfig(provider="openai", model="judge"),
                _qa(),
                ModelResult(provider="answer", model="m", output="No."),
                _retrieval(),
                model_client=fake_model,
            )

        self.assertEqual(fake_model.calls, 1)
        self.assertEqual(result.scores["factual_accuracy"]["score"], 4)
        self.assertIsNone(result.error)

    def test_llm_grade_sends_retrieved_images_to_visual_grader(self) -> None:
        class FakeVisionGrader:
            def __init__(self) -> None:
                self.image_paths: list[str] = []

            def generate_with_images(self, _prompt: str, image_paths) -> ModelResult:
                self.image_paths = [str(path) for path in image_paths]
                return ModelResult(
                    provider="fake",
                    model="judge",
                    output='{"judge_scores": {"figure_grounding": 5}, "judge_confidence": 0.9}',
                )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "figure.png"
            image_path.write_bytes(b"fixture")
            retrieval = RetrievalResult(
                adapter="unit",
                query="question",
                evidence=[Evidence("figure-1", "figure", "A figure", {"image_path": str(image_path)}, 1.0)],
            )
            grader = FakeVisionGrader()
            result = llm_grade(
                GraderConfig(provider="openai", model="judge", options={"vision": True}),
                _qa(),
                ModelResult(provider="answer", model="m", output="The figure shows a sign."),
                retrieval,
                model_client=grader,
            )

        self.assertEqual(grader.image_paths, [str(image_path)])
        self.assertEqual(result.scores["figure_grounding"]["score"], 5)

    def test_grader_accepts_model_provider_aliases(self) -> None:
        seen_configs = []

        class FakeModel:
            def generate(self, _prompt: str) -> ModelResult:
                return ModelResult(
                    provider="fake",
                    model="judge",
                    output='{"judge_scores": {"factual_accuracy": 4}, "judge_confidence": 0.9}',
                )

        def fake_build_model(config):
            seen_configs.append(config)
            return FakeModel()

        with patch("gem_rags.grading.build_model", side_effect=fake_build_model):
            result = grade_answer(
                GraderConfig(provider="qwen", model="qwen-judge"),
                _qa(),
                ModelResult(provider="answer", model="m", output="No."),
                _retrieval(),
            )

        self.assertEqual(seen_configs[0].provider, "qwen")
        self.assertEqual(seen_configs[0].model, "qwen-judge")
        self.assertEqual(result.scores["factual_accuracy"]["score"], 4)

    def test_dry_run_is_not_a_valid_grader_provider(self) -> None:
        with self.assertRaises(ValueError):
            grade_answer(
                GraderConfig(provider="dry_run", model="dry-run"),
                _qa(),
                ModelResult(provider="answer", model="m", output="No."),
                _retrieval(),
            )


if __name__ == "__main__":
    unittest.main()
