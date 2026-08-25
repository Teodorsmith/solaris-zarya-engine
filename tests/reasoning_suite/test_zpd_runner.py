# Copyright (C) 2026 Teodor Smith
import unittest

from agent.brains.mock_brain import MockBrain
from tests.reasoning_suite.runner import ZPDRunner


class MockBrainManager:
    def __init__(self, brain):
        self.brain = brain

    def fallback(self):
        return None


class TestZPDRunner(unittest.TestCase):
    def setUp(self):
        self.brain = MockBrain()
        self.brain_manager = MockBrainManager(self.brain)
        self.runner = ZPDRunner(self.brain_manager)

        # Override fixtures for testing without relying on disk
        self.runner.fixtures = {
            "test_cat": {
                1: {
                    "prompt": "1",
                    "answer_key": "ans1",
                    "expected_inference_rule": "modus_ponens",
                    "srt_required": False,
                },
                2: {
                    "prompt": "2",
                    "answer_key": "ans2",
                    "expected_inference_rule": "modus_ponens",
                    "srt_required": False,
                },
                3: {
                    "prompt": "3",
                    "answer_key": "ans3",
                    "expected_inference_rule": "modus_ponens",
                    "srt_required": False,
                },
                4: {
                    "prompt": "4",
                    "answer_key": "ans4",
                    "expected_inference_rule": "modus_ponens",
                    "srt_required": False,
                },
                5: {
                    "prompt": "5",
                    "answer_key": "ans5",
                    "expected_inference_rule": "modus_ponens",
                    "srt_required": False,
                },
            }
        }

    def test_run_problem_success(self):
        # Mockbrain returns 'ans1' somewhere
        self.brain.generate = lambda x: (
            '{"conclusion": "The answer is ans1", "premises": [], "inference_rule": "modus_ponens", "confidence": 1.0}'
        )
        passed = self.runner._run_problem(self.runner.fixtures["test_cat"][1])
        self.assertTrue(passed)

    def test_run_problem_fail(self):
        # Wrong answer
        self.brain.generate = lambda x: (
            '{"conclusion": "The answer is ans99", "premises": [], "inference_rule": "modus_ponens", "confidence": 1.0}'
        )
        passed = self.runner._run_problem(self.runner.fixtures["test_cat"][1])
        self.assertFalse(passed)

    def test_search_category(self):
        # Let's say brain can pass 1 and 2, but fails 3, 4, 5
        def mock_generate(prompt):
            if (
                "prompt: 1" in prompt.lower()
                or "prompt: 2" in prompt.lower()
                or "1" in prompt
                or "2" in prompt
            ):
                ans = (
                    prompt[-1] if prompt[-1].isdigit() else "2"
                )  # super hacky mock for this specific prompt
                ans = next(c for c in prompt if c.isdigit())
                return f'{{"conclusion": "ans{ans}", "premises": [], "inference_rule": "modus_ponens", "confidence": 1.0}}'
            return '{"conclusion": "wrong", "premises": [], "inference_rule": "modus_ponens", "confidence": 1.0}'

        self.brain.generate = mock_generate

        ceiling = self.runner._search_category("test_cat")
        # should pass 1,2, fail 3,4,5 -> ceiling is 2
        self.assertEqual(ceiling, 2)


if __name__ == "__main__":
    unittest.main()
