# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for ModelTrainer and ZPD Promotion Gate -- Mitigation #69."""

from unittest.mock import Mock, patch

from agent.brains.mock_brain import MockBrain
from agent.engine.trainer import ModelTrainer


def test_promotion_gate_passed():
    trainer = ModelTrainer()
    candidate_brain = MockBrain()
    baseline_brain = MockBrain()

    # Candidate improves on 4/6 categories, maintains 2, regresses on 0
    candidate_ceilings = {
        "decomposition": 4,
        "hypothesis_testing": 3,
        "causal_reasoning": 5,
        "counterexample_gen": 4,
        "planning": 3,
        "adversarial": 3,
    }
    baseline_ceilings = {
        "decomposition": 2,
        "hypothesis_testing": 2,
        "causal_reasoning": 3,
        "counterexample_gen": 2,
        "planning": 3,
        "adversarial": 3,
    }

    with patch("agent.engine.trainer.ZPDRunner") as MockRunner:
        runner_cand = Mock(run_all=Mock(return_value=candidate_ceilings))
        runner_base = Mock(run_all=Mock(return_value=baseline_ceilings))
        MockRunner.side_effect = [runner_cand, runner_base]

        promoted, report = trainer.evaluate_candidate(
            candidate_brain, baseline_brain
        )

        assert promoted is True
        assert report["improved_count"] == 4
        assert report["regressed_count"] == 0
        assert report["promotion_gate_passed"] is True


def test_promotion_gate_failed_insufficient_improvements():
    trainer = ModelTrainer()
    candidate_brain = MockBrain()
    baseline_brain = MockBrain()

    # Candidate improves on only 2/6 categories (< 3 required)
    candidate_ceilings = {
        "decomposition": 3,
        "hypothesis_testing": 3,
        "causal_reasoning": 3,
        "counterexample_gen": 3,
        "planning": 3,
        "adversarial": 3,
    }
    baseline_ceilings = {
        "decomposition": 2,
        "hypothesis_testing": 2,
        "causal_reasoning": 3,
        "counterexample_gen": 3,
        "planning": 3,
        "adversarial": 3,
    }

    with patch("agent.engine.trainer.ZPDRunner") as MockRunner:
        runner_cand = Mock(run_all=Mock(return_value=candidate_ceilings))
        runner_base = Mock(run_all=Mock(return_value=baseline_ceilings))
        MockRunner.side_effect = [runner_cand, runner_base]

        promoted, report = trainer.evaluate_candidate(
            candidate_brain, baseline_brain
        )

        assert promoted is False
        assert report["improved_count"] == 2
        assert report["regressed_count"] == 0


def test_promotion_gate_failed_with_regression():
    trainer = ModelTrainer()
    candidate_brain = MockBrain()
    baseline_brain = MockBrain()

    # Candidate improves on 3 categories, but regresses on 1 category
    candidate_ceilings = {
        "decomposition": 4,
        "hypothesis_testing": 4,
        "causal_reasoning": 4,
        "counterexample_gen": 1,  # Regression (baseline is 2)
        "planning": 3,
        "adversarial": 3,
    }
    baseline_ceilings = {
        "decomposition": 2,
        "hypothesis_testing": 2,
        "causal_reasoning": 2,
        "counterexample_gen": 2,
        "planning": 3,
        "adversarial": 3,
    }

    with patch("agent.engine.trainer.ZPDRunner") as MockRunner:
        runner_cand = Mock(run_all=Mock(return_value=candidate_ceilings))
        runner_base = Mock(run_all=Mock(return_value=baseline_ceilings))
        MockRunner.side_effect = [runner_cand, runner_base]

        promoted, report = trainer.evaluate_candidate(
            candidate_brain, baseline_brain
        )

        assert promoted is False
        assert report["improved_count"] == 3
        assert report["regressed_count"] == 1
        assert "counterexample_gen" in report["regressed_categories"]


def test_train_and_promote_lifecycle():
    self_model = Mock()
    trainer = ModelTrainer(self_model=self_model)
    candidate_brain = MockBrain()
    baseline_brain = MockBrain()

    categories = [
        "decomposition",
        "hypothesis_testing",
        "causal_reasoning",
        "counterexample_gen",
        "planning",
        "adversarial",
    ]
    candidate_ceilings = {cat: 4 for cat in categories}
    baseline_ceilings = {cat: 2 for cat in categories}

    with patch("agent.engine.trainer.ZPDRunner") as MockRunner:
        runner_cand = Mock(run_all=Mock(return_value=candidate_ceilings))
        runner_base = Mock(run_all=Mock(return_value=baseline_ceilings))
        MockRunner.side_effect = [runner_cand, runner_base]

        result = trainer.train_and_promote(
            candidate_brain, baseline_brain, checkpoint_name="qwen_lora_v1"
        )

        assert result["status"] == "promoted"
        assert result["active_model"] == "qwen_lora_v1"
        self_model.update_zpd_ceilings.assert_called_once_with(
            candidate_ceilings
        )
