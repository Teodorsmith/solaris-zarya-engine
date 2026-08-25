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

"""Unit tests for Mixture-of-Agents (MoA) Brain Router -- Mitigation #70."""

from unittest.mock import Mock

from agent.brains.base import QuotaExceededError
from agent.brains.factory import BRAIN_BUILDERS
from agent.brains.mock_brain import MockBrain
from agent.brains.moa_router import MoABrain


def test_complexity_estimation():
    base_brain = MockBrain()
    lora_brain = MockBrain()
    router = MoABrain(base_brain, lora_brain, complexity_threshold=0.5)

    # 1. Simple routine task -> low complexity (< 0.5)
    routine_prompt = "What is the capital of France?"
    assert router.estimate_complexity(routine_prompt) < 0.5

    # 2. Multi-step formal logic & proof -> high complexity (>= 0.5)
    complex_prompt = (
        "Given axioms A → B and B → C, prove by modus_ponens that A → C. "
        "Provide a formal SRT truth table and step-by-step invariant."
    )
    assert router.estimate_complexity(complex_prompt) >= 0.5

    # 3. Context override
    override = router.estimate_complexity(
        "simple", context={"complexity_score": 0.85}
    )
    assert override == 0.85


def test_moa_routing_to_base_and_lora():
    base_brain = Mock()
    base_brain.generate.return_value = "base_response"
    lora_brain = Mock()
    lora_brain.generate.return_value = "lora_response"

    router = MoABrain(base_brain, lora_brain, complexity_threshold=0.5)

    # Routine prompt routes to base
    res_routine = router.generate("What is 2 + 2?")
    assert res_routine == "base_response"
    assert base_brain.generate.called
    assert not lora_brain.generate.called

    base_brain.reset_mock()
    lora_brain.reset_mock()

    # Complex proof routes to lora
    complex_prompt = (
        "Formulate a formal proof for theorem 1 using modus_tollens:\n"
        "1. P → Q\n"
        "2. ¬Q\n"
        "3. Therefore ¬P."
    )
    res_complex = router.generate(complex_prompt)
    assert res_complex == "lora_response"
    assert lora_brain.generate.called
    assert not base_brain.generate.called


def test_moa_explicit_complexity_kwargs():
    base_brain = Mock(generate=Mock(return_value="base"))
    lora_brain = Mock(generate=Mock(return_value="lora"))

    router = MoABrain(base_brain, lora_brain, complexity_threshold=0.5)

    # Forced high complexity
    res = router.generate("simple prompt", complexity_score=0.9)
    assert res == "lora"
    assert lora_brain.generate.called

    base_brain.reset_mock()
    lora_brain.reset_mock()

    # Forced low complexity
    res = router.generate(
        "complex deduction proof theorem", complexity_score=0.1
    )
    assert res == "base"
    assert base_brain.generate.called


def test_moa_fallback_on_lora_error():
    base_brain = Mock(generate=Mock(return_value="fallback_base_response"))
    lora_brain = Mock()
    lora_brain.generate.side_effect = QuotaExceededError("Rate limit exceeded")

    router = MoABrain(base_brain, lora_brain, complexity_threshold=0.5)

    # High complexity prompt routes to lora, but fails and falls back to base
    res = router.generate(
        "prove theorem using modus_ponens", complexity_score=0.8
    )
    assert res == "fallback_base_response"
    assert lora_brain.generate.called
    assert base_brain.generate.called
    assert router.fallbacks == 1


def test_moa_embed_and_stats():
    from agent.memory.embeddings import EmbeddingEngine

    embedder = EmbeddingEngine(force_fallback=True)
    base_brain = MockBrain(embedder=embedder)
    lora_brain = MockBrain(embedder=embedder)
    router = MoABrain(base_brain, lora_brain, complexity_threshold=0.5)

    vec = router.embed("test embedding")
    assert len(vec) == 384

    router.generate("What is 1+1?")
    stats = router.get_stats()
    assert stats["total_routed"] == 1
    assert stats["routed_to_base"] == 1
    assert stats["routed_to_lora"] == 0


def test_moa_factory_registration(monkeypatch):
    monkeypatch.setenv("AI_BRAIN", "moa_router")
    assert "moa_router" in BRAIN_BUILDERS
    builder = BRAIN_BUILDERS["moa_router"]
    brain = builder(embedder=None)
    assert isinstance(brain, MoABrain)
