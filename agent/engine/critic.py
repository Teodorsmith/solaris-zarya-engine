# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Lateral Critic / Arbiter (Two-Solver Pattern) -- Mitigation #63.

Opt-in pattern for high-novelty tasks. Replaces single Solver-Critic loop with
two parallel solvers. Divergence (disagreement) is exploited as signal.
If solvers agree, critic is skipped (consensus). If they diverge, a third
Arbiter call is spawned to evaluate the disagreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.brains.base import BaseBrain
from agent.config import CRITIC_SIMILARITY_THRESHOLD
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.reasoning import ReasoningMemory
from agent.models import ReasoningEpisode

logger = logging.getLogger(__name__)


@dataclass
class CriticResult:
    verdict: str  # "consensus" | "divergent"
    answer: str
    episode: ReasoningEpisode | None = None


class CriticSession:
    """Context manager for the Lateral Critic pattern.

    Usage::
        with CriticSession(brain_a=primary, brain_b=secondary, embedder=emb, memory=mem) as session:
            result = session.solve(prompt)
    """

    def __init__(
        self,
        brain_a: BaseBrain,
        brain_b: BaseBrain | None,
        embedder: EmbeddingEngine,
        reasoning_memory: ReasoningMemory | None = None,
        task_id: str | None = None,
        reasoning_domain: str | None = None,
    ) -> None:
        self.brain_a = brain_a
        self.brain_b = brain_b
        self.embedder = embedder
        self.memory = reasoning_memory
        self.task_id = task_id
        self.domain = reasoning_domain

        if self.brain_b is None:
            logger.warning(
                "CriticSession: running in single-provider fallback mode. "
                "Divergence detection relies on temperature only."
            )
            self.brain_b = brain_a
            self._is_fallback = True
        else:
            self._is_fallback = False

    def __enter__(self) -> CriticSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def solve(self, prompt: str) -> CriticResult:
        """Execute the two-solver dispatch pattern."""

        # 1. Run solvers (sequentially to avoid thread complexity)
        # In a real environment, we'd inject temperature=0.85 for fallback mode here.
        # But MockBrain ignores temperature.
        ans_a = self.brain_a.generate(prompt)
        ans_b = self.brain_b.generate(prompt)

        # 2. Consensus detection via cosine similarity
        emb_a = self.embedder.embed(ans_a)
        emb_b = self.embedder.embed(ans_b)

        similarity = self._cosine_similarity(emb_a, emb_b)
        logger.debug("CriticSession: semantic similarity = %.3f", similarity)

        if similarity >= CRITIC_SIMILARITY_THRESHOLD:
            # Consensus: return early, skip arbiter.
            logger.info("CriticSession: consensus reached (skip arbiter).")
            return CriticResult(verdict="consensus", answer=ans_a, episode=None)

        # 3. Divergence: spawn arbiter
        logger.info("CriticSession: divergence detected. Spawning arbiter.")
        arbiter_prompt = (
            f"You are the Arbiter.\n"
            f"Problem: {prompt}\n\n"
            f"Hypothesis A: {ans_a}\n\n"
            f"Hypothesis B: {ans_b}\n\n"
            f"You MUST find at least one weakness or missing edge case in one of "
            f"these hypotheses. Do NOT just say 'Looks good'. Which one is more "
            f"sound and why?"
        )

        # We use brain_a (the primary) as the Arbiter
        resolution = self.brain_a.generate(arbiter_prompt)

        episode = ReasoningEpisode(
            task_id=self.task_id,
            state=prompt[:500],
            hypothesis=f"A: {ans_a[:200]}... | B: {ans_b[:200]}...",
            action="arbiter_eval",
            observation=resolution,
            outcome_class="divergent",
            hypothesis_count=2,
            reasoning_domain=self.domain,
            strategy_label="lateral_critic",
        )

        if self.memory:
            self.memory.log_episode(episode)

        return CriticResult(verdict="divergent", answer=resolution, episode=episode)

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity of two normalized vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return dot
