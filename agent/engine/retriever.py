"""
Hybrid retrieval + confidence gating + closed-world answer construction.

No LLM call in Phase 0 — MockBrain can't reason, so "answering" means:
retrieve the best-matching facts, then respond according to how confident
that match actually is. This is the honest core of the whole system: it
must never present a low-confidence guess as settled fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.config import CONFIDENT_THRESHOLD, TENTATIVE_THRESHOLD
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.models import EpisodicLog, Fact


@dataclass
class RetrievalResult:
    facts: list[Fact]
    score: float
    tier: str  # "confident" | "tentative" | "refused"


class Retriever:
    def __init__(self, semantic: SemanticMemory, episodic: EpisodicMemory):
        self.semantic = semantic
        self.episodic = episodic

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        facts = self.semantic.search(query, top_k=top_k)
        score = self.semantic.top_score(query)
        if score >= CONFIDENT_THRESHOLD:
            tier = "confident"
        elif score >= TENTATIVE_THRESHOLD:
            tier = "tentative"
        else:
            tier = "refused"
        return RetrievalResult(facts=facts, score=score, tier=tier)

    def answer(self, query: str) -> str:
        """The closed-world gate: never states something not actually in
        semantic memory, and says so honestly when nothing matches well
        enough. Logs the interaction either way."""
        result = self.retrieve(query)

        if result.tier == "refused":
            text = "I haven't learned about that yet. Try `learn` to seed the knowledge base."
            self.episodic.log_event(EpisodicLog(kind="refusal", content=f"Q: {query}", outcome="neutral"))
            return text

        top = result.facts[0]
        if result.tier == "tentative":
            text = (
                f"I have partial information (confidence {result.score:.2f}): {top.text}\n"
                f"Take this with some caution — it didn't clear the confident threshold."
            )
        else:  # confident
            text = top.text

        self.episodic.log_event(
            EpisodicLog(kind="answer", content=f"Q: {query} -> {top.text}", outcome="success")
        )
        return text
