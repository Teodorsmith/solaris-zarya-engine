"""
Deterministic, offline brain. No API key, no network beyond the local
embedding model. Exists so Phase 0 runs and is testable with nothing
external required.

MockBrain does not reason — `generate()` is a stub that's never actually
called by Phase 0's main flow. `ask` is answered directly by the
retriever from retrieved facts (engine/retriever.py), and `learn` only
calls the seeder (see cli.py) — neither needs free-form generation. The
method still has to exist and work, though, so BaseBrain's contract is
fully implemented and Phase 1 can swap in a real brain without changing
any caller.
"""
from __future__ import annotations

from agent.brains.base import BaseBrain
from agent.memory.embeddings import EmbeddingEngine


class MockBrain(BaseBrain):
    def __init__(self, embedder: EmbeddingEngine | None = None):
        self._embedder = embedder

    def generate(self, prompt: str) -> str:
        return "MockBrain cannot generate free-form text. Configure a real brain for that (Phase 1)."

    def embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError(
                "MockBrain has no EmbeddingEngine configured. "
                "Pass one in: MockBrain(embedder=EmbeddingEngine())."
            )
        return self._embedder.embed(text)
