"""
Tests for the confidence-gated retriever. Uses force_fallback embeddings
(see test_memory.py) for fast, offline, deterministic runs.
"""
import pytest

from agent.engine.retriever import Retriever
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.project import ProjectMemory
from agent.brains.mock_brain import MockBrain
from agent.models import Fact


@pytest.fixture
def retriever(tmp_path):
    embedder = EmbeddingEngine(force_fallback=True)
    semantic = SemanticMemory(tmp_path / "semantic.db", embedder)
    episodic = EpisodicMemory(tmp_path / "episodic.db")
    project = ProjectMemory(tmp_path / "projects.db", embedder)
    brain = MockBrain(embedder=embedder)
    semantic.add_fact(Fact(text="git status shows the working tree state.", topic="git", confidence=0.9))
    return Retriever(semantic, episodic, project, brain)


def test_close_match_is_confident(retriever):
    # Near-identical to the seeded fact, so even the crude fallback
    # embedding (bag-of-words, no real semantics) scores it high. This
    # isolates "does the confident path work" from "is the embedding good
    # enough to recognize paraphrases" -- it isn't; that's what FastEmbed
    # is for on a real machine with model access.
    result = retriever.retrieve("git status shows the working tree state")
    assert result.tier == "confident"
    assert "working tree" in result.facts[0].text


def test_unrelated_query_is_refused(retriever):
    result = retriever.retrieve("how do I defeat the ender dragon")
    assert result.tier == "refused"


def test_answer_never_asserts_on_refusal(retriever):
    text = retriever.answer("how do I defeat the ender dragon")
    assert "haven't learned" in text
    assert "ender dragon" not in text  # never echoes an ungrounded claim back


def test_answer_logs_an_episode(retriever):
    before = retriever.episodic.count()
    retriever.answer("git status shows the working tree state")
    assert retriever.episodic.count() == before + 1


def test_empty_memory_always_refuses(tmp_path):
    embedder = EmbeddingEngine(force_fallback=True)
    semantic = SemanticMemory(tmp_path / "semantic_empty.db", embedder)
    episodic = EpisodicMemory(tmp_path / "episodic_empty.db")
    project = ProjectMemory(tmp_path / "projects_empty.db", embedder)
    brain = MockBrain(embedder=embedder)
    retriever = Retriever(semantic, episodic, project, brain)
    result = retriever.retrieve("anything at all")
    assert result.tier == "refused"
    assert result.score == 0.0
