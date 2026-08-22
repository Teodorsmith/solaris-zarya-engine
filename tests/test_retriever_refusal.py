"""
Tests ensuring that project files do not cause false-positive tentative promotions
on unrelated queries, validating honest refusal behavior.
"""
import pytest
from pathlib import Path

from agent.brains.mock_brain import MockBrain
from agent.engine.retriever import Retriever
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.project import ProjectMemory
from agent.memory.semantic import SemanticMemory
from agent.models import Fact


@pytest.fixture
def populated_retriever(tmp_path):
    embedder = EmbeddingEngine(force_fallback=True)
    semantic = SemanticMemory(tmp_path / "semantic.db", embedder)
    episodic = EpisodicMemory(tmp_path / "episodic.db")
    project = ProjectMemory(tmp_path / "projects.db", embedder)
    brain = MockBrain(embedder=embedder)

    # Seed semantic facts
    semantic.add_fact(Fact(text="git status shows the working tree state.", topic="git", confidence=0.9))
    semantic.add_fact(Fact(text="git commit records changes to the repository.", topic="git", confidence=0.9))

    # Index project files
    project.get_or_create_project(tmp_path)
    for p, summary in [
        ("agent/config.py", "Central configuration paths constants and confidence gate thresholds."),
        ("agent/models.py", "Data models for facts goals skills and episodic logs."),
        ("agent/engine/retriever.py", "Hybrid retrieval and confidence gating logic."),
    ]:
        vec = embedder.embed(f"File {p}: {summary}")
        project.conn.execute(
            "INSERT INTO project_files (project_id, path, sha256_hash, summary, embedding, created_at, updated_at) "
            "VALUES (1, ?, 'hash123', ?, ?, 'now', 'now')",
            (p, summary, f"[{','.join(map(str, vec))}]")
        )
    project.conn.commit()

    return Retriever(semantic, episodic, project, brain)


def test_unrelated_query_refused_despite_indexed_project_files(populated_retriever):
    """Unrelated query must NOT be promoted to 'tentative' just because files exist in projects.db."""
    unrelated_queries = [
        "What is the best recipe for homemade pizza dough?",
        "How do I defeat the Ender Dragon in Minecraft?",
        "Current stock price of Apple AAPL today",
    ]
    for q in unrelated_queries:
        result = populated_retriever.retrieve(q)
        assert result.tier == "refused", f"Query '{q}' was unexpectedly given tier '{result.tier}'"
        assert result.score < 0.65, f"Query '{q}' scored {result.score}, expected < 0.65"
        
        answer = populated_retriever.answer(q)
        assert "haven't learned" in answer or "no project files matched" in answer
        assert "pizza" not in answer
        assert "Ender Dragon" not in answer


def test_exact_file_path_query_resolves_confidently(populated_retriever):
    """Query explicitly naming an indexed file path must resolve to that file and tier='confident'."""
    result = populated_retriever.retrieve("Explain agent/config.py")
    assert result.tier == "confident"
    assert any(pf.path == "agent/config.py" for pf in result.project_files)


def test_semantic_match_resolves_confidently(populated_retriever):
    """Query matching a known semantic fact resolves to confident tier."""
    result = populated_retriever.retrieve("git status shows the working tree state")
    assert result.tier == "confident"
    assert len(result.facts) > 0
