"""Tests for cross-tier retrieval (semantic + project) in Phase 1."""
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
def embedder():
    return EmbeddingEngine(force_fallback=True)


@pytest.fixture
def retriever(tmp_path, embedder):
    semantic = SemanticMemory(tmp_path / "semantic.db", embedder)
    episodic = EpisodicMemory(tmp_path / "episodic.db")
    project = ProjectMemory(tmp_path / "projects.db", embedder)
    brain = MockBrain(embedder=embedder)
    
    # Seed semantic memory
    semantic.add_fact(Fact(text="Unity uses the Rigidbody component for physics-based movement.", topic="unity", confidence=0.9))
    
    # Mock some project files directly into the DB for testing
    project.get_or_create_project(tmp_path)
    vec = embedder.embed("PlayerController.cs handles jumping and WASD movement.")
    project.conn.execute(
        "INSERT INTO project_files (project_id, path, sha256_hash, summary, embedding, created_at, updated_at) "
        "VALUES (1, 'Scripts/PlayerController.cs', 'hash', 'Handles jumping and WASD movement.', ?, 'now', 'now')",
        (f"[{','.join(map(str, vec))}]",)
    )
    project.conn.commit()
    
    return Retriever(semantic, episodic, project, brain)


def test_cross_tier_retrieve_finds_project_files(retriever):
    result = retriever.retrieve("Where is player movement handled?")
    
    # "movement" matches both the semantic fact and the project file using the fallback hash embedder
    # Let's ensure the project file is found.
    assert any(pf.path == "Scripts/PlayerController.cs" for pf in result.project_files)


def test_grounded_prompt_formatting(retriever):
    result = retriever.retrieve("Where is player movement handled?")
    prompt = retriever.format_grounded_prompt("Where is player movement handled?", result)
    
    assert "SEMANTIC FACTS" in prompt
    assert "Rigidbody component" in prompt
    assert "PROJECT FILES" in prompt
    assert "Scripts/PlayerController.cs" in prompt
    assert "RULES" in prompt
    assert "Cite file paths" in prompt
