"""Unit tests for Tier 4 Project Codebase Memory."""
import pytest
from pathlib import Path

from agent.brains.mock_brain import MockBrain
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.project import ProjectMemory


@pytest.fixture
def embedder():
    return EmbeddingEngine(force_fallback=True)


@pytest.fixture
def project_mem(tmp_path, embedder):
    return ProjectMemory(tmp_path / "projects.db", embedder)


@pytest.fixture
def dummy_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    
    (ws / "main.py").write_text("def hello():\n    print('world')\n")
    (ws / "README.md").write_text("# Test Project\nDocs go here.")
    
    # Noise directories
    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("...")
    
    # Large/Binary files
    (ws / "large.pyc").write_text("binary data")
    
    return ws


def test_index_workspace_skips_ignored(project_mem, dummy_workspace, embedder):
    brain = MockBrain(embedder=embedder)
    
    indexed = project_mem.index_workspace(dummy_workspace, brain)
    assert indexed == 2  # main.py and README.md
    
    rows = project_mem.conn.execute("SELECT path FROM project_files").fetchall()
    paths = {r["path"] for r in rows}
    
    assert "main.py" in paths
    assert "README.md" in paths
    assert ".git/config" not in paths
    assert "large.pyc" not in paths


def test_heuristic_summary_fallback(project_mem, dummy_workspace, embedder):
    brain = MockBrain(embedder=embedder)
    project_mem.index_workspace(dummy_workspace, brain)
    
    rows = project_mem.conn.execute("SELECT path, summary FROM project_files").fetchall()
    summaries = {r["path"]: r["summary"] for r in rows}
    
    assert "Python module" in summaries["main.py"]
    assert "hello" in summaries["main.py"]
    assert "Test Project" in summaries["README.md"]


def test_incremental_indexing_skips_unchanged(project_mem, dummy_workspace, embedder):
    brain = MockBrain(embedder=embedder)
    
    # First pass
    assert project_mem.index_workspace(dummy_workspace, brain) == 2
    
    # Second pass, nothing changed
    assert project_mem.index_workspace(dummy_workspace, brain) == 0
    
    # Modify one file
    (dummy_workspace / "main.py").write_text("def new_func(): pass\n")
    
    # Add a new file
    (dummy_workspace / "utils.py").write_text("class Helper: pass\n")
    
    # Third pass
    assert project_mem.index_workspace(dummy_workspace, brain) == 2
    assert project_mem.count() == 3
