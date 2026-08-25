"""
Unit tests for the memory tiers. Uses EmbeddingEngine(force_fallback=True)
so these run fast, offline, and deterministically -- no model download,
no network, no first-run latency.
"""

import pytest

from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.semantic import SemanticMemory
from agent.models import EpisodicLog, Fact, Skill


@pytest.fixture
def embedder():
    return EmbeddingEngine(force_fallback=True)


@pytest.fixture
def semantic(embedder, tmp_path):
    return SemanticMemory(tmp_path / "semantic.db", embedder)


@pytest.fixture
def episodic(tmp_path):
    return EpisodicMemory(tmp_path / "episodic.db")


@pytest.fixture
def procedural(tmp_path):
    return ProceduralMemory(tmp_path / "procedural.db")


def test_add_and_search_fact(semantic):
    created, fid = semantic.add_fact(
        Fact(text="git status shows the working tree state.", topic="git")
    )
    assert created is True
    results = semantic.search("git status", top_k=3)
    assert any(r.id == fid for r in results)


def test_dedup_bumps_confidence_not_duplicate(semantic):
    created1, fid1 = semantic.add_fact(
        Fact(text="git status shows the working tree state.", confidence=0.7)
    )
    created2, fid2 = semantic.add_fact(
        Fact(text="git status shows the working tree state.", confidence=0.7)
    )
    assert created1 is True
    assert created2 is False
    assert fid1 == fid2
    assert semantic.count() == 1


def test_correct_fact_marks_user_corrected(semantic):
    _, fid = semantic.add_fact(Fact(text="git push uploads commits.", confidence=0.5))
    semantic.correct_fact(fid, "git push uploads local commits to a remote.")
    updated = next(f for f in semantic.list_all() if f.id == fid)
    assert updated.source_type == "user_corrected"
    assert updated.confidence == 1.0
    assert "remote" in updated.text


def test_episodic_log_and_trace(episodic):
    episodic.log_event(
        EpisodicLog(trace_id="t1", kind="query", content="hi", outcome="neutral")
    )
    episodic.log_event(
        EpisodicLog(trace_id="t1", kind="answer", content="hello", outcome="success")
    )
    episodic.log_event(
        EpisodicLog(trace_id="t2", kind="query", content="unrelated", outcome="neutral")
    )
    trace = episodic.get_trace("t1")
    assert len(trace) == 2


def test_episodic_prune_old(episodic):
    episodic.log_event(
        EpisodicLog(content="ancient", created_at="2000-01-01T00:00:00+00:00")
    )
    episodic.log_event(EpisodicLog(content="recent"))
    assert episodic.count() == 2
    removed = episodic.prune_old(days=90)
    assert removed == 1
    assert episodic.count() == 1


def test_procedural_register_and_load(procedural):
    procedural.register(Skill(name="greet", description="says hello"))
    loaded = procedural.load("greet")
    assert loaded is not None
    assert loaded.name == "greet"
    assert loaded.verification_tier == "mock"
    assert procedural.load("does_not_exist") is None


def test_procedural_register_upsert(procedural):
    id1 = procedural.register(
        Skill(
            name="bitwise_permission_flag",
            description="v1",
            verification_tier="mock",
        )
    )
    assert procedural.count() == 1
    loaded1 = procedural.load("bitwise_permission_flag")
    assert loaded1 is not None
    assert loaded1.description == "v1"
    assert loaded1.verification_tier == "mock"

    # Re-registering the same skill should update cleanly without crash
    id2 = procedural.register(
        Skill(
            name="bitwise_permission_flag",
            description="v2",
            verification_tier="real_local",
        )
    )
    assert procedural.count() == 1
    assert id1 == id2
    loaded2 = procedural.load("bitwise_permission_flag")
    assert loaded2 is not None
    assert loaded2.description == "v2"
    assert loaded2.verification_tier == "real_local"


def test_procedural_aliases(procedural):
    procedural.register_skill(
        name="bitwise_test",
        description="bitwise test skill",
        verification_tier="mock",
    )
    assert procedural.count() == 1
    loaded = procedural.load("bitwise_test")
    assert loaded is not None
    assert loaded.name == "bitwise_test"

    # Test add_skill alias (updating existing skill)
    procedural.add_skill(
        Skill(name="bitwise_test", description="updated bitwise test")
    )
    assert procedural.count() == 1
    assert (
        procedural.load("bitwise_test").description == "updated bitwise test"
    )
