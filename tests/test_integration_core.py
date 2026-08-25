import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from agent.engine.state_machine import TaskFSM
from agent.engine.governor import PermissionGovernor
from agent.models import Goal, Fact
from agent.memory.semantic import SemanticMemory
from agent.memory.project import ProjectMemory
from agent.memory.episodic import EpisodicMemory
from agent.memory.goals import GoalMemory
from agent.brains.mock_brain import MockBrain
from agent.commands.facts import handle_correct


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def fsm(temp_workspace):
    return TaskFSM(temp_workspace / "active.json", temp_workspace / "manifest.json")


@pytest.fixture
def governor():
    ep = MagicMock()
    return PermissionGovernor(ep)


@pytest.fixture
def goals_db(temp_workspace):
    from agent.memory.goals import GoalMemory
    mem = GoalMemory(temp_workspace / "goals.db")
    yield mem
    mem.conn.close()


@pytest.fixture
def semantic(temp_workspace):
    from agent.memory.embeddings import EmbeddingEngine
    mem = SemanticMemory(str(temp_workspace / "semantic.db"), embedder=EmbeddingEngine())
    yield mem
    mem.conn.close()


def test_task_tier2_governor_prompt_blocks_without_approval(fsm, goals_db, governor):
    goal = Goal(id="g1", task_id="t1", description="Write file a.txt", required_tier=2, completion_criteria="Done")
    goals_db.register(goal)
    brain = MockBrain(embedder=MagicMock())
    fsm.start_task("t1")
    
    with patch("builtins.input", return_value="n"):
        result = fsm.run_to_completion("t1", goals_db, brain, governor=governor)
    
    assert "aborted" in result.lower() or "failed" in result.lower()
    g = goals_db.get_goal("g1")
    assert g.status == "FAILED"


def test_task_tier2_approval_writes_file(fsm, goals_db, governor, temp_workspace):
    goal = Goal(id="g2", task_id="t2", description="Write file approval_test.txt", required_tier=2, completion_criteria="Done")
    goals_db.register(goal)
    brain = MockBrain(embedder=MagicMock())
    fsm.start_task("t2")
    
    project_mem = MagicMock()
    project_mem.active_root = temp_workspace

    with patch("builtins.input", return_value="y"):
        with patch.object(fsm, "_write_task_file", return_value=str(temp_workspace / "approval_test.txt")):
            result = fsm.run_to_completion("t2", goals_db, brain, governor=governor, project_memory=project_mem)
            
    assert "completed successfully" in result.lower()
    g = goals_db.get_goal("g2")
    assert g.status == "COMPLETED"


def test_task_tier2_auto_index_after_write(fsm, goals_db, governor, temp_workspace):
    goal = Goal(id="g3", task_id="t3", description="Write file auto_index.txt", required_tier=2, completion_criteria="Done")
    goals_db.register(goal)
    brain = MockBrain(embedder=MagicMock())
    fsm.start_task("t3")
    
    project_mem = MagicMock()
    project_mem.active_root = temp_workspace
    written_path = str(temp_workspace / "auto_index.txt")

    with patch("builtins.input", return_value="y"):
        with patch.object(fsm, "_write_task_file", return_value=written_path):
            fsm.run_to_completion("t3", goals_db, brain, governor=governor, project_memory=project_mem)
            
    project_mem.upsert_file.assert_called_once_with(written_path, brain=brain, project_root=temp_workspace)


def test_correct_fact_by_id(semantic):
    f1 = Fact(text="Apples are blue.", confidence=0.8, source_type="seed", topic="fruits")
    created, fid = semantic.add_fact(f1)
    
    episodic = MagicMock()
    handle_correct(f"{fid} Apples are red.", semantic, episodic)
    
    # Old fact is superseded
    old_fact_row = semantic.conn.execute("SELECT is_superseded FROM facts WHERE id = ?", (fid,)).fetchone()
    assert old_fact_row["is_superseded"] == 1
    
    # New fact exists
    new_facts = semantic.search("Apples are red")
    assert len(new_facts) > 0
    assert new_facts[0].text == "Apples are red."
    assert new_facts[0].confidence == 1.0


def test_correct_fact_by_old_statement(semantic):
    f1 = Fact(text="Bananas are square.", confidence=0.8, source_type="seed", topic="fruits")
    created, fid = semantic.add_fact(f1)
    
    episodic = MagicMock()
    handle_correct('fruits "Bananas are square." -> "Bananas are curved."', semantic, episodic)
    
    # Old fact is superseded
    old_fact_row = semantic.conn.execute("SELECT is_superseded FROM facts WHERE id = ?", (fid,)).fetchone()
    assert old_fact_row["is_superseded"] == 1
    
    # New fact exists
    new_facts = semantic.search("Bananas are curved")
    assert len(new_facts) > 0
    assert new_facts[0].text == "Bananas are curved."
    assert new_facts[0].confidence == 1.0


def test_mockbrain_dynamic_skill_name():
    brain = MockBrain(embedder=MagicMock())
    response = brain.generate("Topic: data_parser")
    assert "data_parser" in response
    assert "email_normalizer" not in response


def test_fsm_resume_ambiguous_action_does_not_rerun(fsm, goals_db):
    goal = Goal(id="g4", task_id="t4", description="Print something", required_tier=1, completion_criteria="Done")
    goals_db.register(goal)
    brain = MockBrain(embedder=MagicMock())
    fsm.start_task("t4")
    
    # Force ambiguous state
    state = fsm.load_state()
    state.pending_action_hash = "g4"
    state.executed_actions = []
    fsm.update_task(state)

    # User chooses 's' for skip
    with patch("builtins.input", return_value="s"):
        fsm.run_to_completion("t4", goals_db, brain)
        
    g = goals_db.get_goal("g4")
    assert g.status == "COMPLETED"


def test_fsm_resume_committed_action_skips(fsm, goals_db):
    goal = Goal(id="g5", task_id="t5", description="Print something", required_tier=1, completion_criteria="Done")
    goals_db.register(goal)
    brain = MockBrain(embedder=MagicMock())
    fsm.start_task("t5")
    
    # Force committed state
    state = fsm.load_state()
    state.pending_action_hash = "g5"
    state.executed_actions = ["g5"]
    fsm.update_task(state)

    with patch("builtins.input") as mock_input:
        fsm.run_to_completion("t5", goals_db, brain)
        mock_input.assert_not_called()
        
    g = goals_db.get_goal("g5")
    assert g.status == "COMPLETED"
