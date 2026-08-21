import pytest
from pathlib import Path
from agent.models import Goal
from agent.memory.goals import GoalMemory

def test_goal_memory(tmp_path: Path):
    mem = GoalMemory(tmp_path / "goals.db")
    g1 = Goal(id="g1", task_id="t1", description="goal 1", completion_criteria="ok")
    g2 = Goal(id="g2", task_id="t1", description="goal 2", parent_id="g1", dependencies=["g1"], completion_criteria="ok")
    g3 = Goal(id="g3", task_id="t2", description="goal 3", completion_criteria="ok")
    mem.register(g1)
    mem.register(g2)
    mem.register(g3)
    
    assert len(mem.get_pending_goals()) == 3
    assert len(mem.get_pending_goals("t1")) == 2
    assert len(mem.get_pending_goals("t2")) == 1
    
    mem.update_status("g1", "COMPLETED")
    assert len(mem.get_pending_goals("t1")) == 1
    
    loaded = mem.get_goal("g2")
    assert loaded.parent_id == "g1"
    assert loaded.dependencies == ["g1"]
    assert loaded.task_id == "t1"

    # Test aborting orphaned goals
    aborted = mem.abort_orphaned_goals("t1")
    assert aborted == 1  # g3 should be aborted
    assert len(mem.get_pending_goals("t2")) == 0
    assert mem.get_goal("g3").status == "ABORTED"
