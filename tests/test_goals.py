import pytest
from pathlib import Path
from agent.models import Goal
from agent.memory.goals import GoalMemory

def test_goal_memory(tmp_path: Path):
    mem = GoalMemory(tmp_path / "goals.db")
    g1 = Goal(id="g1", description="goal 1", completion_criteria="ok")
    g2 = Goal(id="g2", description="goal 2", parent_id="g1", dependencies=["g1"], completion_criteria="ok")
    mem.register(g1)
    mem.register(g2)
    
    assert len(mem.get_pending_goals()) == 2
    mem.update_status("g1", "COMPLETED")
    assert len(mem.get_pending_goals()) == 1
    
    loaded = mem.get_goal("g2")
    assert loaded.parent_id == "g1"
    assert loaded.dependencies == ["g1"]
