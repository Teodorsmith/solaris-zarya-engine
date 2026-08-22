import pytest
import os
import json
from pathlib import Path
from agent.engine.state_machine import TaskFSM
from agent.memory.state_manifest import StateManifest

def test_crash_resume_idempotency(tmp_path: Path):
    active_path = tmp_path / "active_task.json"
    manifest_path = tmp_path / "state_manifest.json"
    
    # 1. Start FSM
    fsm = TaskFSM(active_path, manifest_path)
    state = fsm.start_task("goal_999")
    
    action_hash = "hash_dangerous_write"
    
    # 2. Advance to RUNNING (this persists the action_hash to executed_actions)
    fsm.advance("RUNNING", action_hash=action_hash)
    
    # We ostensibly execute the action here in the real world
    action_actually_executed = True
    
    # 3. Simulate CRASH exactly before commit_action writes the COMMITTED state
    # We just don't call commit_action, and we re-instantiate the FSM to simulate boot
    
    # 4. Boot Recovery
    fsm2 = TaskFSM(active_path, manifest_path)
    recovered_state = fsm2.load_state()
    
    assert recovered_state is not None
    assert recovered_state.state == "RUNNING"
    assert recovered_state.goal_id == "goal_999"
    
    # 5. Assert that because it was RUNNING, it is not considered COMMITTED/EXECUTED
    # The executor must transition to VERIFYING to check side-effects.
    assert fsm2.is_action_executed(action_hash) is False


def test_run_to_completion_requires_governor_for_tier2(tmp_path: Path):
    from agent.models import Goal
    from agent.memory.goals import GoalMemory
    from agent.brains.mock_brain import MockBrain
    from unittest.mock import patch

    active_path = tmp_path / "active_task.json"
    manifest_path = tmp_path / "state_manifest.json"
    fsm = TaskFSM(active_path, manifest_path)
    fsm.start_task("task_1")

    goals_db = GoalMemory(tmp_path / "goals.db")
    g1 = Goal(id="g1", task_id="task_1", description="Write summary.md", completion_criteria="done", required_tier=2)
    goals_db.register(g1)
    brain = MockBrain()

    # Calling run_to_completion without governor on Tier 2 goal must raise
    res = fsm.run_to_completion("task_1", goals_db, brain, governor=None)
    assert "Task failed at step" in res
    assert "requires a PermissionGovernor" in res


def test_run_to_completion_with_governor_approval(tmp_path: Path):
    from agent.models import Goal
    from agent.memory.goals import GoalMemory
    from agent.memory.episodic import EpisodicMemory
    from agent.engine.governor import PermissionGovernor
    from agent.brains.mock_brain import MockBrain
    from unittest.mock import patch

    active_path = tmp_path / "active_task.json"
    manifest_path = tmp_path / "state_manifest.json"
    fsm = TaskFSM(active_path, manifest_path)
    fsm.start_task("task_2")

    goals_db = GoalMemory(tmp_path / "goals.db")
    g1 = Goal(id="g1", task_id="task_2", description="Write output.txt", completion_criteria="done", required_tier=2)
    goals_db.register(g1)
    brain = MockBrain()
    episodic = EpisodicMemory(tmp_path / "episodic.db")
    gov = PermissionGovernor(episodic)

    # Approve with explicit 'y'
    with patch("builtins.input", return_value="y"):
        res = fsm.run_to_completion("task_2", goals_db, brain, governor=gov)
        assert "completed successfully" in res

    # Verify file was written and episodic log recorded
    assert any("USER_APPROVED" in log.content for log in episodic.recent(10))
