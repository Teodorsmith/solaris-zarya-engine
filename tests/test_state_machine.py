import pytest
import os
from pathlib import Path
from agent.engine.state_machine import TaskFSM
from agent.models import TaskState

def test_atomic_write_and_idempotency(tmp_path: Path):
    fsm = TaskFSM(tmp_path / "active_task.json", tmp_path / "state_manifest.json")
    state = fsm.start_task("goal_123")
    assert state.state == "PENDING"
    
    # advance state
    fsm.advance("RUNNING", action_hash="hash_abc")
    
    loaded = fsm.load_state()
    assert loaded.state == "RUNNING"
    assert loaded.step_index == 1
    
    key = fsm.get_idempotency_key()
    assert key == "goal_123_1_hash_abc"
    
    # Test corruption recovery
    with open(tmp_path / "active_task.json", "w") as f:
        f.write("{ invalid json")
    
    assert fsm.load_state() is None
