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
