"""Deterministic Task FSM managing data/active_task.json."""
import json
import os
import logging
from pathlib import Path
from agent.models import TaskState

logger = logging.getLogger(__name__)

from agent.memory.state_manifest import StateManifest

class TaskFSM:
    def __init__(self, state_file: str | Path, manifest_file: str | Path):
        self.state_file = Path(state_file)
        self.tmp_file = self.state_file.with_suffix('.json.tmp')
        self.manifest = StateManifest(manifest_file)
        
    def _write_state(self, state: TaskState) -> None:
        try:
            with open(self.tmp_file, 'w', encoding='utf-8') as f:
                f.write(state.model_dump_json(indent=2))
            os.replace(self.tmp_file, self.state_file)
            self.manifest.write_manifest(state)
        except Exception as e:
            logger.error(f"Failed to atomically write FSM state: {e}")
            raise

    def load_state(self) -> TaskState | None:
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TaskState(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Corrupt active_task.json detected: {e}")
            return None

    def start_task(self, goal_id: str) -> TaskState:
        state = TaskState(goal_id=goal_id, state="PENDING")
        self._write_state(state)
        return state

    def update_task(self, state: TaskState) -> None:
        from agent.models import _now
        state.updated_at = _now()
        self._write_state(state)

    def advance(self, new_state: str, action_hash: str | None = None) -> TaskState:
        state = self.load_state()
        if not state:
            raise RuntimeError("No active task to advance.")
        
        state.state = new_state
        if action_hash:
            state.action_hash = action_hash
            state.pending_action_hash = action_hash
            state.step_index += 1
            
        self.update_task(state)
        return state
        
    def is_action_executed(self, action_hash: str) -> bool:
        state = self.load_state()
        if not state:
            return False
        return action_hash in state.executed_actions
        
    def commit_action(self, action_hash: str) -> TaskState:
        state = self.load_state()
        if not state:
            raise RuntimeError("No active task to commit.")
                
        if action_hash not in state.executed_actions:
            state.executed_actions.append(action_hash)
            if len(state.executed_actions) > 100:
                state.executed_actions = state.executed_actions[-100:]
                
        if state.pending_action_hash == action_hash:
            state.pending_action_hash = None
                
        state.state = "COMMITTED"
        self.update_task(state)
        return state
        
    def record_failure(self) -> TaskState:
        state = self.load_state()
        if state:
            state.consecutive_failures += 1
            self.update_task(state)
        return state

    def clear_task(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
        if self.tmp_file.exists():
            self.tmp_file.unlink()
        self.manifest.write_manifest(None)

    def get_idempotency_key(self) -> str | None:
        state = self.load_state()
        if not state or not state.goal_id or not state.action_hash:
            return None
        return f"{state.goal_id}_{state.step_index}_{state.action_hash}"
