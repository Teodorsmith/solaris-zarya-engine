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

    def run_to_completion(self, task_id: str, goals_db, brain, procedural=None, validator=None) -> str:
        # 1. Clean up orphaned pending goals from past aborted/interrupted runs
        if hasattr(goals_db, "abort_orphaned_goals"):
            goals_db.abort_orphaned_goals(task_id)

        # 2. Retrieve only goals scoped to this task_id
        task_goals = goals_db.get_all_goals(task_id)
        if not task_goals:
            # Fallback if no goals explicitly tagged with this task_id
            task_goals = goals_db.get_all_goals()

        goal_map = {g.id: g for g in task_goals}
        total_goals = len(task_goals)
        step_outputs: dict[str, str] = {}
        
        while True:
            ready_goal = None
            completed_count = sum(1 for g in task_goals if g.status == "COMPLETED")
            
            if completed_count == total_goals and total_goals > 0:
                self.advance("COMPLETED")
                summary_lines = [f"- **{g.description}**: {step_outputs.get(g.id, 'Done')[:120]}" for g in task_goals]
                return f"Final Result: Task '{task_id}' completed successfully ({total_goals}/{total_goals} steps).\n" + "\n".join(summary_lines)
                
            for g in task_goals:
                if g.status == "PENDING":
                    deps_met = all(goal_map[d].status == "COMPLETED" for d in g.dependencies if d in goal_map)
                    if deps_met:
                        ready_goal = g
                        break
                        
            if not ready_goal:
                if any(g.status == "PENDING" for g in task_goals):
                    self.advance("FAILED")
                    raise RuntimeError("Goal deadlock: pending goals remain but dependencies unmet.")
                break
                
            step_num = completed_count + 1
            print(f"[Step {step_num}/{total_goals}] Executing: {ready_goal.description}...", end=" ", flush=True)
            self.advance("RUNNING", action_hash=ready_goal.id)
            
            try:
                # Gather prior step context
                ctx_parts = []
                for d in ready_goal.dependencies:
                    if d in step_outputs:
                        ctx_parts.append(f"Output of prior step ({goal_map[d].description}):\n{step_outputs[d]}")
                ctx = "\n\n".join(ctx_parts) if ctx_parts else "No prior step dependencies."
                
                if ready_goal.required_tier == 0:
                    prompt = (
                        f"You are executing a sub-goal in an autonomous task.\n"
                        f"Goal: {ready_goal.description}\n"
                        f"Completion Criteria: {ready_goal.completion_criteria}\n"
                        f"Context from completed steps:\n{ctx}\n\n"
                        f"Provide a concise, direct result satisfying the criteria."
                    )
                    result = brain.generate(prompt)
                else:
                    # Tool dispatch
                    prompt = (
                        f"Tool Execution for Goal: {ready_goal.description}\n"
                        f"Criteria: {ready_goal.completion_criteria}\n"
                        f"Context:\n{ctx}"
                    )
                    result = brain.generate(prompt)
                    
                step_outputs[ready_goal.id] = result.strip()
                ready_goal.status = "COMPLETED"
                goals_db.update_status(ready_goal.id, "COMPLETED")
                self.commit_action(ready_goal.id)
                print("-> Done.")
                
            except Exception as e:
                ready_goal.status = "FAILED"
                goals_db.update_status(ready_goal.id, "FAILED")
                self.record_failure()
                self.advance("FAILED")
                print(f"-> Failed: {e}")
                return f"Task failed at step: {ready_goal.description} (Error: {e})"
                
        self.advance("COMPLETED")
        return "Final Result: Task completed successfully."
