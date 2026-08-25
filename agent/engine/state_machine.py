# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Deterministic Task FSM managing data/active_task.json."""

import json
import logging
import os
import re
from pathlib import Path

from agent.models import TaskState

logger = logging.getLogger(__name__)

from agent.memory.state_manifest import StateManifest


class TaskFSM:
    # States after which active_task.json and the manifest are automatically
    # cleared so the next boot does not offer a stale resume prompt.
    _TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "ABORTED", "CANCELLED"})

    def __init__(self, state_file: str | Path, manifest_file: str | Path):
        self.state_file = Path(state_file)
        self.tmp_file = self.state_file.with_suffix(".json.tmp")
        self.manifest = StateManifest(manifest_file)

    def _write_state(self, state: TaskState) -> None:
        try:
            with open(self.tmp_file, "w", encoding="utf-8") as f:
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
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TaskState(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Corrupt active_task.json detected: %s — clearing.", e)
            self.clear_task()
            return None

    def start_task(
        self,
        goal_id: str,
        prompt_hash: str | None = None,
        strategy_label: str | None = None,
    ) -> TaskState:
        state = TaskState(
            goal_id=goal_id,
            state="PENDING",
            prompt_hash=prompt_hash,
            strategy_label=strategy_label,
        )
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

        # Auto-clean after any terminal state so the next boot is fresh.
        if new_state in self._TERMINAL_STATES:
            logger.info(
                "FSM reached terminal state '%s' — clearing active_task.json.",
                new_state,
            )
            self.clear_task()

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

    def run_to_completion(
        self,
        task_id: str,
        goals_db,
        brain,
        procedural=None,
        validator=None,
        project_memory=None,
        governor=None,
        episodic_memory=None,
        is_autonomous: bool = False,
    ) -> str:
        """Execute all goals in the DAG to completion.

        Args:
            task_id:        ID of the task whose goals to execute.
            goals_db:       GoalMemory instance.
            brain:          Active brain for generation.
            procedural:     ProceduralMemory (optional, for skill dispatch).
            validator:      SkillValidator (optional).
            project_memory: ProjectMemory (optional).  When provided, any file
                            written by a tier-2 goal is immediately indexed via
                            ``upsert_file`` so it is searchable without a manual
                            ``project index .``.
            governor:       PermissionGovernor (optional). When provided, tier-2
                            actions and file writes are routed through the governor.
            is_autonomous:  bool. If True, unsupervised Tier 2 file writes are
                            automatically denied by the governor.
        """
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
                summary_lines = [
                    f"- **{g.description}**: {step_outputs.get(g.id, 'Done')[:120]}"
                    for g in task_goals
                ]
                return (
                    f"Final Result: Task '{task_id}' completed successfully "
                    f"({total_goals}/{total_goals} steps).\n" + "\n".join(summary_lines)
                )

            for g in task_goals:
                if g.status == "PENDING":
                    deps_met = all(
                        goal_map[d].status == "COMPLETED"
                        for d in g.dependencies
                        if d in goal_map
                    )
                    if deps_met:
                        ready_goal = g
                        break

            if not ready_goal:
                if any(g.status == "PENDING" for g in task_goals):
                    self.advance("FAILED")
                    raise RuntimeError(
                        "Goal deadlock: pending goals remain but dependencies unmet."
                    )
                break

            # Two-marker crash resume check
            state = self.load_state()
            if state and state.pending_action_hash == ready_goal.id:
                if ready_goal.id in state.executed_actions:
                    ready_goal.status = "COMPLETED"
                    goals_db.update_status(ready_goal.id, "COMPLETED")
                    step_outputs[ready_goal.id] = "Already completed (recovered from crash)."
                    continue
                else:
                    print(f"\n[FSM] Ambiguous resume detected.")
                    print(f"Task: {task_id}")
                    print(f"Step: {completed_count + 1}/{total_goals}")
                    print(f"Action: {ready_goal.description}")
                    print("Status: unknown — the action may or may not have completed.\n")
                    print("Options:")
                    print("  r  re-run this action")
                    print("  s  skip it and continue")
                    print("  f  fail this step and stop")
                    print("  a  abort task")
                    ans = input("Choice [r/s/f/a]: ").strip().lower()
                    
                    if ans == "s":
                        ready_goal.status = "COMPLETED"
                        goals_db.update_status(ready_goal.id, "COMPLETED")
                        step_outputs[ready_goal.id] = "Skipped by user after crash."
                        continue
                    elif ans == "r":
                        pass # proceed with execution normally
                    elif ans == "f":
                        ready_goal.status = "FAILED"
                        goals_db.update_status(ready_goal.id, "FAILED")
                        self.advance("FAILED")
                        return "Task failed by user during crash resume."
                    else:
                        self.advance("FAILED")
                        return "Task aborted by user during crash resume."

            step_num = completed_count + 1
            print(
                f"[Step {step_num}/{total_goals}] Executing: {ready_goal.description}...",
                end=" ",
                flush=True,
            )
            self.advance("RUNNING", action_hash=ready_goal.id)

            try:
                # Gather prior step context
                ctx_parts = []
                for d in ready_goal.dependencies:
                    if d in step_outputs:
                        ctx_parts.append(
                            f"Output of prior step ({goal_map[d].description}):\n{step_outputs[d]}"
                        )
                ctx = (
                    "\n\n".join(ctx_parts)
                    if ctx_parts
                    else "No prior step dependencies."
                )

                if ready_goal.required_tier == 0:
                    # Auto-detect misclassified file-write goals (second line of
                    # defence after the planner's deterministic override).
                    if self._is_file_action(ready_goal.description):
                        logger.warning(
                            "Executor detected file-action in Tier-0 goal '%s' — "
                            "escalating to Tier 2.",
                            ready_goal.description,
                        )
                        ready_goal.required_tier = 2

                if ready_goal.required_tier == 0:
                    prompt = (
                        f"You are executing a sub-goal in an autonomous task.\n"
                        f"Goal: {ready_goal.description}\n"
                        f"Completion Criteria: {ready_goal.completion_criteria}\n"
                        f"Context from completed steps:\n{ctx}\n\n"
                        f"Provide a concise, direct result satisfying the criteria."
                    )
                    result = brain.generate(prompt)

                elif ready_goal.required_tier >= 2:
                    # Tier-2: file-write action with per-file HITL approval.
                    # Ask the brain to produce the file content first so the
                    # user can see what will be written before approving.
                    prompt = (
                        f"You are executing a file-write step in an autonomous task.\n"
                        f"Goal: {ready_goal.description}\n"
                        f"Completion Criteria: {ready_goal.completion_criteria}\n"
                        f"Context from completed steps:\n{ctx}\n\n"
                        f"Output ONLY the raw file content to be written. "
                        f"Do not include any preamble, explanation, or markdown fencing."
                    )
                    file_content = brain.generate(prompt)

                    # Determine filename early so the user sees it in the prompt
                    filename = self._extract_filename(ready_goal.description)

                    # Tier 2 file writes strictly require a PermissionGovernor
                    if governor is None:
                        raise RuntimeError(
                            f"Task execution for Tier-2 file write '{filename}' requires a PermissionGovernor."
                        )

                    approved = governor.request_file_write_permission(
                        file_path=filename,
                        content_preview=file_content,
                        goal_description=ready_goal.description,
                    )
                    if not approved:
                        raise RuntimeError(
                            f"User denied file-write for '{filename}'. Task aborted."
                        )

                    written_path = self._write_task_file(
                        ready_goal.description, file_content, project_memory, brain
                    )
                    
                    if project_memory and project_memory.active_root:
                        project_memory.upsert_file(written_path, brain=brain, project_root=project_memory.active_root)

                    result = f"Written to {written_path}: " + file_content[
                        :200
                    ].replace("\n", " ")

                else:
                    # Tier-1: sandboxed generation (no persistent file write)
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
                
                if episodic_memory:
                    import json
                    # look for recent failure for this step to log repair
                    fail_rows = episodic_memory.conn.execute(
                        "SELECT id, content FROM episodic_log WHERE kind = 'task_failure' ORDER BY id DESC LIMIT 50"
                    ).fetchall()
                    for r in fail_rows:
                        try:
                            fail_payload = json.loads(r["content"])
                            if fail_payload.get("step_id") == ready_goal.id:
                                repair_payload = {
                                    "original_fail_id": r["id"],
                                    "task_id": task_id,
                                    "step_id": ready_goal.id,
                                    "chosen_code": locals().get("file_content", locals().get("result", "")),
                                    "exit_code": 0
                                }
                                episodic_memory.log_action(json.dumps(repair_payload), success=True, kind="task_repair_resolved")
                                break
                        except Exception:
                            pass

            except Exception as e:
                ready_goal.status = "FAILED"
                goals_db.update_status(ready_goal.id, "FAILED")
                self.record_failure()
                self.advance("FAILED")
                print(f"-> Failed: {e}")
                
                if episodic_memory:
                    import json
                    payload = {
                        "step_id": ready_goal.id,
                        "task_id": task_id,
                        "prompt": locals().get("prompt", ""),
                        "code": locals().get("file_content", ""),
                        "error": str(e),
                        "exit_code": 1
                    }
                    episodic_memory.log_action(json.dumps(payload), success=False, kind="task_failure")
                    
                return f"Task failed at step: {ready_goal.description} (Error: {e})"

        self.advance("COMPLETED")
        return "Final Result: Task completed successfully."

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    # Keywords that signal a filesystem write at execution time.
    # Used as a second safety net: if the planner misclassified a file-write
    # goal as Tier 0 or 1, the executor detects it and escalates to Tier 2.
    _FILE_ACTION_KEYWORDS: tuple[str, ...] = (
        "create a new",
        "create file",
        "write a",
        "write file",
        "write to file",
        "write the",
        "save to",
        "save file",
        "save a",
        "output to file",
        "generate file",
        "produce file",
        "store to",
        "write output",
        ".md",
        ".txt",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".rst",
        ".html",
        ".csv",
    )

    _FILE_NAME_PATTERNS = [
        re.compile(r'["\']([\w./-]+\.[a-zA-Z]{1,5})["\']'),  # quoted filename
        re.compile(
            r"\b([\w-]+\.(?:md|txt|py|json|yaml|yml|rst|html|csv))\b"
        ),  # bare extension
    ]

    def _is_file_action(self, description: str) -> bool:
        """Return True if *description* contains keywords that indicate a
        filesystem write operation — used to catch goals the planner
        misclassified as Tier 0 or 1."""
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in self._FILE_ACTION_KEYWORDS)

    def _extract_filename(self, goal_description: str) -> str:
        """Extract the target filename from *goal_description*, or generate a
        slug-based fallback.  Shared by the HITL preview and _write_task_file."""
        for pat in self._FILE_NAME_PATTERNS:
            m = pat.search(goal_description)
            if m:
                return m.group(1)
        slug = re.sub(r"[^\w\s-]", "", goal_description.lower())
        slug = re.sub(r"[\s-]+", "_", slug).strip("_")[:40]
        return f"{slug or 'task_output'}.md"

    def _write_task_file(
        self,
        goal_description: str,
        content: str,
        project_memory,
        brain,
    ) -> str:
        """Determine filename from *goal_description*, write *content* to disk,
        then call ``project_memory.upsert_file()`` so the file is immediately
        searchable.

        Returns the path string that was written.
        """
        # 1. Extract filename (shared logic with HITL preview)
        filename = self._extract_filename(goal_description)

        # 2. Resolve output path — write into workspace root when known
        if project_memory is not None and project_memory.active_root is not None:
            out_dir = project_memory.active_root
        else:
            out_dir = Path.cwd()

        out_path = out_dir / filename

        # 3. Write file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        logger.info("task file-write: %s (%d bytes)", out_path, len(content))

        # 4. Immediately index — non-fatal if it fails
        if project_memory is not None:
            project_memory.upsert_file(out_path, brain=brain)
            logger.info("task file-write: upserted %s into project memory", out_path)

        return str(out_path)
