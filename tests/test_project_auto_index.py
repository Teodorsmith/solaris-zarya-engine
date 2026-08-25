# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for project auto-indexing and Tier-2 HITL gate (Phase 4A prerequisite)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.brains.mock_brain import MockBrain
from agent.engine.governor import PermissionGovernor
from agent.engine.state_machine import TaskFSM
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.goals import GoalMemory
from agent.memory.project import ProjectMemory
from agent.memory.state_manifest import StateManifest
from agent.models import Goal


def _make_env(tmp: Path):
    embedder = EmbeddingEngine()
    episodic = EpisodicMemory(tmp / "episodic.db")
    project = ProjectMemory(tmp / "projects.db", embedder)
    goals = GoalMemory(tmp / "goals.db")
    manifest = StateManifest(tmp / "state_manifest.json")
    brain = MockBrain()
    governor = PermissionGovernor(episodic)
    fsm = TaskFSM(tmp / "active_task.json", tmp / "state_manifest.json")
    return embedder, episodic, project, goals, brain, governor, fsm


class TestFileWriteAutoIndexed(unittest.TestCase):
    """Tier-2 FSM goal -> file written -> immediately in project_files."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_file_write_auto_indexed(self):
        embedder, episodic, project, goals, brain, governor, fsm = _make_env(self._tmp)

        # Build a trivial 1-goal plan: write a markdown file
        plan_goal = Goal(
            description="Write summary.md",
            completion_criteria="summary.md exists",
            required_tier=2,
        )
        goals.register(plan_goal)

        # Start the FSM (required before run_to_completion)
        fsm.start_task(plan_goal.task_id)

        # Patch governor to auto-approve and brain to return deterministic content
        with patch.object(governor, "request_file_write_permission", return_value=True):
            with patch.object(
                brain, "generate", return_value="# Summary\n\nHello world."
            ):
                result = fsm.run_to_completion(
                    task_id=plan_goal.task_id,
                    goals_db=goals,
                    brain=brain,
                    project_memory=project,
                    governor=governor,
                )

        self.assertIn("completed", result.lower())

        # The file must now appear in project_files without a manual index
        count = project.count()
        self.assertGreater(
            count, 0, "Expected at least one file in project_files after FSM write"
        )

        files = project.conn.execute("SELECT path FROM project_files").fetchall()
        paths = [r[0] for r in files]
        self.assertTrue(
            any("summary" in p.lower() for p in paths),
            f"summary.md not found in project_files. Indexed paths: {paths}",
        )


class TestTier2FileWriteRequiresActionPrompt(unittest.TestCase):
    """Governor denial must block the file write entirely."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_tier2_file_write_requires_action_prompt(self):
        embedder, episodic, project, goals, brain, governor, fsm = _make_env(self._tmp)

        plan_goal = Goal(
            description="Write report.md",
            completion_criteria="report.md exists",
            required_tier=2,
        )
        goals.register(plan_goal)

        # Start the FSM
        fsm.start_task(plan_goal.task_id)

        written_files = []
        original_write = fsm._write_task_file

        def spy_write(desc, content, pm, b):
            written_files.append(desc)
            return original_write(desc, content, pm, b)

        with patch.object(
            governor, "request_file_write_permission", return_value=False
        ):
            with patch.object(brain, "generate", return_value="# Report content"):
                with patch.object(fsm, "_write_task_file", side_effect=spy_write):
                    result = fsm.run_to_completion(
                        task_id=plan_goal.task_id,
                        goals_db=goals,
                        brain=brain,
                        project_memory=project,
                        governor=governor,
                    )

        # File must NOT have been written
        self.assertEqual(written_files, [], "File was written despite governor denial")
        self.assertIn("fail", result.lower())


if __name__ == "__main__":
    unittest.main()
