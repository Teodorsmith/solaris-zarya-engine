"""Tests for TaskPlanner tier enforcement and FSM file-action fallback.

These tests do NOT call a real LLM — they exercise the deterministic
post-processing logic (_enforce_file_tiers, _is_file_action) directly so
the safety guarantees can be verified offline.
"""
import pytest
from unittest.mock import MagicMock, patch
from agent.engine.task_planner import TaskPlanner, _FILE_ACTION_HINTS
from agent.engine.state_machine import TaskFSM
from agent.memory.goals import GoalMemory
from agent.models import Goal


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_goal(description: str, required_tier: int = 0) -> Goal:
    return Goal(
        description=description,
        completion_criteria="Done",
        required_tier=required_tier,
    )


def _make_planner() -> TaskPlanner:
    brain = MagicMock()
    goal_memory = MagicMock(spec=GoalMemory)
    return TaskPlanner(brain=brain, goal_memory=goal_memory)


# ── _enforce_file_tiers tests ─────────────────────────────────────────────────

class TestEnforceFileTiers:
    def test_tier0_with_md_extension_upgraded(self):
        """A Tier-0 goal mentioning .md must be upgraded to Tier 2."""
        goals = [_make_goal("Create a new markdown file named phase4_ready.md", required_tier=0)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 2

    def test_tier0_write_a_upgraded(self):
        goals = [_make_goal("Write a paragraph describing the phase goals", required_tier=0)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 2

    def test_tier1_create_file_upgraded(self):
        goals = [_make_goal("Create file output.txt", required_tier=1)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 2

    def test_tier2_not_downgraded(self):
        """Tier 2 must never be downgraded."""
        goals = [_make_goal("Write summary.md", required_tier=2)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 2

    def test_pure_reasoning_not_touched(self):
        """Pure reasoning goals (no file keywords) must stay at Tier 0."""
        goals = [_make_goal("Research Unity Input System and summarise findings", required_tier=0)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 0

    def test_mixed_plan_selectively_upgraded(self):
        """Only file-action goals in a mixed plan are upgraded."""
        goals = [
            _make_goal("Research the topic", required_tier=0),
            _make_goal("Write a summary to report.md", required_tier=0),
            _make_goal("Validate calculation", required_tier=1),
        ]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 0   # research — unchanged
        assert goals[1].required_tier == 2   # write .md — upgraded
        assert goals[2].required_tier == 1   # validate — unchanged

    @pytest.mark.parametrize("hint", _FILE_ACTION_HINTS)
    def test_every_hint_triggers_upgrade(self, hint: str):
        """Every keyword in _FILE_ACTION_HINTS must cause a Tier-0 upgrade."""
        desc = f"Step that does: {hint}"
        goals = [_make_goal(desc, required_tier=0)]
        TaskPlanner._enforce_file_tiers(goals)
        assert goals[0].required_tier == 2, (
            f"Hint '{hint}' did not trigger tier upgrade for description: '{desc}'"
        )


# ── FSM _is_file_action fallback tests ───────────────────────────────────────

class TestFSMFileActionHeuristic:
    def _fsm(self, tmp_path):
        return TaskFSM(tmp_path / "active_task.json", tmp_path / "state_manifest.json")

    @pytest.mark.parametrize("desc,expected", [
        ("Create a new markdown file named phase4_ready.md", True),
        ("Write a paragraph about the project goals",       True),
        ("Save file to output.txt",                         True),
        ("Research Unity Input System",                     False),
        ("Summarise findings in memory",                    False),
        ("Validate calculation",                            False),
        ("Generate report.json",                            True),
    ])
    def test_is_file_action(self, tmp_path, desc: str, expected: bool):
        fsm = self._fsm(tmp_path)
        assert fsm._is_file_action(desc) == expected, (
            f"_is_file_action('{desc}') should be {expected}"
        )


# ── _extract_filename tests ───────────────────────────────────────────────────

class TestExtractFilename:
    def _fsm(self, tmp_path):
        return TaskFSM(tmp_path / "active_task.json", tmp_path / "state_manifest.json")

    def test_quoted_filename_extracted(self, tmp_path):
        fsm = self._fsm(tmp_path)
        assert fsm._extract_filename("Write 'phase4_ready.md' to disk") == "phase4_ready.md"

    def test_bare_extension_extracted(self, tmp_path):
        fsm = self._fsm(tmp_path)
        assert fsm._extract_filename("Create a markdown file summary.md") == "summary.md"

    def test_slug_fallback(self, tmp_path):
        fsm = self._fsm(tmp_path)
        result = fsm._extract_filename("Write a paragraph about goals")
        assert result.endswith(".md")
        assert len(result) > 3  # non-empty slug

    def test_slug_fallback_empty_desc(self, tmp_path):
        fsm = self._fsm(tmp_path)
        result = fsm._extract_filename("")
        assert result == "task_output.md"

class TestTaskPlannerGeneration:
    def test_task_planner_success(self):
        from agent.brains.mock_brain import MockBrain
        brain = MockBrain()
        goals = MagicMock()
        planner = TaskPlanner(brain, goals)
        
        with patch.object(brain, "generate", return_value="""[
        {
          "id": "t1",
          "description": "Task 1",
          "required_tier": 1,
          "completion_criteria": "Done",
          "dependencies": []
        },
        {
          "id": "t2",
          "description": "Task 2",
          "required_tier": 2,
          "completion_criteria": "Done",
          "dependencies": ["t1"]
        }
        ]"""):
            plan = planner.plan_task("Test task")
            assert len(plan) == 2
            # Notice the IDs are remapped to UUIDs, so we check descriptions
            assert plan[0].description == "Task 1"
            assert plan[1].description == "Task 2"

    def test_task_planner_invalid_json(self):
        from agent.brains.mock_brain import MockBrain
        brain = MockBrain()
        goals = MagicMock()
        planner = TaskPlanner(brain, goals)
        
        with patch.object(brain, "generate", return_value="Invalid JSON"):
            with pytest.raises(ValueError, match="JSON repair failed"):
                planner.plan_task("Test task")

    def test_task_planner_commit_plan(self):
        from agent.brains.mock_brain import MockBrain
        brain = MockBrain()
        goals = MagicMock()
        planner = TaskPlanner(brain, goals)
        
        with patch.object(brain, "generate", return_value="""[
        {
          "id": "t1",
          "description": "Task 1",
          "required_tier": 1,
          "completion_criteria": "Done",
          "dependencies": []
        }
        ]"""):
            plan = planner.plan_task("Test task")
            planner.commit_plan(plan)
            # Should be called once for each goal in plan
            assert goals.register.call_count == 1
