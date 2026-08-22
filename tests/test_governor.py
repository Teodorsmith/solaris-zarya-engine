import pytest
from unittest.mock import patch
from agent.engine.governor import PermissionGovernor
from agent.models import Goal

class MockEpisodic:
    def __init__(self):
        self.logs = []
    def log_event(self, log):
        self.logs.append(log)

def test_governor_depth_caps():
    episodic = MockEpisodic()
    gov = PermissionGovernor(episodic)
    
    g1 = Goal(id="g1", description="1", completion_criteria="x")
    g2 = Goal(id="g2", description="2", parent_id="g1", completion_criteria="x")
    g3 = Goal(id="g3", description="3", parent_id="g2", completion_criteria="x")
    g4 = Goal(id="g4", description="4", parent_id="g3", completion_criteria="x")
    
    all_goals = [g1, g2, g3, g4]
    
    # g4 is depth 3 (g1=0, g2=1, g3=2, g4=3).
    # Autonomous hard cap is depth 2, so depth 3 should be DENIED.
    assert gov.request_permission("action", g4, all_goals, is_autonomous=True) is False
    assert "DENIED" in episodic.logs[-1].content
    
    # Supervised depth 3 should ask for [Y/n]
    with patch("builtins.input", return_value="y"):
        assert gov.request_permission("action", g4, all_goals, is_autonomous=False) is True
        assert "USER_APPROVED" in episodic.logs[-1].content

    # Supervised depth 3 should cancel on [n]
    with patch("builtins.input", return_value="n"):
        assert gov.request_permission("action", g4, all_goals, is_autonomous=False) is False
        assert "USER_DENIED" in episodic.logs[-1].content

def test_governor_fail_closed():
    episodic = MockEpisodic()
    gov = PermissionGovernor(episodic)
    
    g1 = Goal(id="g1", description="1", completion_criteria="x", required_tier=2)
    all_goals = [g1]
    
    # Test 'y'
    with patch("builtins.input", return_value="y"):
        assert gov.request_permission("action", g1, all_goals) is True
        assert "USER_APPROVED" in episodic.logs[-1].content
        
    # Test 'n'
    with patch("builtins.input", return_value="n"):
        assert gov.request_permission("action", g1, all_goals) is False
        assert "USER_DENIED" in episodic.logs[-1].content
        
    # Test empty input -> immediately denies (fail closed)
    with patch("builtins.input", return_value=""):
        assert gov.request_permission("action", g1, all_goals) is False
        assert "USER_DENIED" in episodic.logs[-1].content

    # Test trace_id presence
    assert getattr(episodic.logs[-1], "trace_id", None) is not None


def test_governor_skill_write():
    episodic = MockEpisodic()
    gov = PermissionGovernor(episodic)

    # Autonomous mode auto-denies
    assert gov.request_skill_write_permission("calc_fib", "skills/calc_fib.py", "def execute(): ...", is_autonomous=True) is False
    assert "DENIED" in episodic.logs[-1].content

    # Supervised approval
    with patch("builtins.input", return_value="y"):
        assert gov.request_skill_write_permission("calc_fib", "skills/calc_fib.py", "def execute(): ...", is_autonomous=False) is True
        assert "USER_APPROVED" in episodic.logs[-1].content
        assert "calc_fib" in episodic.logs[-1].content

    # Supervised denial with 'n'
    with patch("builtins.input", return_value="n"):
        assert gov.request_skill_write_permission("calc_fib", "skills/calc_fib.py", "def execute(): ...", is_autonomous=False) is False
        assert "USER_DENIED" in episodic.logs[-1].content

    # Supervised denial with empty enter (fail closed)
    with patch("builtins.input", return_value=""):
        assert gov.request_skill_write_permission("calc_fib", "skills/calc_fib.py", "def execute(): ...", is_autonomous=False) is False
        assert "USER_DENIED" in episodic.logs[-1].content


def test_governor_file_write():
    episodic = MockEpisodic()
    gov = PermissionGovernor(episodic)

    # Autonomous mode auto-denies
    assert gov.request_file_write_permission("test.txt", "content", is_autonomous=True) is False
    assert "DENIED" in episodic.logs[-1].content

    # Supervised approval
    with patch("builtins.input", return_value="y"):
        assert gov.request_file_write_permission("test.txt", "content", goal_description="Write test file") is True
        assert "USER_APPROVED" in episodic.logs[-1].content
        assert "test.txt" in episodic.logs[-1].content

    # Supervised denial with 'n'
    with patch("builtins.input", return_value="n"):
        assert gov.request_file_write_permission("test.txt", "content", goal_description="Write test file") is False
        assert "USER_DENIED" in episodic.logs[-1].content

    # Supervised denial with empty enter (fail closed)
    with patch("builtins.input", return_value=""):
        assert gov.request_file_write_permission("test.txt", "content", goal_description="Write test file") is False
        assert "USER_DENIED" in episodic.logs[-1].content

