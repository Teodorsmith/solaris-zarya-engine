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
        
    # Test empty input -> loops, side_effect returns "" then "n"
    with patch("builtins.input", side_effect=["", "n"]):
        assert gov.request_permission("action", g1, all_goals) is False
        assert "USER_DENIED" in episodic.logs[-1].content

    # Test trace_id presence
    assert getattr(episodic.logs[-1], "trace_id", None) is not None
