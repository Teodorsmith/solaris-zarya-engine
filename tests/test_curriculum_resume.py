import pytest
import json
from pathlib import Path
from agent.engine.planner import CurriculumPlanner
from agent.brains.mock_brain import MockBrain

def test_curriculum_checkpointing_and_resume(tmp_path):
    brain = MockBrain()
    planner = CurriculumPlanner(brain)
    
    # override path to temp
    planner.checkpoint_file = tmp_path / "active_curriculum.json"
    
    topic = "Test Topic"
    units = ["Unit 1", "Unit 2", "Unit 3", "Unit 4"]
    completed_units = [1, 2]
    
    assert planner.has_checkpoint(topic) is False
    
    planner.save_checkpoint(topic, units, completed_units)
    
    assert planner.has_checkpoint(topic) is True
    assert planner.has_checkpoint("Different Topic") is False
    
    ckpt = planner.load_checkpoint()
    assert ckpt is not None
    assert ckpt["topic"] == topic
    assert ckpt["units_data"] == units
    assert ckpt["completed_units"] == completed_units
    assert ckpt["total_units"] == 4
    
    planner.clear_checkpoint()
    assert planner.has_checkpoint(topic) is False
    assert planner.load_checkpoint() is None
