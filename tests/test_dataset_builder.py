import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.engine.dataset_builder import DatasetBuilder

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
        
@pytest.fixture
def db_builder(temp_dir, monkeypatch):
    # monkeypatch config.DATA_DIR so we don't mess with real db
    monkeypatch.setattr("agent.engine.dataset_builder.DATA_DIR", temp_dir)
    builder = DatasetBuilder(dataset_path=temp_dir / "test_dpo.jsonl")
    yield builder
    builder.conn.close()

def test_success_validation_rejects_nonzero_exit(db_builder):
    # Chosen trajectory has non-zero exit code
    metadata = {"chosen_exit_code": 1, "rejected_exit_code": 1}
    assert not db_builder.harvest_dpo_pair("prompt", "chosen", "rejected", metadata)
    
def test_rejected_from_real_failed_attempt(db_builder):
    # Rejected didn't fail
    metadata = {"chosen_exit_code": 0, "rejected_exit_code": 0}
    assert not db_builder.harvest_dpo_pair("prompt", "chosen", "rejected", metadata)
    
    # Correct case
    metadata = {"chosen_exit_code": 0, "rejected_exit_code": 1}
    assert db_builder.harvest_dpo_pair("prompt", "chosen", "rejected", metadata)

def test_novelty_filter_blocks_duplicate(temp_dir, monkeypatch):
    monkeypatch.setattr("agent.engine.dataset_builder.DATA_DIR", temp_dir)
    
    # Need semantic mem mock
    class MockEmbedder:
        def embed(self, text):
            return [1.0, 0.0] if "prompt1" in text else [0.0, 1.0]
            
    semantic = MagicMock()
    semantic.embedder = MockEmbedder()
    
    builder = DatasetBuilder(dataset_path=temp_dir / "test_dpo.jsonl", semantic_mem=semantic)
    
    metadata = {"chosen_exit_code": 0, "rejected_exit_code": 1}
    
    # First one succeeds
    assert builder.harvest_dpo_pair("prompt1", "chosen", "rejected", metadata)
    
    # Exact duplicate should fail
    assert not builder.harvest_dpo_pair("prompt1", "chosen2", "rejected2", metadata)
    
    # Different prompt succeeds
    assert builder.harvest_dpo_pair("prompt2", "chosen", "rejected", metadata)
    
    builder.conn.close()

def test_benchmark_contamination_blocked(db_builder):
    metadata = {"chosen_exit_code": 0, "rejected_exit_code": 1, "task_id": "benchmark_1"}
    assert not db_builder.harvest_dpo_pair("prompt", "chosen", "rejected", metadata)
    
def test_user_correction_harvested_as_pair(db_builder):
    metadata = {"chosen_exit_code": 0, "rejected_exit_code": 0, "source": "user_correction"}
    assert db_builder.harvest_dpo_pair("prompt", "chosen", "rejected", metadata)
    
def test_dataset_builder_dry_run_does_not_write(temp_dir, monkeypatch):
    monkeypatch.setattr("agent.engine.dataset_builder.DATA_DIR", temp_dir)
    
    # Mock episodic log
    class MockEpisodic:
        def __init__(self):
            self.conn = MagicMock()
            row1 = {"content": json.dumps({
                "original_fail_id": 1,
                "chosen_code": "print(2)",
                "exit_code": 0,
                "task_id": "test_1"
            })}
            row2 = {"content": json.dumps({
                "prompt": "print 1",
                "code": "print(1)",
                "exit_code": 1
            })}
            
            # set up fetchall for the task_repair_resolved
            self.conn.execute.return_value.fetchall.return_value = [row1]
            # set up fetchone for the task_failure
            self.conn.execute.return_value.fetchone.return_value = row2
            
    episodic = MockEpisodic()
    builder = DatasetBuilder(dataset_path=temp_dir / "test_dpo.jsonl", episodic_mem=episodic)
    
    candidates = builder.harvest_from_episodic(dry_run=True)
    
    assert len(candidates) > 0
    assert candidates[0]["prompt"] == "print 1"
    
    # File should not exist because we did dry_run
    assert not (temp_dir / "test_dpo.jsonl").exists()
    
    builder.conn.close()
