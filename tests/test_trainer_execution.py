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

import json
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from agent.engine.trainer import ModelTrainer
from agent.brains.moa_router import MoABrain
from agent.config import DATA_DIR

@pytest.fixture
def temp_env():
    # Setup temp dirs
    checkpoints = DATA_DIR / "checkpoints"
    datasets = DATA_DIR / "datasets"
    
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
        
    checkpoints.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    
    # Create dummy dataset
    dpo_file = datasets / "dpo_reasoning_pairs.jsonl"
    with open(dpo_file, "w") as f:
        f.write(json.dumps({"prompt": "test", "chosen": "a", "rejected": "b"}) + "\n")
        
    yield dpo_file
    
    # Cleanup
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    if dpo_file.exists():
        os.remove(dpo_file)

import sys
# Mock torch globally for tests to prevent heavy imports and GPU alloc
mock_torch = Mock()
mock_torch.cuda.is_available.return_value = False
sys.modules["torch"] = mock_torch

def test_trainer_dry_run_fallback(temp_env):
    trainer = ModelTrainer()
    
    # Force dry run
    out_dir = trainer.train_dpo(dataset_path=temp_env, dry_run=True)
    
    assert out_dir.exists()
    assert out_dir.name == "lora_v1"
    
    meta_file = out_dir / "adapter_meta.json"
    assert meta_file.exists()
    
    with open(meta_file, "r") as f:
        meta = json.load(f)
        assert meta["status"] == "dry-run"
        assert meta["version"] == "lora_v1"
        assert str(temp_env) in meta["dataset"]
        
    # Test sequential versioning
    out_dir2 = trainer.train_dpo(dataset_path=temp_env, dry_run=True)
    assert out_dir2.name == "lora_v2"

def test_moa_router_load_checkpoint(temp_env):
    # Setup mock lora_brain
    mock_base = Mock()
    mock_lora = Mock()
    # Give it load_adapter method
    mock_lora.load_adapter = Mock()
    
    router = MoABrain(base_brain=mock_base, lora_brain=mock_lora)
    
    # Generate a mock checkpoint
    trainer = ModelTrainer()
    out_dir = trainer.train_dpo(dataset_path=temp_env, dry_run=True)
    
    # Load checkpoint
    success = router.load_checkpoint(out_dir)
    assert success is True
    
    # Verify load_adapter was called with the checkpoint path
    mock_lora.load_adapter.assert_called_once_with(str(out_dir))
    
def test_moa_router_load_checkpoint_no_support(temp_env):
    # Setup mock lora_brain WITHOUT load_adapter or adapter_path
    mock_base = Mock()
    mock_lora = Mock(spec=[]) # No attributes
    
    router = MoABrain(base_brain=mock_base, lora_brain=mock_lora)
    
    trainer = ModelTrainer()
    out_dir = trainer.train_dpo(dataset_path=temp_env, dry_run=True)
    
    # Load checkpoint should return False (no support)
    success = router.load_checkpoint(out_dir)
    assert success is False


def test_dataset_validation_fails(temp_env):
    trainer = ModelTrainer()
    
    # Write bad dataset
    bad_file = temp_env.parent / "bad.jsonl"
    with open(bad_file, "w") as f:
        # First row good, second row bad
        f.write(json.dumps({"prompt": "test", "chosen": "a", "rejected": "b"}) + "\n")
        f.write(json.dumps({"prompt": "test", "chosen": ""}) + "\n")
        
    with pytest.raises(ValueError, match="missing or empty required DPO fields at line 2"):
        trainer.train_dpo(dataset_path=bad_file, dry_run=True)

