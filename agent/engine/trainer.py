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

"""Experience -> Training Pipeline & Promotion Coordinator -- Mitigation #69.

Coordinates model candidate evaluation across the 6-category ZPD suite.
Enforces the Promotion Gate:
- Candidate must improve ZPD ceilings on >= 3 of 6 categories.
- Candidate must have zero regressions on all other categories.
- Automatic rollback to baseline weights if promotion fails.
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.brains.base import BaseBrain
from agent.brains.factory import BrainManager
from agent.config import ZPD_CATEGORIES, DATA_DIR
from tests.reasoning_suite.runner import ZPDRunner

logger = logging.getLogger(__name__)


class PromotionError(Exception):
    """Raised when a candidate fails the benchmark promotion gate."""


class ModelTrainer:
    """Coordinates fine-tuning evaluation and ZPD benchmark promotion."""

    def __init__(
        self,
        brain_manager: BrainManager | None = None,
        self_model=None,
    ) -> None:
        self.brain_manager = brain_manager
        self.self_model = self_model

    def evaluate_candidate(
        self,
        candidate_brain: BaseBrain,
        baseline_brain: BaseBrain,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate candidate against baseline on the 6-category ZPD benchmark.

        Returns (promoted, report_dict).
        Promotion requires:
        1. Improvement on >= 3 of 6 reasoning categories.
        2. Zero regressions (cand ceiling >= base ceiling for all categories).
        """
        cand_mgr = SimpleNamespace(
            brain=candidate_brain, fallback=lambda: None
        )
        base_mgr = SimpleNamespace(
            brain=baseline_brain, fallback=lambda: None
        )

        runner_cand = ZPDRunner(cand_mgr)
        runner_base = ZPDRunner(base_mgr)

        cand_ceilings = runner_cand.run_all(dry_run=True)
        base_ceilings = runner_base.run_all(dry_run=True)

        improved_categories: list[str] = []
        regressed_categories: list[str] = []
        maintained_categories: list[str] = []

        for cat in ZPD_CATEGORIES:
            c_cand = cand_ceilings.get(cat, 0)
            c_base = base_ceilings.get(cat, 0)

            if c_cand > c_base:
                improved_categories.append(cat)
            elif c_cand < c_base:
                regressed_categories.append(cat)
            else:
                maintained_categories.append(cat)

        num_improved = len(improved_categories)
        num_regressed = len(regressed_categories)

        # Gate criterion: >= 3 improvements and 0 regressions
        promoted = (num_improved >= 3) and (num_regressed == 0)

        if promoted:
            reason = (
                f"Candidate improved {num_improved}/6 categories "
                f"with {num_regressed} regressions."
            )
        else:
            reason = (
                f"Candidate failed promotion gate ({num_improved}/6 "
                f"improved, {num_regressed} regressions)."
            )

        report = {
            "promoted": promoted,
            "improved_count": num_improved,
            "regressed_count": num_regressed,
            "maintained_count": len(maintained_categories),
            "improved_categories": improved_categories,
            "regressed_categories": regressed_categories,
            "maintained_categories": maintained_categories,
            "candidate_ceilings": cand_ceilings,
            "baseline_ceilings": base_ceilings,
            "promotion_gate_passed": promoted,
            "reason": reason,
        }

        logger.info(
            "ModelTrainer: evaluation result=%s (improved=%d, regressed=%d)",
            promoted,
            num_improved,
            num_regressed,
        )
        return promoted, report

    def train_and_promote(
        self,
        candidate_brain: BaseBrain,
        baseline_brain: BaseBrain,
        checkpoint_name: str = "candidate_lora_adapter",
    ) -> dict[str, Any]:
        """Execute candidate promotion and trigger rollback on failure."""
        promoted, report = self.evaluate_candidate(
            candidate_brain, baseline_brain
        )

        if promoted:
            logger.info(
                "ModelTrainer: PROMOTION PASSED! Checkpoint '%s'",
                checkpoint_name,
            )
            report["active_model"] = checkpoint_name
            report["status"] = "promoted"
            if self.self_model:
                try:
                    self.self_model.update_zpd_ceilings(
                        report["candidate_ceilings"]
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to update self-model ceilings: %s", exc
                    )
        else:
            logger.warning(
                "ModelTrainer: PROMOTION FAILED. Rollback. Reason: %s",
                report["reason"],
            )
            report["active_model"] = "baseline"
            report["status"] = "rolled_back"

        return report

    def _mock_training_run(self, output_dir: Path, dataset_path: Path):
        """Simulate a training run for fallback environments (CPU-only/CI)."""
        logger.info(f"Executing dry-run fallback training. Mocking artifacts in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write mock adapter meta
        meta = {
            "version": output_dir.name,
            "base_model": "mock-base-model",
            "status": "dry-run",
            "dataset": str(dataset_path)
        }
        with open(output_dir / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def train_dpo(
        self, 
        dataset_path: Path | str | None = None, 
        epochs: int = 1, 
        batch_size: int = 2, 
        dry_run: bool = False,
        model_id: str | None = None
    ) -> Path:
        """Execute QLoRA Fine-Tuning using trl.DPOTrainer."""
        dataset_path = Path(dataset_path) if dataset_path else DATA_DIR / "datasets" / "dpo_reasoning_pairs.jsonl"
        
        # Validate dataset (every row)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        valid_rows = 0
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not all(k in record and record[k] for k in ("prompt", "chosen", "rejected")):
                        raise ValueError(f"Dataset missing or empty required DPO fields at line {line_no}")
                    valid_rows += 1
                except json.JSONDecodeError:
                    raise ValueError(f"Dataset is not valid JSONL at line {line_no}")
        
        if valid_rows == 0:
            raise ValueError("Dataset is empty.")
            
        # Determine next checkpoint version
        checkpoint_dir = DATA_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [d.name for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("lora_v")]
        next_version = len(existing_versions) + 1
        output_dir = checkpoint_dir / f"lora_v{next_version}"
        
        has_torch = False
        try:
            import torch
            has_torch = True
        except ImportError:
            pass
            
        if dry_run or not has_torch or not torch.cuda.is_available():
            self._mock_training_run(output_dir, dataset_path)
            return output_dir
            
        logger.info(f"Starting QLoRA DPO training. Output: {output_dir}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig
        from trl import DPOTrainer
        from datasets import load_dataset
        from agent.config import TRAINING_BASE_MODEL_ID, TRAINING_MAX_SEQ_LENGTH, TRAINING_MAX_PROMPT_LENGTH
        
        base_model_id = model_id or TRAINING_BASE_MODEL_ID
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=bnb_config, device_map="auto"
            )
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}. WARNING: Falling back to float16 LoRA. This requires a CUDA GPU.")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.float16, device_map="auto"
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            lora_dropout=0.05, 
            task_type="CAUSAL_LM"
        )
        
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_steps=100,
            remove_unused_columns=False,
        )
        
        trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=0.1,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
            max_length=TRAINING_MAX_SEQ_LENGTH,
            max_prompt_length=TRAINING_MAX_PROMPT_LENGTH,
        )
        
        trainer.train()
        
        trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        meta = {
            "version": output_dir.name,
            "base_model": base_model_id,
            "status": "trained",
            "dataset": str(dataset_path),
            "rows": valid_rows
        }
        with open(output_dir / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        return output_dir
