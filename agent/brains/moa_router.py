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

"""Mixture-of-Agents (MoA) Router -- Mitigation #70.

Routes tasks dynamically based on task-intrinsic complexity_score:
- complexity_score < 0.5  -> Base Brain (cheap, routine)
- complexity_score >= 0.5 -> LoRA Reasoning Brain (deep reasoning)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.brains.base import BaseBrain, QuotaExceededError
from agent.config import MOA_ROUTING_THRESHOLD

logger = logging.getLogger(__name__)

# Keywords / patterns indicative of high-complexity reasoning tasks
_REASONING_PATTERNS = [
    r"\b(proof|prove|theorem|deduce|deduction|axiom|contrapositive)\b",
    r"\b(modus_ponens|modus_tollens|syllogism|disjunctive)\b",
    r"\b(step-by-step|decomposition|sub-goal|prerequisite|dag)\b",
    r"\b(symbolic|srt|invariant|formal logic|truth table)\b",
    r"[\u2200\u2203\u2192\u2227\u2228\u00AC\u22A2\u22A8]",
]


class MoABrain(BaseBrain):
    """Mixture-of-Agents Brain routing between base and reasoning LoRA."""

    def __init__(
        self,
        base_brain: BaseBrain,
        lora_brain: BaseBrain,
        complexity_threshold: float = MOA_ROUTING_THRESHOLD,
    ) -> None:
        self.base_brain = base_brain
        self.lora_brain = lora_brain
        self.complexity_threshold = complexity_threshold

        # Telemetry counters
        self.total_routed = 0
        self.routed_to_lora = 0
        self.routed_to_base = 0
        self.fallbacks = 0

    def estimate_complexity(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> float:
        """Estimate task-intrinsic complexity_score in range [0.0, 1.0].

        Note: complexity_score is task-intrinsic (structure/depth), orthogonal
        to novelty_score (task-relative historical distance).
        """
        if context and "complexity_score" in context:
            try:
                return float(
                    max(0.0, min(1.0, float(context["complexity_score"])))
                )
            except (ValueError, TypeError):
                pass

        score = 0.1  # baseline routine task score

        # 1. Check reasoning and logic keywords/symbols
        prompt_lower = prompt.lower()
        matched_patterns = sum(
            1
            for pat in _REASONING_PATTERNS
            if re.search(pat, prompt_lower, re.IGNORECASE)
        )
        score += min(0.4, matched_patterns * 0.15)

        # 2. Structural length & multi-constraint heuristics
        lines = prompt.splitlines()
        if len(lines) > 10:
            score += 0.15
        elif len(lines) > 4:
            score += 0.08

        # 3. Explicit constraints / rules count
        constraint_count = len(
            re.findall(
                r"\b(must|should|require|constraint|rule)\b",
                prompt_lower,
            )
        )
        score += min(0.2, constraint_count * 0.05)

        # 4. Code & formal specification markers
        if "```" in prompt or "def " in prompt or "class " in prompt:
            score += 0.15

        return float(max(0.0, min(1.0, score)))

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Route generation to LoRA reasoning co-processor or base brain."""
        self.total_routed += 1
        explicit_complexity = kwargs.pop("complexity_score", None)

        if explicit_complexity is not None:
            try:
                complexity = float(explicit_complexity)
            except (ValueError, TypeError):
                complexity = self.estimate_complexity(prompt)
        else:
            complexity = self.estimate_complexity(prompt)

        logger.debug(
            "MoABrain: prompt complexity=%.2f (threshold=%.2f)",
            complexity,
            self.complexity_threshold,
        )

        if complexity >= self.complexity_threshold:
            self.routed_to_lora += 1
            try:
                logger.info(
                    "MoABrain: routing high-complexity (%.2f) to LoRA brain",
                    complexity,
                )
                return self.lora_brain.generate(prompt, **kwargs)
            except (QuotaExceededError, Exception) as exc:
                self.fallbacks += 1
                logger.warning(
                    "MoABrain: LoRA failed (%s). Falling back to BaseBrain.",
                    exc,
                )
                return self.base_brain.generate(prompt, **kwargs)
        else:
            self.routed_to_base += 1
            logger.info(
                "MoABrain: routing routine task (%.2f) to BaseBrain",
                complexity,
            )
            return self.base_brain.generate(prompt, **kwargs)

    def embed(self, text: str) -> list[float]:
        """Embed text using base brain embedding engine or fallback."""
        try:
            return self.base_brain.embed(text)
        except Exception:
            try:
                return self.lora_brain.embed(text)
            except Exception:
                from agent.memory.embeddings import EmbeddingEngine

                return EmbeddingEngine(force_fallback=True).embed(text)

    def get_stats(self) -> dict[str, Any]:
        """Return routing telemetry."""
        return {
            "total_routed": self.total_routed,
            "routed_to_lora": self.routed_to_lora,
            "routed_to_base": self.routed_to_base,
            "fallbacks": self.fallbacks,
            "complexity_threshold": self.complexity_threshold,
            "base_brain": self.base_brain.__class__.__name__,
            "lora_brain": self.lora_brain.__class__.__name__,
        }

    def load_checkpoint(self, checkpoint_path: Path | str) -> bool:
        """Dynamically load a LoRA checkpoint onto the lora_brain if supported."""
        checkpoint_path = Path(checkpoint_path)
        meta_path = checkpoint_path / "adapter_meta.json"
        
        if not meta_path.exists():
            logger.error(f"Cannot load checkpoint: {meta_path} does not exist.")
            return False
            
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            logger.info(f"Loading checkpoint metadata: {meta}")
        except Exception as e:
            logger.error(f"Failed to read checkpoint metadata: {e}")
            return False
            
        if hasattr(self.lora_brain, "load_adapter"):
            try:
                self.lora_brain.load_adapter(str(checkpoint_path)) # type: ignore
                logger.info(f"Successfully hot-reloaded adapter onto {self.lora_brain.__class__.__name__}")
                return True
            except Exception as e:
                logger.error(f"Failed to hot-reload adapter: {e}")
                return False
        elif hasattr(self.lora_brain, "adapter_path"):
            self.lora_brain.adapter_path = str(checkpoint_path) # type: ignore
            logger.info(f"Updated adapter_path on {self.lora_brain.__class__.__name__}. Requires re-initialization.")
            return True
        else:
            logger.info(
                f"Brain {self.lora_brain.__class__.__name__} does not support dynamic LoRA hot-reloading. "
                "Ignoring checkpoint swap."
            )
            return False
