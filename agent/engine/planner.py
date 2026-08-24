# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Curriculum Planner: Decomposes broad research topics into sub-units."""

import logging
import json
from pathlib import Path
from agent.brains.base import BaseBrain

logger = logging.getLogger(__name__)

class CurriculumPlanner:
    def __init__(self, brain: BaseBrain):
        self.brain = brain
        self.checkpoint_file = Path("data/active_curriculum.json")

    def has_checkpoint(self, topic: str) -> bool:
        if not self.checkpoint_file.exists():
            return False
        try:
            data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
            return data.get("topic") == topic
        except Exception:
            return False

    def load_checkpoint(self) -> dict | None:
        if not self.checkpoint_file.exists():
            return None
        try:
            return json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_checkpoint(self, topic: str, units: list[str], completed_units: list[int]) -> None:
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "topic": topic,
            "total_units": len(units),
            "completed_units": completed_units,
            "units_data": units
        }
        self.checkpoint_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clear_checkpoint(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def plan_curriculum(self, topic: str) -> list[str]:
        """
        Decomposes a massive topic into as many focused sub-units as needed.
        Returns a list of strings representing the units.
        """
        prompt = f"""
You are an expert researcher and curriculum designer.
The user wants to deeply research and learn about: "{topic}"

Decompose this massive topic into a logical, sequential curriculum of focused sub-units.
Generate as many sub-units as necessary to comprehensively cover the topic.
Ensure the units flow logically (e.g., chronological, foundational to advanced).

Output ONLY a valid JSON array of strings, where each string is the title/description of a sub-unit.
Example:
[
  "Unit 1: Origins & Pre-War Timeline (1931-1939): Treaty of Versailles, rise of Axis powers, invasion of Poland.",
  "Unit 2: European Theatre & Major Battles (1939-1945): Battle of Britain, Stalingrad, Normandy landings."
]
"""
        response = self.brain.generate(prompt)
        
        parsed = self.brain.extract_json(response)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
            
        # Fallback repair if extract_json fails
        logger.warning("Curriculum generation failed strict JSON parsing. Attempting repair.")
        import re
        import json
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                repaired = json.loads(match.group(0))
                if isinstance(repaired, list):
                    return [str(x) for x in repaired]
            except Exception:
                pass
                
        raise ValueError(f"Failed to parse curriculum plan for '{topic}'. Output: {response}")
