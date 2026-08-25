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

"""Curriculum Planner: Decomposes broad research topics into sub-units."""

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from agent.brains.base import BaseBrain

if TYPE_CHECKING:
    from agent.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


class CurriculumPlanner:
    def __init__(self, brain: BaseBrain, semantic: "SemanticMemory | None" = None):
        self.brain = brain
        self.semantic = semantic
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

    def save_checkpoint(
        self, topic: str, units: list[str], completed_units: list[int]
    ) -> None:
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "topic": topic,
            "total_units": len(units),
            "completed_units": completed_units,
            "units_data": units,
        }
        self.checkpoint_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clear_checkpoint(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def plan_curriculum(self, topic: str) -> list[str]:
        """
        Decomposes a topic into focused sub-units.
        If prior knowledge already exists in SemanticMemory or a Markdown note
        file, generates 2-4 targeted differential gap units instead of the
        baseline 3-6 foundational units.
        """
        from agent.engine.exporter import get_topic_slug

        prior_facts: list = []
        completed_unit_titles: list[str] = []

        # 1. Query prior facts from SemanticMemory
        if self.semantic is not None:
            prior_facts = self.semantic.get_facts_by_topic(topic, limit=20)

        # 2. Check existing Markdown note for previously completed units
        slug = get_topic_slug(topic)
        note_path = Path("data/knowledge") / f"{slug}.md"
        if note_path.exists():
            try:
                note_text = note_path.read_text(encoding="utf-8")
                # Extract unit titles from "## Unit N/M: Title" headers
                completed_unit_titles = re.findall(
                    r"^## Unit \d+/\d+: (.+)$", note_text, re.MULTILINE
                )
            except Exception:
                pass

        has_prior_knowledge = bool(prior_facts or completed_unit_titles)

        if has_prior_knowledge:
            known_facts_summary = "\n".join(
                f"- {f.text}" for f in prior_facts[:10]
            )
            covered_units_summary = "\n".join(
                f"- {t}" for t in completed_unit_titles
            )
            prompt = f"""
You are an expert researcher and curriculum designer enriching an existing knowledge base.
The user wants to deepen their understanding of: "{topic}"

Known concepts already in memory:
{known_facts_summary or '(none yet)'}

Previously completed study units:
{covered_units_summary or '(none yet)'}

Your task: Identify missing subtopics, advanced edge cases, or internals that have NOT been covered above.
Generate exactly 2 to 4 targeted differential study units to fill these specific gaps.
Do NOT repeat concepts already covered.

Output ONLY a valid JSON array of strings, where each string is the title/description of a gap unit.
Example:
[
  "Advanced Edge Cases in Enum Flag Composition & Bit Masking",
  "Enum Integration with Dataclasses and JSON Serialization"
]
"""
        else:
            prompt = f"""
You are an expert researcher and curriculum designer.
The user wants to deeply research and learn about: "{topic}"

Decompose this massive topic into a logical, sequential curriculum of focused sub-units.
Generate exactly 3 to 6 sub-units to comprehensively cover the topic without hallucinating or looping.
Ensure the units flow logically (e.g., chronological, foundational to advanced).

Output ONLY a valid JSON array of strings, where each string is the title/description of a sub-unit.
Example:
[
  "Unit 1: Origins & Pre-War Timeline (1931-1939): Treaty of Versailles, rise of Axis powers, invasion of Poland.",
  "Unit 2: European Theatre & Major Battles (1939-1945): Battle of Britain, Stalingrad, Normandy landings."
]
"""

        response = self.brain.generate(prompt, temperature=0.1, repetition_penalty=1.2)

        parsed = self.brain.extract_json(response)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed

        # Fallback repair if extract_json fails
        logger.warning(
            "Curriculum generation failed strict JSON parsing. Attempting repair."
        )

        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                repaired = json.loads(match.group(0))
                if isinstance(repaired, list):
                    return [str(x) for x in repaired]
            except Exception:
                pass

        raise ValueError(
            f"Failed to parse curriculum plan for '{topic}'. Output: {response}"
        )


def build_search_query(main_topic: str, unit_title: str) -> str:
    import re

    # Strip "Unit 1:", "Unit 12 -", etc.
    cleaned_title = re.sub(
        r"^Unit\s+\d+[\s:\-–—]+", "", unit_title, flags=re.IGNORECASE
    ).strip()
    # Remove parenthetical details if too long
    cleaned_title = re.sub(r"\(.*?\)", "", cleaned_title).strip()
    return f"{main_topic} {cleaned_title}".strip()
