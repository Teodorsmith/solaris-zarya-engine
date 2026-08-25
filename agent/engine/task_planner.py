# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Task Planner: Decomposes tasks into Goal DAGs."""

import logging

from agent.brains.base import BaseBrain
from agent.memory.goals import GoalMemory
from agent.models import Goal

logger = logging.getLogger(__name__)

# Keywords that unambiguously signal a filesystem write operation.
# Any goal whose description matches one of these is deterministically
# upgraded to Tier 2, overriding whatever the LLM returned.
_FILE_ACTION_HINTS: tuple[str, ...] = (
    "create a new",
    "create file",
    "write a",
    "write file",
    "write to file",
    "write the",
    "save to",
    "save file",
    "save a",
    "output to file",
    "generate file",
    "produce file",
    "store to",
    "write output",
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".rst",
    ".html",
    ".csv",
)


class TaskPlanner:
    def __init__(
        self,
        brain: BaseBrain,
        goal_memory: GoalMemory,
        episodic_memory=None,
        embedder=None,
    ):
        self.brain = brain
        self.goal_memory = goal_memory
        self.episodic_memory = episodic_memory
        self.embedder = embedder

    def _parse_plan_json(self, raw_text: str) -> list[dict]:
        import json

        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>", 1)[1]

        text = raw_text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 1:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start_idx = text.find("[")
        if start_idx != -1:
            text = text[start_idx:]

        text = text.rstrip()
        if not text.endswith("]"):
            text = text.removesuffix(",")
            text += "\n]"

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON repair failed: {e}")

    def plan_task(self, prompt: str, is_autonomous: bool = False) -> list[Goal]:
        max_depth = 2 if is_autonomous else 4

        system_prompt = f"""
You are a Task Planner. The user wants to: {prompt}

Decompose this into a Directed Acyclic Graph (DAG) of sub-goals.
Maximum depth allowed: {max_depth}.

For each goal, assign a required_tier using EXACTLY these rules — do not guess:

  Tier 0 — Pure reasoning only (no external side-effects):
    Read, search memory, summarise, outline, reason, draft text IN MEMORY.
    Examples: "Research X", "Summarise Y", "Determine Z", "Plan steps".

  Tier 1 — Sandboxed computational verification (no persistent output):
    Run sandboxed code or math checks that produce no lasting files.
    Examples: "Validate calculation", "Run unit test in sandbox".

  Tier 2 — Filesystem or external mutations (REQUIRED for any file operation):
    Create, write, modify, delete, move, or rename ANY file on disk.
    THIS INCLUDES writing .md, .txt, .py, .json, or any other extension.
    Examples: "Write summary.md", "Create phase4_ready.md",
              "Save result to output.txt", "Generate report.md".
    If a step creates or writes a file, required_tier MUST be 2. No exceptions.

Output ONLY a JSON list of goals matching this schema:
[
  {{
    "id": "goal_1",
    "description": "Short description of the goal",
    "parent_id": null,
    "dependencies": [],
    "completion_criteria": "How to know this is done",
    "required_tier": 0
  }}
]
Use internal string IDs (like "goal_1") to set up dependencies.
"""
        novelty_score = 1.0
        if self.episodic_memory and self.embedder:
            try:
                rows = self.episodic_memory.conn.execute(
                    "SELECT content FROM episodic_log WHERE kind='query'"
                ).fetchall()
                if rows:
                    curr_emb = self.embedder.embed(prompt)
                    past_embs = self.embedder.embed_batch([r["content"] for r in rows])
                    max_sim = max(
                        self.embedder.similarity(curr_emb, e) for e in past_embs
                    )
                    novelty_score = max(0.0, 1.0 - max_sim)
            except Exception as e:
                logger.warning(f"Failed to compute novelty score: {e}")

        logger.info(f"Task novelty score: {novelty_score:.2f}")

        parsed = None
        response = ""

        for attempt in range(2):
            if attempt == 0:
                p = system_prompt
                if novelty_score > 0.8:
                    p += "\n\nNOVELTY ALERT (score > 0.8): First generate 3-5 distinct competing hypotheses for how to decompose this task. Evaluate their edge cases, then output ONLY the final JSON array for the best hypothesis."
            else:
                p = f"Your previous plan output was invalid or truncated JSON. Output ONLY a valid, complete JSON array with max 3 concise steps for goal: {prompt}"

            if attempt == 0 and novelty_score > 0.8 and self.embedder:
                from agent.engine.critic import CriticSession

                with CriticSession(self.brain, self.brain, self.embedder) as session:
                    res = session.solve(p)
                    response = res.answer
            else:
                response = self.brain.generate(p)

            try:
                parsed = self._parse_plan_json(response)
                if not isinstance(parsed, list):
                    raise ValueError("Parsed JSON is not a list")
                break
            except Exception as e:
                if attempt == 1:
                    raise ValueError(
                        f"Failed to generate valid plan JSON after retry. Output was: {response}. Error: {e}"
                    )

        goals = []
        # Convert internal string IDs to actual UUIDs
        id_map = {}
        from uuid import uuid4

        plan_task_id = str(uuid4())

        for p in parsed:
            new_id = str(uuid4())
            id_map[p["id"]] = new_id

        for p in parsed:
            parent = id_map.get(p.get("parent_id")) if p.get("parent_id") else None
            deps = [id_map.get(d) for d in p.get("dependencies", []) if d in id_map]

            goal = Goal(
                id=id_map[p["id"]],
                task_id=plan_task_id,
                description=p["description"],
                parent_id=parent,
                dependencies=deps,
                completion_criteria=p["completion_criteria"],
                required_tier=p["required_tier"],
                status="PENDING",
            )
            goals.append(goal)

        # Deterministic post-processing: upgrade any file-action goal to Tier 2
        # regardless of what the LLM returned.  This is the safety net that
        # prevents a model from accidentally marking a file write as Tier 0.
        self._enforce_file_tiers(goals)

        return goals

    @staticmethod
    def _enforce_file_tiers(goals: list[Goal]) -> None:
        """Deterministically upgrade goals that describe file operations to Tier 2.

        Mutates *goals* in-place.  Called after LLM plan parsing so the
        model cannot accidentally classify a file write as Tier 0 or 1.
        """
        for goal in goals:
            desc_lower = goal.description.lower()
            if any(hint in desc_lower for hint in _FILE_ACTION_HINTS):
                if goal.required_tier < 2:
                    logger.info(
                        "Tier override: '%s' upgraded from Tier %d to Tier 2 (file-action keyword match).",
                        goal.description,
                        goal.required_tier,
                    )
                    goal.required_tier = 2

    def commit_plan(self, goals: list[Goal]) -> None:
        """Saves the approved plan to the goals database."""
        for g in goals:
            self.goal_memory.register(g)
