# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Hybrid retrieval + confidence gating + closed-world answer construction.

Phase 1 extension: integrates ProjectMemory and uses a real Brain to
generate the final grounded answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.brains.base import BaseBrain
from agent.brains.mock_brain import MockBrain
from agent.config import CONFIDENT_THRESHOLD, TENTATIVE_THRESHOLD
from agent.memory.episodic import EpisodicMemory
from agent.memory.project import ProjectMemory
from agent.memory.semantic import SemanticMemory
from agent.models import EpisodicLog, Fact, ProjectFile


def merge_project_results(path_matches, semantic_matches, k=5):
    merged = {}
    for p in path_matches:
        merged[p.path] = p
    for p in semantic_matches:
        if p.path not in merged:
            merged[p.path] = p
    return list(merged.values())[:k]


@dataclass
class RetrievalResult:
    facts: list[Fact]
    project_files: list[ProjectFile]
    score: float
    tier: str  # "confident" | "tentative" | "refused"


class Retriever:
    def __init__(
        self,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        project: ProjectMemory,
        brain: BaseBrain,
    ):
        self.semantic = semantic
        self.episodic = episodic
        self.project = project
        self.brain = brain

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        # 1. Retrieve Semantic Facts
        facts = self.semantic.search(query, top_k=top_k)
        semantic_score = self.semantic.top_score(query)

        # 2. Retrieve Project Files (filtered by TENTATIVE_THRESHOLD)
        semantic_files = self.project.search(
            query, top_k=5, min_score=TENTATIVE_THRESHOLD
        )
        project_score = self.project.top_score(query)

        # 3. Exact path matches
        path_matches = []
        tokens = [t.strip("?.,;:\"'`") for t in query.split()]
        path_tokens = [
            t
            for t in tokens
            if "/" in t
            or "\\" in t
            or any(
                t.endswith(ext)
                for ext in (
                    ".py",
                    ".md",
                    ".txt",
                    ".cs",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".bat",
                    ".sh",
                )
            )
        ]

        if path_tokens:
            rows = self.project.conn.execute("SELECT * FROM project_files").fetchall()
            for row in rows:
                p = row["path"]
                for t in path_tokens:
                    if (
                        p == t
                        or p.endswith("/" + t)
                        or p.endswith("\\" + t)
                        or t.endswith(p)
                    ):
                        path_matches.append(
                            ProjectFile(
                                id=row["id"],
                                project_id=row["project_id"],
                                path=row["path"],
                                sha256_hash=row["sha256_hash"],
                                summary=row["summary"],
                                created_at=row["created_at"],
                                updated_at=row["updated_at"],
                            )
                        )
                        break

        project_files = merge_project_results(path_matches, semantic_files, k=5)

        # Determine overall best score across semantic and project modalities
        has_path_match = len(path_matches) > 0
        effective_score = max(semantic_score, project_score if semantic_files else 0.0)
        if has_path_match:
            effective_score = max(effective_score, 1.0)

        if effective_score >= CONFIDENT_THRESHOLD:
            tier = "confident"
        elif effective_score >= TENTATIVE_THRESHOLD:
            tier = "tentative"
        else:
            tier = "refused"

        return RetrievalResult(
            facts=facts, project_files=project_files, score=effective_score, tier=tier
        )

    def format_grounded_prompt(self, query: str, result: RetrievalResult) -> str:
        prompt = ""

        if result.facts:
            prompt += "SEMANTIC FACTS\n"
            for f in result.facts:
                prompt += f"- {f.text} (source: {f.source_type})\n"
            prompt += "\n"

        if result.project_files:
            prompt += "PROJECT FILES\n"
            for pf in result.project_files:
                prompt += f"- {pf.path} — {pf.summary}\n"
            prompt += "\n"

        prompt += (
            "RULES\n"
            "Answer ONLY using the above.\n"
            "Cite file paths when you use project files.\n"
            "If the exact detail is not present, say so.\n\n"
            f"Question: {query}"
        )
        return prompt

    def answer(self, query: str) -> str:
        """The closed-world gate."""
        result = self.retrieve(query)

        if result.tier == "refused":
            text = "I haven't learned about that yet, and no project files matched. Try `learn` to seed the knowledge base."
            self.episodic.log_event(
                EpisodicLog(kind="refusal", content=f"Q: {query}", outcome="neutral")
            )
            return text

        # If using MockBrain (offline mode), skip generation and use the Phase 0 fallback
        if isinstance(self.brain, MockBrain):
            top = result.facts[0] if result.facts else None
            if not top:
                text = f"Found {len(result.project_files)} related project files, but MockBrain cannot generate an answer."
            elif result.tier == "tentative":
                text = (
                    f"I have partial information (confidence {result.score:.2f}): {top.text}\n"
                    f"Take this with some caution — it didn't clear the confident threshold."
                )
            else:
                text = top.text

            self.episodic.log_event(
                EpisodicLog(
                    kind="answer",
                    content=f"Q: {query} -> [MockBrain Answer]",
                    outcome="success",
                )
            )
            return text

        # Phase 1: Real Generation
        grounded_prompt = self.format_grounded_prompt(query, result)

        try:
            text = self.brain.generate(grounded_prompt).strip()
            self.episodic.log_event(
                EpisodicLog(
                    kind="answer",
                    content=f"Q: {query} -> {text[:100]}...",
                    outcome="success",
                )
            )
            return text
        except Exception as e:
            text = f"Failed to generate answer from API: {e!s}"
            self.episodic.log_event(
                EpisodicLog(
                    kind="refusal", content=f"Q: {query} -> Error", outcome="failure"
                )
            )
            return text
