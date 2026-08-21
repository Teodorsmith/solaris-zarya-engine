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
        brain: BaseBrain
    ):
        self.semantic = semantic
        self.episodic = episodic
        self.project = project
        self.brain = brain

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        # 1. Retrieve Semantic Facts
        facts = self.semantic.search(query, top_k=top_k)
        score = self.semantic.top_score(query)
        
        # 2. Retrieve Project Files
        semantic_files = self.project.search(query, top_k=5)
        
        # Exact path matches
        path_matches = []
        tokens = [t.strip('?.,;:"\'`') for t in query.split()]
        path_tokens = [t for t in tokens if "/" in t or t.endswith(".py") or t.endswith(".md") or t.endswith(".txt")]
        
        if path_tokens:
            rows = self.project.conn.execute("SELECT * FROM project_files").fetchall()
            for row in rows:
                p = row["path"]
                for t in path_tokens:
                    if p == t or p.endswith("/" + t) or p.endswith("\\" + t):
                        path_matches.append(
                            ProjectFile(
                                id=row["id"], project_id=row["project_id"], path=row["path"], 
                                sha256_hash=row["sha256_hash"], summary=row["summary"], 
                                created_at=row["created_at"], updated_at=row["updated_at"]
                            )
                        )
                        break

        project_files = merge_project_results(path_matches, semantic_files, k=5)
        
        # We consider a result 'confident' or 'tentative' if EITHER the semantic score 
        # is high enough, OR we found relevant project files (since project files
        # don't share the exact same confidence metric calibration yet).
        # For now, we lean on semantic score for the primary gate, but will pass
        # both to the LLM.
        
        # If we have project files but semantic score is low, let's bump it to tentative
        # to let the LLM look at the files.
        has_files = len(project_files) > 0
        
        if score >= CONFIDENT_THRESHOLD:
            tier = "confident"
        elif score >= TENTATIVE_THRESHOLD or has_files:
            tier = "tentative"
        else:
            tier = "refused"
            
        return RetrievalResult(facts=facts, project_files=project_files, score=score, tier=tier)

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
            self.episodic.log_event(EpisodicLog(kind="refusal", content=f"Q: {query}", outcome="neutral"))
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
                EpisodicLog(kind="answer", content=f"Q: {query} -> [MockBrain Answer]", outcome="success")
            )
            return text

        # Phase 1: Real Generation
        grounded_prompt = self.format_grounded_prompt(query, result)
        
        try:
            text = self.brain.generate(grounded_prompt).strip()
            self.episodic.log_event(
                EpisodicLog(kind="answer", content=f"Q: {query} -> {text[:100]}...", outcome="success")
            )
            return text
        except Exception as e:
            text = f"Failed to generate answer from API: {str(e)}"
            self.episodic.log_event(
                EpisodicLog(kind="refusal", content=f"Q: {query} -> Error", outcome="failure")
            )
            return text
