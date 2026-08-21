"""Entry point: python -m agent.main [--demo] [--reseed]"""
from __future__ import annotations

import argparse

from agent.cli import run_repl
from agent.config import EPISODIC_DB, PROCEDURAL_DB, SEMANTIC_DB, ensure_dirs
from agent.engine.retriever import Retriever
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.seeder import seed_knowledge
from agent.memory.semantic import SemanticMemory


def build_stores() -> tuple[SemanticMemory, EpisodicMemory, ProceduralMemory]:
    ensure_dirs()
    embedder = EmbeddingEngine()
    semantic = SemanticMemory(SEMANTIC_DB, embedder)
    episodic = EpisodicMemory(EPISODIC_DB)
    procedural = ProceduralMemory(PROCEDURAL_DB)
    return semantic, episodic, procedural


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 agent - memory core + honest retrieval")
    parser.add_argument("--demo", action="store_true", help="run a scripted demo instead of the REPL")
    parser.add_argument("--reseed", action="store_true", help="force-reload seed_data/facts.json")
    args = parser.parse_args()

    semantic, episodic, procedural = build_stores()

    if args.reseed:
        inserted = seed_knowledge(semantic, force=True)
        print(f"Reseeded: {inserted} facts inserted.")
    elif not semantic.count():
        inserted = seed_knowledge(semantic, force=False)
        print(f"First boot: seeded {inserted} facts.")

    if args.demo:
        _run_demo(semantic, episodic)
    else:
        run_repl(semantic, episodic, procedural)


def _run_demo(semantic: SemanticMemory, episodic: EpisodicMemory) -> None:
    retriever = Retriever(semantic, episodic)
    demo_questions = [
        "how do I check the state of my working directory in git",
        "what does a github pull request do",
        "how do I defeat the ender dragon",  # deliberately out of scope -> should refuse
    ]
    for q in demo_questions:
        print(f"\n> ask {q}")
        print(retriever.answer(q))


if __name__ == "__main__":
    main()
