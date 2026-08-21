"""Entry point: python -m agent.main [--demo] [--reseed]"""
from __future__ import annotations

import argparse

from agent.cli import run_repl
from agent.config import EPISODIC_DB, PROCEDURAL_DB, SEMANTIC_DB, PROJECTS_DB, ensure_dirs, load_env
from agent.brains.factory import get_brain
from agent.engine.retriever import Retriever
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.memory.seeder import seed_knowledge
from agent.memory.semantic import SemanticMemory


def build_stores() -> tuple[SemanticMemory, EpisodicMemory, ProceduralMemory, ProjectMemory, EmbeddingEngine]:
    ensure_dirs()
    embedder = EmbeddingEngine()
    semantic = SemanticMemory(SEMANTIC_DB, embedder)
    episodic = EpisodicMemory(EPISODIC_DB)
    procedural = ProceduralMemory(PROCEDURAL_DB)
    project = ProjectMemory(PROJECTS_DB, embedder)
    return semantic, episodic, procedural, project, embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 agent - real brain + project memory")
    parser.add_argument("--demo", action="store_true", help="run a scripted demo instead of the REPL")
    parser.add_argument("--reseed", action="store_true", help="force-reload seed_data/facts.json")
    args, unknown = parser.parse_known_args()

    # Phase 1: load environment variables and instantiate the brain
    load_env()
    
    semantic, episodic, procedural, project, embedder = build_stores()
    brain = get_brain(embedder)

    if args.reseed:
        inserted = seed_knowledge(semantic, force=True)
        print(f"Reseeded: {inserted} facts inserted.")
    elif not semantic.count():
        inserted = seed_knowledge(semantic, force=False)
        print(f"First boot: seeded {inserted} facts.")

    from agent.cli import dispatch_command, run_repl
    
    # We need to get the unknown args to pass to dispatch_command
    # e.g., skill "Calculate..."
    if unknown:
        command = unknown[0]
        rest = " ".join(unknown[1:])
        dispatch_command(command, rest, semantic, episodic, procedural, project, brain)
    elif args.demo:
        _run_demo(semantic, episodic, project, brain)
    else:
        run_repl(semantic, episodic, procedural, project, brain, embedder)


def _run_demo(semantic: SemanticMemory, episodic: EpisodicMemory, project: ProjectMemory, brain) -> None:
    retriever = Retriever(semantic, episodic, project, brain)
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
