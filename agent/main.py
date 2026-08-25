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

"""Entry point: python -m agent.main [--demo] [--reseed] [--no-daemon]"""

from __future__ import annotations

import argparse
import threading

from agent.brains.factory import get_brain
from agent.cli import run_repl
from agent.config import (
    EPISODIC_DB,
    GOALS_DB,
    PROCEDURAL_DB,
    PROJECTS_DB,
    SELF_MODEL_BAK_JSON,
    SELF_MODEL_JSON,
    SEMANTIC_DB,
    STATE_MANIFEST_JSON,
    ensure_dirs,
    load_env,
)
from agent.engine.retriever import Retriever
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.goals import GoalMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.memory.seeder import seed_knowledge
from agent.memory.self_model import SelfModel
from agent.memory.semantic import SemanticMemory
from agent.memory.state_manifest import StateManifest


def build_stores():
    ensure_dirs()
    embedder = EmbeddingEngine()
    semantic = SemanticMemory(SEMANTIC_DB, embedder)
    episodic = EpisodicMemory(EPISODIC_DB)
    procedural = ProceduralMemory(PROCEDURAL_DB)
    project = ProjectMemory(PROJECTS_DB, embedder)
    goals = GoalMemory(GOALS_DB)
    return semantic, episodic, procedural, project, goals, embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Agent REPL")
    parser.add_argument(
        "--demo", action="store_true", help="run a scripted demo instead of the REPL"
    )
    parser.add_argument(
        "--reseed", action="store_true", help="force-reload seed_data/facts.json"
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="disable the Heartbeat background daemon",
    )
    args, unknown = parser.parse_known_args()

    load_env()

    semantic, episodic, procedural, project, goals, embedder = build_stores()
    brain = get_brain(embedder)

    # ------------------------------------------------------------------ #
    # Phase 4A: Self-Model boot sequence (Mitigations #40, #52)           #
    # ------------------------------------------------------------------ #
    manifest = StateManifest(STATE_MANIFEST_JSON)
    self_model = SelfModel(SELF_MODEL_JSON, SELF_MODEL_BAK_JSON, manifest, episodic)
    self_model.load()  # tamper detection + 3-state rollback
    self_model.increment_boot_count()

    # ------------------------------------------------------------------ #
    # Phase 4A: Heartbeat Daemon (Mitigations #41, #46)                   #
    # ------------------------------------------------------------------ #
    pause_event = threading.Event()
    heartbeat = None
    if not args.no_daemon:
        from agent.engine.heartbeat import HeartbeatDaemon

        heartbeat = HeartbeatDaemon(self_model=self_model, pause_event=pause_event)
        heartbeat.start()

    if args.reseed:
        inserted = seed_knowledge(semantic, force=True)
        print(f"Reseeded: {inserted} facts inserted.")
    elif not semantic.count():
        inserted = seed_knowledge(semantic, force=False)
        print(f"First boot: seeded {inserted} facts.")

    from agent.cli import dispatch_command

    if unknown:
        command = unknown[0]
        rest = " ".join(unknown[1:])
        dispatch_command(
            command,
            rest,
            semantic,
            episodic,
            procedural,
            project,
            goals,
            brain,
            self_model=self_model,
        )
    elif args.demo:
        _run_demo(semantic, episodic, project, brain)
    else:
        run_repl(
            semantic,
            episodic,
            procedural,
            project,
            goals,
            brain,
            embedder,
            self_model=self_model,
            pause_event=pause_event,
        )


def _run_demo(semantic, episodic, project, brain) -> None:
    retriever = Retriever(semantic, episodic, project, brain)
    demo_questions = [
        "how do I check the state of my working directory in git",
        "what does a github pull request do",
        "how do I defeat the ender dragon",
    ]
    for q in demo_questions:
        print(f"\n> ask {q}")
        print(retriever.answer(q))


if __name__ == "__main__":
    main()
