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

"""REPL loop: ask, learn, facts, stats, project, exit."""

from __future__ import annotations

from rich.console import Console

from agent.brains.base import BaseBrain
from agent.engine.retriever import Retriever
from agent.engine.synthesizer import SkillSynthesizer
from agent.engine.validator import SkillValidator
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.goals import GoalMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.memory.semantic import SemanticMemory

from agent.commands.learn import handle_learn
from agent.commands.skills import handle_skill, handle_skills, handle_run_skill
from agent.commands.facts import handle_facts
from agent.commands.project import handle_project_cmd
from agent.commands.dataset import handle_dataset
from agent.commands.system import (
    handle_read_file,
    handle_stats,
    handle_self_model,
    handle_benchmark_reasoning,
    handle_brain_cmd,
)
from agent.commands.task import handle_task
from agent.commands.ingest import handle_ingest_paper
from agent.commands.train import handle_train_cmd
from agent.engine.chat import ChatEngine

console = Console()

HELP = """
Commands:
  ask <question>      Ask something. Answered honestly from seeded facts & project files.
  learn                Phase 0 stub — seeds facts.json.
  skill <topic>        Synthesize and validate a new local Python skill.
  facts                List everything currently in semantic memory.
  project index <dir>  Index workspace directory into Project Memory.
  project list         List indexed project files.
  stats                Show memory counts.
  self-model           Show empirical competence matrix and boot count.
  benchmark reasoning  Run ZPD reasoning calibration.
  dataset stats|build  Manage DPO reasoning datasets (Mitigations #68, #69).
  brain switch <provider> [model]  Switch active brain (gemini, groq, openai, local, mock, moa_router).
  brain list           Show registered providers and current brain.
  /clear               Clear conversational chat context.
  help                 Show this message.
  exit                 Quit.

Any unrecognised text is routed directly to the Conversational Chat Engine.
"""


def run_repl(
    semantic: SemanticMemory,
    episodic: EpisodicMemory,
    procedural: ProceduralMemory,
    project: ProjectMemory,
    goals: GoalMemory,
    brain: BaseBrain,
    embedder: EmbeddingEngine,
    self_model=None,
    pause_event=None,
) -> None:
    from agent.brains.factory import BrainManager
    from agent.config import ACTIVE_TASK_JSON, STATE_MANIFEST_JSON
    from agent.engine.state_machine import TaskFSM

    brain_manager = BrainManager(embedder=embedder, brain=brain)
    brain_name = brain_manager.brain.__class__.__name__
    _pause = pause_event
    console.print(f"[bold cyan]Agent REPL — Phase 3 (Brain: {brain_name})[/bold cyan]")
    console.print(HELP)

    fsm = TaskFSM(ACTIVE_TASK_JSON, STATE_MANIFEST_JSON)
    active_state = fsm.load_state()
    manifest_data = fsm.manifest.read_manifest()

    if active_state:
        console.print(
            f"[bold yellow]WARNING: Unfinished task detected (Goal: {active_state.goal_id}, State: {active_state.state})[/bold yellow]"
        )
        resp = console.input("Resume task? [Y/n]: ").strip().lower()
        if resp in ("y", "yes", ""):
            console.print("Resuming...")
            if (
                active_state.pending_action_hash
                and active_state.pending_action_hash not in active_state.executed_actions
            ):
                console.print(
                    f"[bold red]CRITICAL: Task crashed during action {active_state.pending_action_hash}.[/bold red]"
                )
                console.print(
                    "[yellow]The agent cannot guarantee if the last action finished executing. Transitioning to VERIFYING state.[/yellow]"
                )
                fsm.advance("VERIFYING")
        else:
            fsm.clear_task()
            console.print("Task cleared.")
    elif manifest_data and manifest_data.get("active_task_hash"):
        console.print(
            "[bold red]CRITICAL: active_task.json is missing but state_manifest.json indicates a task was running![/bold red]"
        )
        console.print(
            "[yellow]Please inspect episodic.db or backup states before proceeding to avoid state corruption.[/yellow]"
        )
        console.print("Clearing manifest to allow boot...")
        fsm.clear_task()

    while True:
        try:
            if _pause is not None:
                _pause.set()
            raw = console.input("[bold green]>[/bold green] ").strip()
            if _pause is not None:
                _pause.clear()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            break

        if not raw:
            continue

        command, _, rest = raw.partition(" ")
        command = command.lower()
        rest = rest.strip()

        if command in ("exit", "quit"):
            console.print("bye.")
            break

        try:
            dispatch_command(
                command,
                rest,
                semantic,
                episodic,
                procedural,
                project,
                goals,
                brain_manager,
                self_model=self_model,
            )
        except Exception as e:
            console.print(f"[bold red]System Error:[/bold red] {e}")


def dispatch_command(
    command: str,
    rest: str,
    semantic: SemanticMemory,
    episodic: EpisodicMemory,
    procedural: ProceduralMemory,
    project: ProjectMemory,
    goals: GoalMemory,
    brain_or_manager,
    self_model=None,
) -> None:
    from agent.brains.factory import BrainManager
    from agent.config import ACTIVE_TASK_JSON, STATE_MANIFEST_JSON
    from agent.engine.governor import PermissionGovernor
    from agent.engine.state_machine import TaskFSM
    from agent.engine.task_planner import TaskPlanner

    if isinstance(brain_or_manager, BrainManager):
        brain_manager = brain_or_manager
        brain = brain_manager.brain
    else:
        brain_manager = None
        brain = brain_or_manager

    from agent.engine.dataset_builder import DatasetBuilder
    governor = PermissionGovernor(episodic)
    retriever = Retriever(semantic, episodic, project, brain)
    validator = SkillValidator()
    dataset_builder = DatasetBuilder(episodic_mem=episodic, semantic_mem=semantic)
    synthesizer = SkillSynthesizer(
        brain, retriever, procedural, validator, project=project, governor=governor,
        episodic_memory=episodic, dataset_builder=dataset_builder,
    )
    fsm = TaskFSM(ACTIVE_TASK_JSON, STATE_MANIFEST_JSON)
    planner = TaskPlanner(brain, goals)
    chat_engine = ChatEngine(brain, episodic, semantic, self_model)

    from agent.engine.trainer import ModelTrainer
    trainer = ModelTrainer(brain_manager=brain_manager, self_model=self_model)

    KNOWN_COMMANDS = {
        "help", "ask", "learn", "skill", "skills", "run-skill", "facts",
        "read", "project", "stats", "task", "self-model", "benchmark",
        "dataset", "brain", "chat", "/clear", "ingest-paper", "train", "correct"
    }

    if command == "help":
        console.print(HELP)
    elif command == "ask":
        if not rest:
            console.print("[yellow]usage: ask <question>[/yellow]")
            return
        console.print(retriever.answer(rest))
    elif command == "learn":
        handle_learn(rest, semantic, brain, brain_manager)
    elif command == "skill":
        handle_skill(rest, synthesizer)
    elif command == "skills":
        handle_skills(rest, procedural)
    elif command == "run-skill":
        handle_run_skill(rest, procedural, validator)
    elif command == "facts":
        handle_facts(rest, semantic)
    elif command == "correct":
        from agent.commands.facts import handle_correct
        handle_correct(rest, semantic, episodic)
    elif command == "read":
        handle_read_file(rest)
    elif command == "project":
        handle_project_cmd(rest, project, brain)
    elif command == "stats":
        handle_stats(semantic, episodic, procedural, project)
    elif command == "task":
        handle_task(rest, planner, fsm, goals, brain, procedural, validator, project, governor)
    elif command == "self-model":
        handle_self_model(self_model)
    elif command == "benchmark" and rest.startswith("reasoning"):
        handle_benchmark_reasoning(rest, brain_manager, self_model)
    elif command == "dataset":
        from agent.engine.dataset_builder import DatasetBuilder
        builder = DatasetBuilder(episodic_mem=episodic, semantic_mem=semantic)
        handle_dataset(rest, builder)
    elif command == "brain":
        handle_brain_cmd(rest, brain_manager)
    elif command == "ingest-paper":
        handle_ingest_paper(rest, semantic)
    elif command == "train":
        handle_train_cmd(rest, trainer)
    elif command == "/clear" or (command == "chat" and rest == "clear"):
        chat_engine.clear_context()
    elif command == "chat" and rest == "mode off":
        import agent.cli as cli_mod
        cli_mod.CHAT_FALLBACK_ENABLED = False
        console.print("[green]Chat fallback disabled.[/green]")
    elif command == "chat" and rest == "mode on":
        import agent.cli as cli_mod
        cli_mod.CHAT_FALLBACK_ENABLED = True
        console.print("[green]Chat fallback enabled.[/green]")
    elif command == "chat":
        chat_engine.respond(rest)
    else:
        import agent.cli as cli_mod
        import difflib
        
        # Check for typos of known commands
        matches = difflib.get_close_matches(command, KNOWN_COMMANDS, n=1, cutoff=0.7)
        if matches:
            console.print(f"[yellow]Unknown command '{command}'. Did you mean '{matches[0]}'? (Use 'chat {command}' if this was meant as conversation)[/yellow]")
        elif getattr(cli_mod, "CHAT_FALLBACK_ENABLED", True):
            # Fallback routing to conversational chat
            raw_input = f"{command} {rest}".strip()
            chat_engine.respond(raw_input)
        else:
            console.print(f"[yellow]unknown command: {command}[/yellow] (try `help`)")

