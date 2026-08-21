"""REPL loop: ask, learn, facts, stats, exit."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agent.engine.retriever import Retriever
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.seeder import seed_knowledge
from agent.memory.semantic import SemanticMemory

console = Console()

HELP = """
Commands:
  ask <question>    Ask something. Answered honestly from what's actually been seeded.
  learn              Phase 0 stub — seeds facts.json, nothing more (see note below).
  facts               List everything currently in semantic memory.
  stats               Show memory counts.
  help                Show this message.
  exit                Quit.
"""


def run_repl(semantic: SemanticMemory, episodic: EpisodicMemory, procedural: ProceduralMemory) -> None:
    retriever = Retriever(semantic, episodic)
    console.print("[bold cyan]Agent REPL — Phase 0[/bold cyan]")
    console.print(HELP)

    while True:
        try:
            raw = console.input("[bold green]>[/bold green] ").strip()
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
        elif command == "help":
            console.print(HELP)
        elif command == "ask":
            if not rest:
                console.print("[yellow]usage: ask <question>[/yellow]")
                continue
            console.print(retriever.answer(rest))
        elif command == "learn":
            # Deliberately not real ingestion in Phase 0 — real learning
            # needs a real brain (Phase 1). This only (re-)seeds facts.json.
            inserted = seed_knowledge(semantic, force=False)
            if inserted:
                console.print(f"Knowledge base seeded ({inserted} new facts).")
            else:
                console.print("Knowledge base already seeded.")
            console.print("Real learning requires a real Brain (Phase 1).")
        elif command == "facts":
            _print_facts(semantic)
        elif command == "stats":
            _print_stats(semantic, episodic, procedural)
        else:
            console.print(f"[yellow]unknown command: {command}[/yellow] (try `help`)")


def _print_facts(semantic: SemanticMemory) -> None:
    table = Table(title="Semantic memory")
    table.add_column("id", justify="right")
    table.add_column("topic")
    table.add_column("text")
    table.add_column("source")
    table.add_column("confidence", justify="right")
    for fact in semantic.list_all():
        table.add_row(str(fact.id), fact.topic or "-", fact.text, fact.source_type, f"{fact.confidence:.2f}")
    console.print(table)


def _print_stats(semantic: SemanticMemory, episodic: EpisodicMemory, procedural: ProceduralMemory) -> None:
    table = Table(title="Memory stats")
    table.add_column("store")
    table.add_column("count", justify="right")
    table.add_row("semantic facts", str(semantic.count()))
    table.add_row("episodic log entries", str(episodic.count()))
    table.add_row("skills (stub)", str(procedural.count()))
    console.print(table)
