"""REPL loop: ask, learn, facts, stats, project, exit."""
from __future__ import annotations
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent.brains.base import BaseBrain
from agent.engine.retriever import Retriever
from agent.engine.validator import SkillValidator
from agent.engine.synthesizer import SkillSynthesizer, SynthesizerError
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.memory.seeder import seed_knowledge
from agent.memory.semantic import SemanticMemory
from agent.memory.goals import GoalMemory

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
  brain switch <provider> [model]  Switch active brain (gemini, groq, openai, local, mock).
  brain list           Show registered providers and current brain.
  help                 Show this message.
  exit                 Quit.
"""


def run_repl(
    semantic: SemanticMemory, 
    episodic: EpisodicMemory, 
    procedural: ProceduralMemory,
    project: ProjectMemory,
    goals: GoalMemory,
    brain: BaseBrain,
    embedder: EmbeddingEngine
) -> None:
    from agent.engine.state_machine import TaskFSM
    from agent.brains.factory import BrainManager
    from agent.config import ACTIVE_TASK_JSON, STATE_MANIFEST_JSON

    # Wrap the initial brain in a BrainManager so `brain switch` can hot-swap it.
    brain_manager = BrainManager.__new__(BrainManager)
    brain_manager._embedder = embedder
    brain_manager._brain = brain
    
    brain_name = brain_manager.brain.__class__.__name__
    console.print(f"[bold cyan]Agent REPL — Phase 3 (Brain: {brain_name})[/bold cyan]")
    console.print(HELP)

    fsm = TaskFSM(ACTIVE_TASK_JSON, STATE_MANIFEST_JSON)
    active_state = fsm.load_state()
    manifest_data = fsm.manifest.read_manifest()
    
    if active_state:
        console.print(f"[bold yellow]WARNING: Unfinished task detected (Goal: {active_state.goal_id}, State: {active_state.state})[/bold yellow]")
        resp = console.input("Resume task? [Y/n]: ").strip().lower()
        if resp in ('y', 'yes', ''):
            console.print("Resuming...")
            if active_state.pending_action_hash and active_state.pending_action_hash not in active_state.executed_actions:
                console.print(f"[bold red]CRITICAL: Task crashed during action {active_state.pending_action_hash}.[/bold red]")
                console.print("[yellow]The agent cannot guarantee if the last action finished executing. Transitioning to VERIFYING state.[/yellow]")
                fsm.advance("VERIFYING")
            # stub for resume loop
        else:
            fsm.clear_task()
            console.print("Task cleared.")
    elif manifest_data and manifest_data.get("active_task_hash"):
        console.print("[bold red]CRITICAL: active_task.json is missing but state_manifest.json indicates a task was running![/bold red]")
        console.print("[yellow]Please inspect episodic.db or backup states before proceeding to avoid state corruption.[/yellow]")
        console.print("Clearing manifest to allow boot...")
        fsm.clear_task()

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
        
        try:
            dispatch_command(command, rest, semantic, episodic, procedural, project, goals, brain_manager)
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
    brain_or_manager,  # BaseBrain (legacy) or BrainManager
) -> None:
    from agent.engine.state_machine import TaskFSM
    from agent.engine.task_planner import TaskPlanner
    from agent.engine.governor import PermissionGovernor
    from agent.brains.factory import BrainManager
    from agent.config import ACTIVE_TASK_JSON, STATE_MANIFEST_JSON

    # Support both a bare brain and a BrainManager (from run_repl)
    if isinstance(brain_or_manager, BrainManager):
        brain_manager = brain_or_manager
        brain = brain_manager.brain
    else:
        brain_manager = None
        brain = brain_or_manager

    retriever = Retriever(semantic, episodic, project, brain)
    validator = SkillValidator()
    synthesizer = SkillSynthesizer(brain, retriever, procedural, validator, project=project)
    fsm = TaskFSM(ACTIVE_TASK_JSON, STATE_MANIFEST_JSON)
    planner = TaskPlanner(brain, goals)
    governor = PermissionGovernor(episodic)

    if command == "help":
        console.print(HELP)
    elif command == "ask":
        if not rest:
            console.print("[yellow]usage: ask <question>[/yellow]")
            return
        console.print(retriever.answer(rest))
    elif command == "learn":
        inserted = seed_knowledge(semantic, force=False)
        if inserted:
            console.print(f"Knowledge base seeded ({inserted} new facts).")
        else:
            console.print("Knowledge base already seeded.")
    elif command == "skill":
        if not rest:
            console.print("[yellow]usage: skill <topic>[/yellow]")
            return
        console.print(f"Synthesizing skill for topic: '{rest}'...")
        try:
            skill = synthesizer.learn_skill(rest)
            console.print(f"[green]Successfully synthesized and validated skill '{skill.name}'[/green]")
        except SynthesizerError as e:
            console.print(f"[red]Failed to synthesize skill: {str(e)}[/red]")
    elif command == "skills":
        _print_skills(procedural)
    elif command == "run-skill":
        name, _, args_raw = rest.partition(" ")
        name = name.strip()
        args_raw = args_raw.strip()
        if not name:
            console.print("[yellow]usage: run-skill <name> [json_args][/yellow]")
            return
        skill = procedural.load(name)
        if not skill:
            console.print(f"[red]Skill '{name}' not found.[/red]")
            return
        try:
            console.print(f"Running '{name}'...")
            res = validator.run_saved_skill(skill, args_raw)
            console.print(f"Result: {res.model_dump_json(indent=2)}")
        except Exception as e:
            console.print(f"[red]Execution failed: {e}[/red]")
    elif command == "facts":
        _print_facts(semantic)
    elif command == "project":
        _handle_project_cmd(rest, project, brain)
    elif command == "stats":
        _print_stats(semantic, episodic, procedural, project)
    elif command == "task":
        if not rest:
            console.print("[yellow]usage: task <description>[/yellow]")
            return
        
        # 1. Plan
        console.print(f"Planning task: '{rest}'...")
        try:
            plan = planner.plan_task(rest, is_autonomous=False)
        except Exception as e:
            console.print(f"[red]Failed to plan task: {e}[/red]")
            return
            
        # 2. Show plan
        table = Table(title="Proposed Task Plan (Goal DAG)")
        table.add_column("ID")
        table.add_column("Description")
        table.add_column("Tier", justify="right")
        table.add_column("Deps")
        
        # map for short display ids
        short_ids = {g.id: f"g{i}" for i, g in enumerate(plan)}
        
        for g in plan:
            short_deps = [short_ids.get(d, d[:4]) for d in g.dependencies]
            table.add_row(short_ids[g.id], g.description, str(g.required_tier), ",".join(short_deps))
        console.print(table)
        
        # 3. Approve
        response = console.input("Approve plan and begin execution? [Y/n]: ").strip().lower()
        if response not in ('y', 'yes', ''):
            console.print("Task cancelled.")
            return
            
        # 4. Save and Start FSM
        planner.commit_plan(plan)
        task_id = plan[0].task_id if plan else "root"
        console.print("Plan committed. Starting TaskFSM...")
        fsm.start_task(task_id)
        console.print("[green]Task initialized in goals.db[/green]")
        
        try:
            result = fsm.run_to_completion(
                task_id=task_id,
                goals_db=goals,
                brain=brain,
                procedural=procedural,
                validator=validator,
                project_memory=project,
            )
            console.print(f"[bold green]{result}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Task execution failed: {e}[/bold red]")
        
    elif command == "brain":
        _handle_brain_cmd(rest, brain_manager)
    else:
        console.print(f"[yellow]unknown command: {command}[/yellow] (try `help`)")  


def _handle_project_cmd(rest: str, project: ProjectMemory, brain: BaseBrain) -> None:
    subcmd, _, args = rest.partition(" ")
    subcmd = subcmd.lower()
    args = args.strip()
    
    if subcmd == "index":
        target = Path(args) if args else Path(".")
        if not target.exists():
            console.print(f"[red]Directory not found: {target}[/red]")
            return
        
        console.print(f"Indexing workspace: {target.resolve()}")
        try:
            count = project.index_workspace(target, brain)
            console.print(f"Indexed {count} new or changed files.")
        except Exception as e:
            console.print(f"[red]Failed to index: {e}[/red]")
            
    elif subcmd == "list":
        rows = project.conn.execute("SELECT project_id, path, summary FROM project_files").fetchall()
        if not rows:
            console.print("No files indexed. Run `project index .` first.")
            return
            
        table = Table(title="Indexed Project Files")
        table.add_column("project_id", justify="right")
        table.add_column("path")
        table.add_column("summary")
        
        for r in rows:
            table.add_row(str(r["project_id"]), r["path"], r["summary"][:60])
        console.print(table)
    else:
        console.print("[yellow]usage: project index <dir> | project list[/yellow]")


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

def _print_skills(procedural: ProceduralMemory) -> None:
    table = Table(title="Procedural memory (Skills)")
    table.add_column("id", justify="right")
    table.add_column("name")
    table.add_column("tier")
    table.add_column("path")
    for skill in procedural.list():
        table.add_row(str(skill.id), skill.name, skill.verification_tier, skill.file_path)
    console.print(table)

def _print_stats(semantic: SemanticMemory, episodic: EpisodicMemory, procedural: ProceduralMemory, project: ProjectMemory) -> None:
    table = Table(title="Memory stats")
    table.add_column("store")
    table.add_column("count", justify="right")
    table.add_row("semantic facts", str(semantic.count()))
    table.add_row("episodic log entries", str(episodic.count()))
    table.add_row("skills", str(procedural.count()))
    table.add_row("project files", str(project.count()))
    console.print(table)


def _handle_brain_cmd(rest: str, brain_manager) -> None:
    """Handle `brain switch <provider> [model]` and `brain list`."""
    if brain_manager is None:
        console.print("[red]Brain switching is not available in this context.[/red]")
        return

    subcmd, _, args = rest.partition(" ")
    subcmd = subcmd.strip().lower()
    args = args.strip()

    if subcmd == "list":
        providers = brain_manager.list_available()
        current = brain_manager.brain.__class__.__name__
        table = Table(title="Brain Providers")
        table.add_column("provider")
        table.add_column("status")
        for p in providers:
            status = f"[green]active ({current})[/green]" if current.lower().startswith(p.replace("_", "")) else ""
            table.add_row(p, status)
        console.print(table)
        console.print(f"Active brain: [bold cyan]{current}[/bold cyan]")
        if hasattr(brain_manager.brain, "model"):
            # Safe print — only show base_url (no query strings) and model
            base = getattr(brain_manager.brain, "base_url", "")
            safe_base = base.split("?")[0] if base else "n/a"
            console.print(f"  provider=local  base_url={safe_base}  model={brain_manager.brain.model}")

    elif subcmd == "switch":
        if not args:
            console.print("[yellow]usage: brain switch <provider> [model][/yellow]")
            console.print("Providers: " + ", ".join(brain_manager.list_available()))
            return

        parts = args.split()
        provider = parts[0]
        model = parts[1] if len(parts) > 1 else "auto"

        console.print(f"Switching brain to [bold]{provider}[/bold] (model={model})...")
        new_brain = brain_manager.switch_brain(provider, model=model)
        new_name = new_brain.__class__.__name__

        # Safe summary — never print API keys or query string tokens
        base = getattr(new_brain, "base_url", "")
        safe_base = base.split("?")[0] if base else "n/a"
        actual_model = getattr(new_brain, "model", model)

        console.print(f"[green]Brain switched.[/green]")
        console.print(f"  provider={provider}")
        if safe_base != "n/a":
            console.print(f"  base_url={safe_base}")
        console.print(f"  model={actual_model}")
        console.print(f"  class={new_name}")

    else:
        console.print("[yellow]usage: brain switch <provider> [model] | brain list[/yellow]")
