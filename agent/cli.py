# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
  self-model           Show empirical competence matrix and boot count.
  benchmark reasoning  Run ZPD reasoning calibration.
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
    embedder: EmbeddingEngine,
    self_model=None,          # Phase 4A: SelfModel | None
    pause_event=None,         # Phase 4A: threading.Event | None
) -> None:
    from agent.engine.state_machine import TaskFSM
    from agent.brains.factory import BrainManager
    from agent.config import ACTIVE_TASK_JSON, STATE_MANIFEST_JSON

    # Wrap the initial brain in a BrainManager so `brain switch` can hot-swap it.
    brain_manager = BrainManager(embedder=embedder, brain=brain)
    
    brain_name = brain_manager.brain.__class__.__name__
    _pause = pause_event  # local alias — set before input, clear after
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
            if _pause is not None:
                _pause.set()       # signal heartbeat: user is active
            raw = console.input("[bold green]>[/bold green] ").strip()
            if _pause is not None:
                _pause.clear()     # user finished typing; daemon may resume
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
            dispatch_command(command, rest, semantic, episodic, procedural, project, goals,
                             brain_manager, self_model=self_model)
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
    brain_or_manager,           # BaseBrain (legacy) or BrainManager
    self_model=None,            # Phase 4A: SelfModel | None
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

    governor = PermissionGovernor(episodic)
    retriever = Retriever(semantic, episodic, project, brain)
    validator = SkillValidator()
    synthesizer = SkillSynthesizer(brain, retriever, procedural, validator, project=project, governor=governor)
    fsm = TaskFSM(ACTIVE_TASK_JSON, STATE_MANIFEST_JSON)
    planner = TaskPlanner(brain, goals)

    if command == "help":
        console.print(HELP)
    elif command == "ask":
        if not rest:
            console.print("[yellow]usage: ask <question>[/yellow]")
            return
        console.print(retriever.answer(rest))
    elif command == "learn":
        if not rest:
            console.print("[yellow]usage: learn <topic> | learn resume[/yellow]")
            return
            
        from agent.engine.planner import CurriculumPlanner
        from agent.engine.ingest import search_sources, extract_clean_text, IngestionAbortError
        from agent.engine.synthesizer import KnowledgeSynthesizer
        from agent.brains.base import QuotaExceededError
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        from agent.engine.exporter import init_markdown_note, append_unit_to_markdown
        import time
        
        planner_cur = CurriculumPlanner(brain)
        synthesizer_know = KnowledgeSynthesizer(brain, semantic)
        
        topic = rest
        units = []
        completed_units = []
        
        if topic == "resume":
            ckpt = planner_cur.load_checkpoint()
            if not ckpt:
                console.print("[yellow]No active curriculum found to resume.[/yellow]")
                return
            topic = ckpt["topic"]
            units = ckpt["units_data"]
            completed_units = ckpt.get("completed_units", [])
            console.print(f"[green]Resuming curriculum for '{topic}' (completed {len(completed_units)}/{len(units)} units).[/green]")
        else:
            if planner_cur.has_checkpoint(topic):
                resp = console.input(f"Found active curriculum for '{topic}'. Resume? [Y/n]: ").strip().lower()
                if resp in ('y', 'yes', ''):
                    ckpt = planner_cur.load_checkpoint()
                    units = ckpt["units_data"]
                    completed_units = ckpt.get("completed_units", [])
                    console.print(f"[green]Resuming...[/green]")
                else:
                    console.print("Starting fresh...")
                    planner_cur.clear_checkpoint()
            
            if not units:
                console.print(f"Initializing Curriculum Planner for '{topic}'...")
                try:
                    units = planner_cur.plan_curriculum(topic)
                    console.print(f"[green]Decomposed topic into {len(units)} study units.[/green]")
                    for i, u in enumerate(units, 1):
                        console.print(f"  {i}. {u}")
                    planner_cur.save_checkpoint(topic, units, completed_units)
                except Exception as e:
                    console.print(f"[red]Failed to plan curriculum: {e}[/red]")
                    return
        
        total_facts, total_passages = 0, 0
        quota_hit = False
        
        with Progress(
            SpinnerColumn(style="green"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            overall_task = progress.add_task("Learning curriculum...", total=len(units))
            
            brain_name = getattr(brain, "model", brain.__class__.__name__)
            init_markdown_note(topic, len(units), brain_name)
            
            if completed_units:
                progress.advance(overall_task, len(completed_units))
                
            for i, unit in enumerate(units, 1):
                if i in completed_units:
                    continue
                    
                progress.update(overall_task, description=f"[cyan]Unit {i}/{len(units)}:[/] {unit[:45]}...")
                
                unit_success = True
                unit_facts = 0
                f_added_this_unit, p_added_this_unit = 0, 0
                
                unit_exported_facts = []
                unit_exported_passages = []
                unit_sources = []
                
                urls = search_sources(unit, max_results=2)
                if not urls:
                    console.print(f"  [yellow][WARN] Unit {i}: Search returned 0 links[/yellow]")
                    unit_success = False
                
                for url in urls:
                    unit_sources.append(url)
                    try:
                        raw_text = extract_clean_text(url)
                        # Retry loop for failover
                        while True:
                            try:
                                added_facts, added_passages = synthesizer_know.distill_to_semantic_db(raw_text, topic)
                                
                                f_count = len(added_facts)
                                p_count = len(added_passages)
                                
                                total_facts += f_count
                                total_passages += p_count
                                f_added_this_unit += f_count
                                p_added_this_unit += p_count
                                unit_facts += f_count + p_count
                                
                                unit_exported_facts.extend([{"statement": f.text, "confidence": f.confidence} for f in added_facts])
                                unit_exported_passages.extend([p.text for p in added_passages])
                                break
                            except QuotaExceededError as qe:
                                if brain_manager:
                                    try:
                                        brain_manager.switch_to_next_available()
                                        synthesizer_know.brain = brain_manager.brain
                                        planner_cur.brain = brain_manager.brain
                                        continue
                                    except RuntimeError:
                                        unit_success = False
                                        quota_hit = True
                                        break
                                else:
                                    unit_success = False
                                    quota_hit = True
                                    break
                    except IngestionAbortError:
                        pass
                    except Exception as e:
                        console.print(f"  [yellow]Warning: Ingestion failed for {url} - {e}[/yellow]")
                        
                    if quota_hit:
                        break
                
                if quota_hit:
                    break
                    
                if unit_success and unit_facts == 0:
                    console.print(f"  [yellow][WARN] Unit {i}: Distillation parsed 0 facts (Skipping checkpoint).[/yellow]")
                    unit_success = False
                    
                if unit_success:
                    completed_units.append(i)
                    planner_cur.save_checkpoint(topic, units, completed_units)
                    
                    console.print(
                        f"  [green]✓[/] [bold]Unit {i}/{len(units)}:[/] {unit[:60]}... "
                        f"— [dim]Added {f_added_this_unit} facts, {p_added_this_unit} passages[/]"
                    )
                    
                    append_unit_to_markdown(
                        topic=topic,
                        unit_index=i,
                        total_units=len(units),
                        unit_title=unit,
                        passages=unit_exported_passages,
                        facts=unit_exported_facts,
                        sources=unit_sources
                    )
                
                progress.advance(overall_task, 1)
                
                # Pacing Delay
                time.sleep(3.0)
                
        if quota_hit:
            console.print("[bold red]All brain quotas exhausted. Saved progress to active_curriculum.json.[/bold red]")
            if total_facts == 0:
                console.print("[bold yellow][WARNING] Ingestion failed: 0 facts extracted due to API quota errors. Try switching brains or wait for quota reset.[/bold yellow]")
        else:
            from agent.engine.exporter import get_topic_slug
            console.print(f"[bold green]Ingestion complete![/bold green] Added {total_facts} facts and {total_passages} passages to Semantic Memory.")
            console.print(f"[bold green]Saved human-readable research notes to:[/] [cyan]data/knowledge/{get_topic_slug(topic)}.md[/]")
            planner_cur.clear_checkpoint()
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
        _print_facts(semantic, rest)
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
                governor=governor,
            )
            console.print(f"[bold green]{result}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Task execution failed: {e}[/bold red]")
        
    elif command == "self-model":
        _print_self_model(self_model)
    elif command == "benchmark" and rest.startswith("reasoning"):
        _handle_benchmark_reasoning(rest, brain_manager, self_model)
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


def _print_facts(semantic: SemanticMemory, query: str = "") -> None:
    query = query.strip()
    if query:
        query_term = f"%{query.lower()}%"
        rows = semantic.conn.execute(
            """
            SELECT id, topic, text, source_type, confidence 
            FROM facts 
            WHERE LOWER(topic) LIKE ? OR LOWER(text) LIKE ?
            ORDER BY id ASC
            """,
            (query_term, query_term)
        ).fetchall()
        
        if not rows:
            console.print(f"[yellow]No facts found matching '{query}'.[/yellow]")
            return
            
        from agent.models import Fact
        facts = [Fact(id=r["id"], topic=r["topic"], text=r["text"], source_type=r["source_type"], confidence=r["confidence"]) for r in rows]
    else:
        facts = semantic.list_all()

    table = Table(title=f"Semantic memory{' (Search: ' + query + ')' if query else ''}")
    table.add_column("id", justify="right")
    table.add_column("topic")
    table.add_column("text")
    table.add_column("source")
    table.add_column("confidence", justify="right")
    for fact in facts:
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


def _print_self_model(self_model) -> None:
    if self_model is None:
        console.print("[yellow]Self-model not available (agent started without Phase 4A).[/yellow]")
        return
    data = self_model.as_dict()
    # Header info
    console.print(f"[bold cyan]Self-Model[/bold cyan]  identity={data.get('identity')}  "
                  f"boot_count={data.get('boot_count', 0)}  "
                  f"last_reflection={data.get('last_reflection_at') or 'never'}")

    # Competence matrix
    matrix = data.get("empirical_competence_matrix", {})
    if matrix:
        table = Table(title="Empirical Competence Matrix")
        table.add_column("topic")
        table.add_column("verified", justify="right")
        table.add_column("failed", justify="right")
        table.add_column("pass_ratio", justify="right")
        table.add_column("confidence", justify="right")
        for topic, entry in sorted(matrix.items()):
            table.add_row(
                topic,
                str(entry.get("skills_verified", 0)),
                str(entry.get("skills_failed", 0)),
                f"{entry.get('pass_ratio', 0.0):.0%}",
                f"{entry.get('confidence', 0.0):.2f}",
            )
        console.print(table)
    else:
        console.print("[dim]No competence data yet.[/dim]")

    # Reasoning profile global scores
    profile = data.get("reasoning_profile", {})
    global_scores = profile.get("global_scores", {})
    if global_scores:
        t2 = Table(title="Reasoning Global Scores")
        t2.add_column("strategy")
        t2.add_column("score", justify="right")
        for k, v in sorted(global_scores.items()):
            t2.add_row(k, f"{v:.3f}")
        console.print(t2)

    # Known gaps
    gaps = data.get("known_knowledge_gaps", [])
    if gaps:
        console.print(f"[yellow]Known gaps:[/yellow] {', '.join(gaps)}")


def _handle_benchmark_reasoning(rest: str, brain_manager, self_model) -> None:
    from agent.engine.benchmark import run_reasoning_benchmark
    run_reasoning_benchmark(rest, brain_manager.brain, self_model)


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
