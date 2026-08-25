from rich.console import Console
from rich.table import Table

console = Console()

def handle_task(rest: str, planner, fsm, goals, brain, procedural, validator, project, governor) -> None:
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
        table.add_row(
            short_ids[g.id],
            g.description,
            str(g.required_tier),
            ",".join(short_deps),
        )
    console.print(table)

    # 3. Approve
    response = console.input("Approve plan and begin execution? [Y/n]: ").strip().lower()
    if response not in ("y", "yes", ""):
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
            episodic_memory=governor.episodic_memory if governor else None,
        )
        console.print(f"[bold green]{result}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Task execution failed: {e}[/bold red]")
