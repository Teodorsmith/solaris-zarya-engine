from rich.console import Console
from rich.table import Table

console = Console()

def handle_read_file(rest: str) -> None:
    if not rest:
        console.print("[yellow]usage: read <file_path>[/yellow]")
        return

    import pathlib
    file_path = pathlib.Path(rest.strip())
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        return

    try:
        content = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".md":
            import rich.markdown
            console.print(rich.markdown.Markdown(content))
        else:
            import rich.syntax
            lexer_name = file_path.suffix.lstrip(".") if file_path.suffix else "text"
            console.print(rich.syntax.Syntax(content, lexer_name, theme="monokai", line_numbers=True))
    except Exception as e:
        console.print(f"[red]Failed to read file: {e}[/]")

def handle_stats(semantic, episodic, procedural, project) -> None:
    table = Table(title="Memory stats")
    table.add_column("store")
    table.add_column("count", justify="right")
    table.add_row("semantic facts", str(semantic.count()))
    table.add_row("episodic log entries", str(episodic.count()))
    table.add_row("skills", str(procedural.count()))
    table.add_row("project files", str(project.count()))
    console.print(table)


def handle_self_model(self_model) -> None:
    if self_model is None:
        console.print("[yellow]Self-model not available (agent started without Phase 4A).[/yellow]")
        return
    data = self_model.as_dict()
    console.print(f"[bold cyan]Self-Model[/bold cyan]  identity={data.get('identity')}  boot_count={data.get('boot_count', 0)}  last_reflection={data.get('last_reflection_at') or 'never'}")

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

    profile = data.get("reasoning_profile", {})
    global_scores = profile.get("global_scores", {})
    if global_scores:
        t2 = Table(title="Reasoning Global Scores")
        t2.add_column("strategy")
        t2.add_column("score", justify="right")
        for k, v in sorted(global_scores.items()):
            t2.add_row(k, f"{v:.3f}")
        console.print(t2)

    gaps = data.get("known_knowledge_gaps", [])
    if gaps:
        console.print(f"[yellow]Known gaps:[/yellow] {', '.join(gaps)}")


def handle_benchmark_reasoning(rest: str, brain_manager, self_model) -> None:
    from agent.engine.benchmark import run_reasoning_benchmark
    run_reasoning_benchmark(rest, brain_manager, self_model)


def handle_brain_cmd(rest: str, brain_manager) -> None:
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
            # Check if this provider string is contained within the current active class name
            is_active = p.replace("_", "").lower() in current.lower()
            status = f"[green]active ({current})[/green]" if is_active else "[dim]available[/dim]"
            table.add_row(p, status)
        console.print(table)
        console.print(f"Active brain: [bold cyan]{current}[/bold cyan]")
        if hasattr(brain_manager.brain, "model"):
            base = getattr(brain_manager.brain, "base_url", "")
            safe_base = base.split("?")[0] if base else "n/a"
            console.print(f"  provider=local  base_url={safe_base}  model={brain_manager.brain.model}")

    elif subcmd == "switch":
        if not args:
            console.print("[yellow]usage: brain switch <provider> [model][/yellow]")
            return
        provider, _, model = args.partition(" ")
        provider = provider.strip()
        model = model.strip()
        try:
            old_name = brain_manager.brain.__class__.__name__
            if model:
                brain_manager.switch(provider, model=model)
            else:
                brain_manager.switch(provider)
            new_name = brain_manager.brain.__class__.__name__
            console.print(f"[green]Switched brain: {old_name} -> {new_name}[/green]")
            if hasattr(brain_manager.brain, "model"):
                console.print(f"Model: {brain_manager.brain.model}")
        except Exception as e:
            console.print(f"[red]Failed to switch brain: {e}[/red]")
    else:
        console.print("[yellow]usage: brain list | brain switch <provider> [model][/yellow]")
