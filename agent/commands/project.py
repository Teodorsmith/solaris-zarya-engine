from pathlib import Path
from rich.console import Console
from rich.table import Table

from agent.brains.base import BaseBrain
from agent.memory.project import ProjectMemory

console = Console()

def handle_project_cmd(rest: str, project: ProjectMemory, brain: BaseBrain) -> None:
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
