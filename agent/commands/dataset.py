from rich.console import Console
from rich.table import Table

console = Console()


def handle_dataset(rest: str, dataset_builder) -> None:
    args = rest.strip().split()
    if not args:
        console.print("[yellow]usage: dataset <stats|clear|build [--dry-run] [--limit N]>[/yellow]")
        return
        
    cmd = args[0].lower()
    
    if cmd == "stats":
        stats = dataset_builder.get_stats()
        table = Table(title="DPO Dataset Statistics")
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Total Pairs", str(stats["total_pairs"]))
        table.add_row("Unique Cached Prompts", str(stats["unique_prompts"]))
        
        # format bytes
        size_bytes = stats["file_size"]
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024**2:
            size_str = f"{size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{size_bytes / (1024**2):.2f} MB"
            
        table.add_row("File Size", size_str)
        table.add_row("Last Addition", stats["last_addition"] or "Never")
        
        console.print(table)
        
    elif cmd == "clear":
        ans = console.input("This will delete data/dpo_dataset.jsonl. Proceed? [y/N]: ").strip().lower()
        if ans == "y":
            dataset_builder.clear_dataset()
            console.print("[green]Dataset cleared.[/green]")
        else:
            console.print("Operation cancelled.")
            
    elif cmd == "build":
        dry_run = "--dry-run" in args
        limit = None
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                try:
                    limit = int(args[idx + 1])
                except ValueError:
                    console.print("[red]Invalid limit value.[/red]")
                    return
                    
        candidates = dataset_builder.harvest_from_episodic(limit=limit, dry_run=dry_run)
        
        if dry_run:
            console.print("[yellow]--- DRY RUN PREVIEW ---[/yellow]")
            table = Table(title=f"Candidates found ({len(candidates)})")
            table.add_column("Prompt Extract")
            table.add_column("Chosen Size")
            table.add_column("Rejected Size")
            table.add_column("Metadata")
            for c in candidates:
                table.add_row(
                    c["prompt"][:50].replace("\n", " "),
                    str(len(c["chosen"])),
                    str(len(c["rejected"])),
                    str(c["metadata"])
                )
            console.print(table)
            console.print("[yellow]--- NO DATA WRITTEN ---[/yellow]")
        else:
            console.print(f"[green]Successfully harvested {len(candidates)} pairs.[/green]")
            
    else:
        console.print(f"[red]Unknown dataset command: {cmd}[/red]")
