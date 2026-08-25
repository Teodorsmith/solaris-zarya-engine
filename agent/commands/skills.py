# Solaris Zarya Engine
# Copyright (C) 2026 Teodor Smith <teosmith.studios@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

from rich.console import Console
from rich.table import Table

from agent.engine.synthesizer import SynthesizerError
from agent.memory.procedural import ProceduralMemory

console = Console()

def handle_skill(rest: str, synthesizer) -> None:
    if not rest:
        console.print("[yellow]usage: skill <topic>[/yellow]")
        return
    console.print(f"Synthesizing skill for topic: '{rest}'...")
    try:
        skill = synthesizer.learn_skill(rest)
        console.print(f"[green]Successfully synthesized and validated skill '{skill.name}'[/green]")
    except SynthesizerError as e:
        console.print(f"[red]Failed to synthesize skill: {e!s}[/red]")


def handle_skills(rest: str, procedural: ProceduralMemory) -> None:
    query = rest.strip()
    if query:
        query_pattern = f"%{query.lower()}%"
        rows = procedural.conn.execute(
            """
            SELECT id, name, verification_tier, file_path, description 
            FROM skills 
            WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
            ORDER BY id ASC
            """,
            (query_pattern, query_pattern),
        ).fetchall()

        if not rows:
            console.print(f"[yellow]No skills found matching '{query}'.[/yellow]")
            return

        from agent.models import Skill
        skills = [
            Skill(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                file_path=r["file_path"],
                verification_tier=r["verification_tier"],
                created_at="",
            )
            for r in rows
        ]
    else:
        skills = procedural.list()

    table = Table(title=f"Procedural memory (Skills){' (Search: ' + query + ')' if query else ''}")
    table.add_column("id", justify="right")
    table.add_column("name")
    table.add_column("tier")
    table.add_column("path")
    for skill in skills:
        table.add_row(str(skill.id), skill.name, skill.verification_tier, skill.file_path)
    console.print(table)


def handle_run_skill(rest: str, procedural: ProceduralMemory, validator) -> None:
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
