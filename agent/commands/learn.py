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

import time
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from agent.brains.base import QuotaExceededError
from agent.engine.exporter import append_unit_to_markdown, init_markdown_note, get_topic_slug
from agent.engine.ingest import IngestionAbortError, extract_clean_text
from agent.engine.planner import CurriculumPlanner, build_search_query
from agent.engine.search import search_sources
from agent.engine.synthesizer import KnowledgeSynthesizer

console = Console()

def handle_learn(rest: str, semantic, brain, brain_manager=None) -> None:
    if not rest:
        console.print("[yellow]usage: learn <topic> | learn resume[/yellow]")
        return

    planner_cur = CurriculumPlanner(brain, semantic=semantic)
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
            if resp in ("y", "yes", ""):
                ckpt = planner_cur.load_checkpoint()
                units = ckpt["units_data"]
                completed_units = ckpt.get("completed_units", [])
                console.print("[green]Resuming...[/green]")
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
        console=console,
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

            search_query = build_search_query(topic, unit)
            urls = search_sources(search_query, max_results=2)
            if not urls:
                console.print(f"  [yellow][WARN] Unit {i}: Search returned 0 links[/yellow]")
                unit_success = False

            for url in urls:
                unit_sources.append(url)
                if semantic.is_url_ingested(url):
                    continue
                try:
                    raw_text = extract_clean_text(url)
                    if not raw_text:
                        continue
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
                            semantic.mark_url_ingested(url)
                            break
                        except QuotaExceededError:
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
                console.print(f"  [green]✓[/] [bold]Unit {i}/{len(units)}:[/] {unit[:60]}... — [dim]Added {f_added_this_unit} facts, {p_added_this_unit} passages[/]")
                append_unit_to_markdown(topic=topic, unit_index=i, total_units=len(units), unit_title=unit, passages=unit_exported_passages, facts=unit_exported_facts, sources=unit_sources)

            progress.advance(overall_task, 1)
            time.sleep(3.0)

    if quota_hit:
        console.print("[bold red]All brain quotas exhausted. Saved progress to active_curriculum.json.[/bold red]")
        if total_facts == 0:
            console.print("[bold yellow][WARNING] Ingestion failed: 0 facts extracted due to API quota errors. Try switching brains or wait for quota reset.[/bold yellow]")
    else:
        console.print(f"[bold green]Ingestion complete![/bold green] Added {total_facts} facts and {total_passages} passages to Semantic Memory.")
        console.print(f"[bold green]Saved human-readable research notes to:[/] [cyan]data/knowledge/{get_topic_slug(topic)}.md[/]")
        planner_cur.clear_checkpoint()
