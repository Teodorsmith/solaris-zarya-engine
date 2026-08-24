# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Benchmark reasoning command wrapper."""
from __future__ import annotations

import argparse
import logging
import shlex

from rich.console import Console
from rich.table import Table

from agent.brains.base import BaseBrain
from agent.brains.mock_brain import MockBrain
from tests.reasoning_suite.runner import ZPDRunner

logger = logging.getLogger(__name__)


def run_reasoning_benchmark(rest: str, brain: BaseBrain, self_model) -> None:
    console = Console()
    
    # Simple parsing of the rest of the command
    parts = shlex.split(rest)
    if parts and parts[0] == "reasoning":
        parts = parts[1:]
        
    parser = argparse.ArgumentParser(prog="benchmark reasoning")
    parser.add_argument("--category", type=str, help="Run only this category")
    parser.add_argument("--dry-run", action="store_true", help="Do not write ceilings to self-model")
    
    try:
        args = parser.parse_args(parts)
    except SystemExit:
        return
        
    console.print(f"[bold cyan]ZPD Reasoning Calibration[/bold cyan] (Brain: {brain.__class__.__name__})")
    
    runner = ZPDRunner(brain)
    
    # We run the binary search
    if args.category:
        if args.category not in runner.fixtures:
            console.print(f"[red]Error: unknown category {args.category}[/red]")
            return
        ceilings = {args.category: runner._search_category(args.category)}
    else:
        ceilings = runner.run_all(dry_run=args.dry_run)
        
    # Output table
    table = Table(title="ZPD Reasoning Ceilings")
    table.add_column("Category")
    table.add_column("Ceiling", justify="right")
    
    for cat, ceil in sorted(ceilings.items()):
        table.add_row(cat, str(ceil))
        
    console.print(table)
    
    # Update self-model (unless dry-run or mock brain)
    if args.dry_run:
        console.print("[dim]Dry-run: skipping self-model update.[/dim]")
    elif isinstance(brain, MockBrain):
        console.print("[yellow]Notice: MockBrain detected. Not persisting ZPD ceilings.[/yellow]")
    elif self_model:
        # Assuming self_model has a method to update zpd ceilings, which we need to add.
        self_model.update_zpd_ceilings(ceilings)
        console.print("[green]ZPD ceilings saved to self_model.[/green]")
    else:
        console.print("[yellow]Warning: no self-model loaded, cannot persist results.[/yellow]")
