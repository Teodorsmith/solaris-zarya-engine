# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from rich.console import Console

from agent.engine.academic import AcademicIngester
from agent.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)
console = Console()


def handle_ingest_paper(query: str, semantic: SemanticMemory) -> None:
    """CLI handler for ingest-paper <query>."""
    if not query.strip():
        console.print("[red]Error: You must provide an arXiv ID, URL, or search query.[/red]")
        return
        
    console.print(f"[bold cyan]Ingesting academic paper for query:[/bold cyan] {query}")
    ingester = AcademicIngester()
    
    try:
        facts = ingester.ingest_paper(query)
    except Exception as e:
        console.print(f"[red]Error during academic ingestion: {e}[/red]")
        return
        
    if not facts:
        console.print("[yellow]No facts could be extracted from the paper. See logs for details.[/yellow]")
        return
        
    console.print(f"[green]Successfully parsed paper into {len(facts)} chunks. Ingesting to Semantic Memory...[/green]")
    
    for fact in facts:
        semantic.add_fact(fact)
        
    console.print("[bold green]Ingestion complete![/bold green]")
