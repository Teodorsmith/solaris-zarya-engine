from rich.console import Console
from rich.table import Table

from agent.memory.semantic import SemanticMemory

console = Console()

def handle_facts(rest: str, semantic: SemanticMemory) -> None:
    query = rest.strip()
    if query:
        query_term = f"%{query.lower()}%"
        rows = semantic.conn.execute(
            """
            SELECT id, topic, text, source_type, confidence 
            FROM facts 
            WHERE LOWER(topic) LIKE ? OR LOWER(text) LIKE ?
            ORDER BY id ASC
            """,
            (query_term, query_term),
        ).fetchall()

        if not rows:
            console.print(f"[yellow]No facts found matching '{query}'.[/yellow]")
            return

        from agent.models import Fact
        facts = [
            Fact(
                id=r["id"],
                topic=r["topic"],
                text=r["text"],
                source_type=r["source_type"],
                confidence=r["confidence"],
            )
            for r in rows
        ]
    else:
        facts = semantic.list_all()

    table = Table(title=f"Semantic memory{' (Search: ' + query + ')' if query else ''}")
    table.add_column("id", justify="right")
    table.add_column("topic")
    table.add_column("text")
    table.add_column("source")
    table.add_column("confidence", justify="right")
    for fact in facts:
        table.add_row(
            str(fact.id),
            fact.topic or "-",
            fact.text,
            fact.source_type,
            f"{fact.confidence:.2f}",
        )
    console.print(table)
