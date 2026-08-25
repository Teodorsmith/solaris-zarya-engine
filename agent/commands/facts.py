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

def handle_correct(rest: str, semantic: SemanticMemory, episodic) -> None:
    import re
    rest = rest.strip()
    
    # Format 1: correct <fact_id> <new_text>
    match_id = re.match(r"^(\d+)\s+(.+)$", rest)
    if match_id:
        fact_id = int(match_id.group(1))
        new_text = match_id.group(2)
        try:
            new_id = semantic.correct_fact(fact_id, new_text)
            episodic.log_action(f"User corrected fact {fact_id} to new fact {new_id}: {new_text}", success=True)
            console.print(f"[green]Corrected fact {fact_id}. New fact ID: {new_id}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to correct fact: {e}[/red]")
        return
        
    # Format 2: correct <topic> "<old>" -> "<new>"
    # e.g. correct Unity "old thing" -> "new thing"
    match_topic = re.match(r'^([\w\s-]+?)\s+"([^"]+)"\s*->\s*"([^"]+)"$', rest)
    if match_topic:
        topic = match_topic.group(1).strip()
        old_text = match_topic.group(2)
        new_text = match_topic.group(3)
        
        # Find the old fact
        rows = semantic.conn.execute(
            "SELECT id FROM facts WHERE topic = ? AND text = ? AND is_superseded = 0",
            (topic, old_text)
        ).fetchall()
        
        if not rows:
            console.print(f"[yellow]Could not find active fact under topic '{topic}' matching: {old_text}[/yellow]")
            return
            
        fact_id = rows[0]["id"]
        try:
            new_id = semantic.correct_fact(fact_id, new_text)
            episodic.log_action(f"User corrected fact {fact_id} to new fact {new_id}: {new_text}", success=True)
            console.print(f"[green]Corrected fact {fact_id}. New fact ID: {new_id}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to correct fact: {e}[/red]")
        return
        
    console.print("[yellow]Usage:\n  correct <fact_id> <new_text>\n  correct <topic> \"<old_text>\" -> \"<new_text>\"[/yellow]")
