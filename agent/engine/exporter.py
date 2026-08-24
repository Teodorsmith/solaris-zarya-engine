import re
from pathlib import Path
from datetime import datetime, timezone
import json

def get_topic_slug(topic: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]+', '_', topic.strip().lower()).strip('_')

def init_markdown_note(topic: str, total_units: int, brain_model: str) -> Path:
    knowledge_dir = Path("data/knowledge")
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    file_path = knowledge_dir / f"{get_topic_slug(topic)}.md"
    
    if not file_path.exists():
        header = f"""# Curriculum Research: {topic}

- **Generated on:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Brain Model:** {brain_model}
- **Planned Units:** {total_units}

---
"""
        file_path.write_text(header, encoding="utf-8")
    return file_path

def append_unit_to_markdown(
    topic: str,
    unit_index: int,
    total_units: int,
    unit_title: str,
    passages: list[str],
    facts: list[dict],
    sources: list[str]
) -> None:
    file_path = Path("data/knowledge") / f"{get_topic_slug(topic)}.md"
    
    content = [
        f"\n## Unit {unit_index}/{total_units}: {unit_title}\n"
    ]
    
    if passages:
        content.append("### Context & Narrative Summary\n")
        for p in passages:
            content.append(f"{p.strip()}\n")
    
    if facts:
        content.append("\n### Distilled Atomic Facts\n")
        for f in facts:
            statement = f.get("statement", "")
            confidence = f.get("confidence", 0.95)
            content.append(f"- **[Confidence: {confidence:.2f}]** {statement}")
        content.append("")

    if sources:
        content.append("\n### Sources Consulted\n")
        for s in sources:
            content.append(f"- <{s}>")
        content.append("")
        
    content.append("\n---\n")
    
    with file_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(content))
