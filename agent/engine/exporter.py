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

import re
from datetime import datetime, timezone
from pathlib import Path


def get_topic_slug(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", topic.strip().lower()).strip("_")


def init_markdown_note(topic: str, total_units: int, brain_model: str) -> Path:
    knowledge_dir = Path("data/knowledge")
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    file_path = knowledge_dir / f"{get_topic_slug(topic)}.md"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if not file_path.exists():
        header = f"""# Curriculum Research: {topic}

- **Generated on:** {now_str}
- **Last Enriched:** {now_str}
- **Brain Model:** {brain_model}
- **Total Units:** {total_units}

---
"""
        file_path.write_text(header, encoding="utf-8")
    else:
        # Update / add enrichment metadata in-place without touching unit content
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        updated_lines = []
        enriched_seen = False
        units_seen = False
        for line in lines:
            if line.startswith("- **Last Enriched:**"):
                updated_lines.append(f"- **Last Enriched:** {now_str}\n")
                enriched_seen = True
            elif line.startswith("- **Total Units:**"):
                updated_lines.append(f"- **Total Units:** {total_units}\n")
                units_seen = True
            else:
                updated_lines.append(line)
        # If the fields weren't present (older format), insert after the first header line
        if not enriched_seen or not units_seen:
            insert_after = 0
            for i, line in enumerate(updated_lines):
                if line.startswith("# "):
                    insert_after = i + 1
                    break
            injection = []
            if not enriched_seen:
                injection.append(f"- **Last Enriched:** {now_str}\n")
            if not units_seen:
                injection.append(f"- **Total Units:** {total_units}\n")
            updated_lines = updated_lines[:insert_after] + ["\n"] + injection + updated_lines[insert_after:]
        file_path.write_text("".join(updated_lines), encoding="utf-8")
    return file_path


def append_unit_to_markdown(
    topic: str,
    unit_index: int,
    total_units: int,
    unit_title: str,
    passages: list[str],
    facts: list[dict],
    sources: list[str],
) -> None:
    file_path = Path("data/knowledge") / f"{get_topic_slug(topic)}.md"

    content = [f"\n## Unit {unit_index}/{total_units}: {unit_title}\n"]

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
