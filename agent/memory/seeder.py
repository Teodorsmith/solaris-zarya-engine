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

"""
Loads seed_data/facts.json into semantic memory on first boot.
Idempotent: skipped if SEED_VERSION in config.py matches what's already
been loaded into this database, unless force=True.
"""

from __future__ import annotations

import json
import sqlite3

from agent.config import SEED_DATA_DIR, SEED_VERSION
from agent.memory.semantic import SemanticMemory
from agent.models import Fact


def _seed_marker_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seed_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()


def _already_seeded_at_current_version(conn: sqlite3.Connection) -> bool:
    _seed_marker_table(conn)
    row = conn.execute(
        "SELECT value FROM seed_meta WHERE key='seed_version'"
    ).fetchone()
    return row is not None and int(row[0]) == SEED_VERSION


def _mark_seeded(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO seed_meta (key, value) VALUES ('seed_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SEED_VERSION),),
    )
    conn.commit()


def seed_knowledge(semantic: SemanticMemory, force: bool = False) -> int:
    """Returns the number of facts inserted (0 if already seeded at the
    current SEED_VERSION and not forced)."""
    if _already_seeded_at_current_version(semantic.conn) and not force:
        return 0

    facts_path = SEED_DATA_DIR / "facts.json"
    if not facts_path.exists():
        raise FileNotFoundError(f"No seed data at {facts_path}")

    raw = json.loads(facts_path.read_text(encoding="utf-8"))
    inserted = 0
    for item in raw:
        fact = Fact(
            text=item["text"],
            topic=item.get("topic"),
            confidence=item.get("confidence", 0.7),
            source_type="seed",
        )
        created, _id = semantic.add_fact(fact)
        if created:
            inserted += 1

    _mark_seeded(semantic.conn)
    return inserted
