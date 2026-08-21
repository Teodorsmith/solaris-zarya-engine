"""
Tier 3: Procedural memory (the skill store). STUB for Phase 0.

The schema and basic CRUD exist so Phase 2 (real skill synthesis and
execution) has a stable table to build on, but nothing in Phase 0 ever
writes a skill here — there's no synthesis pipeline yet. `learn` in
Phase 0 only seeds facts (see cli.py); it does not create skills.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.models import Skill


class ProceduralMemory:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                file_path TEXT,
                verification_tier TEXT NOT NULL DEFAULT 'mock',
                success_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def register(self, skill: Skill) -> int:
        """Records skill metadata. Never called by the agent itself in Phase 0 —
        exists so the schema and this class are proven before Phase 2 needs them."""
        cur = self.conn.execute(
            "INSERT INTO skills (name, description, file_path, verification_tier, created_at) "
            "VALUES (?,?,?,?,?)",
            (skill.name, skill.description, skill.file_path, skill.verification_tier, skill.created_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def load(self, name: str) -> Skill | None:
        row = self.conn.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return self._row_to_skill(row)

    def list(self) -> list[Skill]:
        rows = self.conn.execute("SELECT * FROM skills ORDER BY id").fetchall()
        return [self._row_to_skill(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        return Skill(id=row["id"], name=row["name"], description=row["description"],
                    file_path=row["file_path"], verification_tier=row["verification_tier"],
                    success_count=row["success_count"], fail_count=row["fail_count"],
                    created_at=row["created_at"])
