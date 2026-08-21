"""Meta-Cognitive Substrate: Goal DAG storage."""
from __future__ import annotations
import sqlite3
import json
from pathlib import Path

from agent.models import Goal

class GoalMemory:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                parent_id TEXT,
                dependencies_json TEXT NOT NULL,
                status TEXT NOT NULL,
                completion_criteria TEXT NOT NULL,
                required_tier INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
        self.conn.commit()

    def register(self, goal: Goal) -> None:
        deps_json = json.dumps(goal.dependencies)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO goals 
            (id, description, parent_id, dependencies_json, status, completion_criteria, required_tier, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (goal.id, goal.description, goal.parent_id, deps_json, goal.status, goal.completion_criteria, goal.required_tier, goal.created_at)
        )
        self.conn.commit()

    def update_status(self, goal_id: str, status: str) -> None:
        self.conn.execute("UPDATE goals SET status=? WHERE id=?", (status, goal_id))
        self.conn.commit()

    def get_goal(self, goal_id: str) -> Goal | None:
        row = self.conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        if not row:
            return None
        return self._row_to_goal(row)

    def get_pending_goals(self) -> list[Goal]:
        rows = self.conn.execute("SELECT * FROM goals WHERE status='PENDING'").fetchall()
        return [self._row_to_goal(r) for r in rows]
        
    def get_all_goals(self) -> list[Goal]:
        rows = self.conn.execute("SELECT * FROM goals").fetchall()
        return [self._row_to_goal(r) for r in rows]

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            description=row["description"],
            parent_id=row["parent_id"],
            dependencies=json.loads(row["dependencies_json"]),
            status=row["status"],
            completion_criteria=row["completion_criteria"],
            required_tier=row["required_tier"],
            created_at=row["created_at"]
        )
