"""Tier 1: Episodic memory. Chronological interaction log with retention pruning."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.config import EPISODIC_RETENTION_DAYS
from agent.models import EpisodicLog


class EpisodicMemory:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_trace ON episodic_log(trace_id)")
        self.conn.commit()

    def log_event(self, event: EpisodicLog) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodic_log (trace_id, kind, content, outcome, created_at) VALUES (?,?,?,?,?)",
            (event.trace_id, event.kind, event.content, event.outcome, event.created_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_trace(self, trace_id: str) -> list[EpisodicLog]:
        rows = self.conn.execute(
            "SELECT * FROM episodic_log WHERE trace_id=? ORDER BY id", (trace_id,)
        ).fetchall()
        return [self._row_to_log(r) for r in rows]

    def recent(self, n: int = 20) -> list[EpisodicLog]:
        rows = self.conn.execute("SELECT * FROM episodic_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [self._row_to_log(r) for r in rows]

    def prune_old(self, days: int = EPISODIC_RETENTION_DAYS) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self.conn.execute("DELETE FROM episodic_log WHERE created_at < ?", (cutoff,))
        self.conn.commit()
        return cur.rowcount

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodic_log").fetchone()[0]

    def _row_to_log(self, row: sqlite3.Row) -> EpisodicLog:
        return EpisodicLog(id=row["id"], trace_id=row["trace_id"], kind=row["kind"],
                           content=row["content"], outcome=row["outcome"], created_at=row["created_at"])
