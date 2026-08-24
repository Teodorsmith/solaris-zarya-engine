# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
        # Add new columns (Phase 4C) if they don't exist
        for col, col_type in [
            ("prompt_hash", "TEXT"),
            ("strategy_label", "TEXT"),
            ("novelty_score", "REAL"),
            ("reasoning_domain", "TEXT"),
            ("outcome_class", "TEXT"),
            ("hypothesis_count", "INTEGER DEFAULT 1"),
        ]:
            try:
                self.conn.execute(f"ALTER TABLE episodic_log ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass # Column already exists
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_trace ON episodic_log(trace_id)")
        self.conn.commit()

    def log_event(self, event: EpisodicLog) -> int:
        cur = self.conn.execute(
            """INSERT INTO episodic_log 
               (trace_id, kind, content, outcome, prompt_hash, strategy_label, 
                novelty_score, reasoning_domain, outcome_class, hypothesis_count, created_at) 
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (event.trace_id, event.kind, event.content, event.outcome,
             event.prompt_hash, event.strategy_label, event.novelty_score,
             event.reasoning_domain, event.outcome_class, event.hypothesis_count,
             event.created_at),
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
        keys = row.keys()
        return EpisodicLog(
            id=row["id"], trace_id=row["trace_id"], kind=row["kind"],
            content=row["content"], outcome=row["outcome"],
            prompt_hash=row["prompt_hash"] if "prompt_hash" in keys else None, 
            strategy_label=row["strategy_label"] if "strategy_label" in keys else None,
            novelty_score=row["novelty_score"] if "novelty_score" in keys else None, 
            reasoning_domain=row["reasoning_domain"] if "reasoning_domain" in keys else None,
            outcome_class=row["outcome_class"] if "outcome_class" in keys else None, 
            hypothesis_count=row["hypothesis_count"] if "hypothesis_count" in keys and row["hypothesis_count"] is not None else 1,
            created_at=row["created_at"]
        )
