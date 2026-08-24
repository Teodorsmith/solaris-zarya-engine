# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tier 2.5: Reasoning Memory (data/reasoning.db) -- Mitigation #61.

Sole writer for reasoning_episodes. The agent cannot write to reasoning.db
directly; all writes flow through this module (same write-protection contract
as self_model.py and project.py).

Schema: SHyAOEDRGL tuples capturing the full inferential chain.
Failure-first value: high-novelty failures are prioritised for curriculum replay.
Permanent retention: no TTL. Reasoning episodes are the most valuable long-term asset.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.models import ReasoningEpisode

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS reasoning_episodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT    NOT NULL,
    task_id          TEXT,
    state            TEXT    NOT NULL,
    hypothesis       TEXT    NOT NULL,
    action           TEXT    NOT NULL,
    observation      TEXT    NOT NULL,
    error            TEXT,
    diagnosis        TEXT,
    revised_hypo     TEXT,
    generalized_rule TEXT,
    strategy_label   TEXT,
    reasoning_domain TEXT,
    outcome_class    TEXT    NOT NULL DEFAULT 'success',
    hypothesis_count INTEGER NOT NULL DEFAULT 1,
    verified         INTEGER NOT NULL DEFAULT 0,
    srt_json         TEXT,
    confidence       REAL,
    created_at       TEXT    NOT NULL
);
"""
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_re_trace   ON reasoning_episodes(trace_id);",
    "CREATE INDEX IF NOT EXISTS idx_re_outcome ON reasoning_episodes(outcome_class, verified);",
    "CREATE INDEX IF NOT EXISTS idx_re_domain  ON reasoning_episodes(reasoning_domain, strategy_label);",
    "CREATE INDEX IF NOT EXISTS idx_re_created ON reasoning_episodes(created_at);",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReasoningMemory:
    """Manages data/reasoning.db.  Sole writer -- see module docstring."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self.conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=3000")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                self.conn.execute(idx_sql)

    # ------------------------------------------------------------------
    # Write paths (sole writer -- Mitigation #61)
    # ------------------------------------------------------------------

    def log_episode(self, episode: ReasoningEpisode) -> int:
        """Insert a new reasoning episode and return its row id."""
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO reasoning_episodes
                    (trace_id, task_id, state, hypothesis, action, observation,
                     error, diagnosis, revised_hypo, generalized_rule,
                     strategy_label, reasoning_domain, outcome_class,
                     hypothesis_count, verified, srt_json, confidence, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    episode.trace_id,
                    episode.task_id,
                    episode.state,
                    episode.hypothesis,
                    episode.action,
                    episode.observation,
                    episode.error,
                    episode.diagnosis,
                    episode.revised_hypo,
                    episode.generalized_rule,
                    episode.strategy_label,
                    episode.reasoning_domain,
                    episode.outcome_class,
                    episode.hypothesis_count,
                    int(episode.verified),
                    episode.srt_json,
                    episode.confidence,
                    episode.created_at,
                ),
            )
        row_id = cur.lastrowid
        logger.debug("ReasoningMemory: logged episode id=%s outcome=%s", row_id, episode.outcome_class)
        return row_id

    def mark_verified(self, episode_id: int, srt_json: str) -> None:
        """Set verified=1 and store the SRT JSON.

        Called by the SRT Verifier after a successful logic check.
        The verifier never writes directly -- it returns a VerificationResult
        and the caller invokes this method (M#65 contract).
        """
        with self.conn:
            self.conn.execute(
                "UPDATE reasoning_episodes SET verified=1, srt_json=? WHERE id=?",
                (srt_json, episode_id),
            )
        logger.debug("ReasoningMemory: marked episode %s as verified.", episode_id)

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def get_failures(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[ReasoningEpisode]:
        """Return failure episodes for curriculum replay (highest-value signal)."""
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM reasoning_episodes WHERE outcome_class='failure' "
                "AND reasoning_domain=? ORDER BY created_at DESC LIMIT ?",
                (domain, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM reasoning_episodes WHERE outcome_class='failure' "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_unverified(self, limit: int = 50) -> list[ReasoningEpisode]:
        """Return episodes awaiting SRT verification."""
        rows = self.conn.execute(
            "SELECT * FROM reasoning_episodes WHERE verified=0 "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_by_domain(
        self,
        domain: str,
        strategy: str | None = None,
        limit: int = 20,
    ) -> list[ReasoningEpisode]:
        if strategy:
            rows = self.conn.execute(
                "SELECT * FROM reasoning_episodes WHERE reasoning_domain=? "
                "AND strategy_label=? ORDER BY created_at DESC LIMIT ?",
                (domain, strategy, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM reasoning_episodes WHERE reasoning_domain=? "
                "ORDER BY created_at DESC LIMIT ?",
                (domain, limit),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM reasoning_episodes"
        ).fetchone()[0]

    def count_by_outcome(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT outcome_class, COUNT(*) FROM reasoning_episodes GROUP BY outcome_class"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> ReasoningEpisode:
        d = dict(row)
        d["verified"] = bool(d.get("verified", 0))
        return ReasoningEpisode(**d)
