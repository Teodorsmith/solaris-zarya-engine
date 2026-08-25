# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tier 2: Semantic memory. Facts and passages, hybrid (cosine + keyword)
search, deduplication, and in-place correction.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import sqlite_vec

from agent.config import COSINE_WEIGHT, KEYWORD_WEIGHT
from agent.memory.embeddings import EmbeddingEngine
from agent.models import Fact, Passage

DEDUP_THRESHOLD = 0.95


class SemanticMemory:
    def __init__(self, db_path: str | Path, embedder: EmbeddingEngine):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.embedder = embedder
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_type TEXT NOT NULL,
                topic TEXT,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                topic TEXT,
                source_type TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                embedding float[384]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_passages USING vec0(
                embedding float[384]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                text, content='facts', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TABLE IF NOT EXISTS sources_ingested (
                url TEXT PRIMARY KEY,
                scraped_at TEXT NOT NULL
            );
            """
        )
        # Safe migration: add text_hash column to existing facts tables that lack it
        existing_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        if "text_hash" not in existing_cols:
            self.conn.execute("ALTER TABLE facts ADD COLUMN text_hash TEXT")
        self.conn.commit()

    # ---- facts -----------------------------------------------------------
    def add_fact(self, fact: Fact) -> tuple[bool, int]:
        """Returns (created, id). Dedup logic (in priority order):
        1. SHA-256 exact-text hash match → skip immediately (O(1)).
        2. Cosine near-duplicate (>= DEDUP_THRESHOLD) → bump confidence.
        3. Otherwise insert.
        """
        text_hash = hashlib.sha256(fact.text.encode("utf-8")).hexdigest()
        # 1. Exact-hash dedup
        existing = self.conn.execute(
            "SELECT id FROM facts WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        if existing:
            return False, existing["id"]

        vec = self.embedder.embed(fact.text)
        # 2. Cosine near-dup
        dup = self._nearest_by_cosine(vec)
        if dup and dup[1] > DEDUP_THRESHOLD:
            fid = dup[0]
            self.conn.execute(
                "UPDATE facts SET confidence = MIN(1.0, confidence + 0.05) WHERE id=?",
                (fid,),
            )
            self.conn.commit()
            return False, fid

        cur = self.conn.execute(
            "INSERT INTO facts (text, confidence, source_type, topic, embedding, created_at, text_hash) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                fact.text,
                fact.confidence,
                fact.source_type,
                fact.topic,
                json.dumps(vec),
                fact.created_at,
                text_hash,
            ),
        )
        fid = cur.lastrowid
        vec_blob = struct.pack(f"{len(vec)}f", *vec)
        self.conn.execute("INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)", (fid, vec_blob))
        self.conn.commit()
        return True, fid

    def correct_fact(self, fact_id: int, new_text: str) -> None:
        """User corrections are the highest-authority source: confidence -> 1.0,
        source_type -> user_corrected, re-embedded and re-indexed for search."""
        vec = self.embedder.embed(new_text)
        self.conn.execute(
            "UPDATE facts SET text=?, embedding=?, source_type='user_corrected', confidence=1.0 "
            "WHERE id=?",
            (new_text, json.dumps(vec), fact_id),
        )
        vec_blob = struct.pack(f"{len(vec)}f", *vec)
        self.conn.execute(
            "UPDATE vec_facts SET embedding=? WHERE rowid=?",
            (vec_blob, fact_id)
        )
        self.conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[Fact]:
        return [self._row_to_fact(row) for row, _score in self._ranked(query)[:top_k]]

    def top_score(self, query: str) -> float:
        """Blended score of the single best match — what the retriever's
        confidence gate reads."""
        ranked = self._ranked(query)
        return ranked[0][1] if ranked else 0.0

    def list_all(self) -> list[Fact]:
        rows = self.conn.execute("SELECT * FROM facts ORDER BY id").fetchall()
        return [self._row_to_fact(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def get_facts_by_topic(self, topic: str, limit: int = 20) -> list[Fact]:
        """Returns the most recent facts for the given topic, used by the planner
        to build prior-knowledge context for differential curriculum generation."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE topic = ? ORDER BY id DESC LIMIT ?",
            (topic, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ---- URL ingestion dedup ------------------------------------------------
    @staticmethod
    def _canonicalize_url(url: str) -> str:
        """Strip fragments and trailing slashes for reliable dedup."""
        parsed = urlparse(url.strip())
        # Drop the fragment entirely; normalise path trailing slash
        canonical = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") or "/",
             parsed.params, parsed.query, "")
        )
        return canonical

    def mark_url_ingested(self, url: str) -> None:
        """Record that a URL was successfully scraped so future runs skip it."""
        import datetime
        from datetime import timezone
        canonical = self._canonicalize_url(url)
        self.conn.execute(
            "INSERT OR REPLACE INTO sources_ingested (url, scraped_at) VALUES (?, ?)",
            (canonical, datetime.datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def is_url_ingested(self, url: str) -> bool:
        """Returns True if this URL has been successfully scraped before."""
        canonical = self._canonicalize_url(url)
        row = self.conn.execute(
            "SELECT 1 FROM sources_ingested WHERE url = ?", (canonical,)
        ).fetchone()
        return row is not None

    # ---- passages ----------------------------------------------------------
    def add_passage(self, passage: Passage) -> int:
        vec = self.embedder.embed(passage.text)
        cur = self.conn.execute(
            "INSERT INTO passages (text, topic, source_type, embedding, created_at) VALUES (?,?,?,?,?)",
            (
                passage.text,
                passage.topic,
                passage.source_type,
                json.dumps(vec),
                passage.created_at,
            ),
        )
        pid = cur.lastrowid
        vec_blob = struct.pack(f"{len(vec)}f", *vec)
        self.conn.execute("INSERT INTO vec_passages(rowid, embedding) VALUES (?, ?)", (pid, vec_blob))
        self.conn.commit()
        return pid

    def get_passage(self, passage_id: int) -> Passage | None:
        row = self.conn.execute(
            "SELECT * FROM passages WHERE id=?", (passage_id,)
        ).fetchone()
        if not row:
            return None
        return Passage(
            id=row["id"],
            text=row["text"],
            topic=row["topic"],
            source_type=row["source_type"],
            created_at=row["created_at"],
        )

    # ---- internals -----------------------------------------------------------
    def _ranked(self, query: str) -> list[tuple[sqlite3.Row, float]]:
        qvec = self.embedder.embed(query)
        kw = self._keyword_scores(query)
        
        vec_blob = struct.pack(f"{len(qvec)}f", *qvec)
        
        # We fetch all matching distances from sqlite-vec and rank
        # sqlite-vec uses 'distance' which is 1 - cosine_similarity roughly, or L2 distance
        # Actually, sqlite-vec `distance` is cosine distance (1 - cosine) if vectors are normalized.
        # So cosine similarity = 1.0 - distance
        rows = self.conn.execute(
            "SELECT f.*, v.distance "
            "FROM facts f "
            "JOIN vec_facts v ON f.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = 100 "
            "ORDER BY v.distance ASC", (vec_blob,)
        ).fetchall()
        
        if not rows:
            return []
            
        scored = [
            (
                row,
                COSINE_WEIGHT * (1.0 - (row["distance"] ** 2) / 2.0)
                + KEYWORD_WEIGHT * kw.get(row["id"], 0.0),
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    @staticmethod
    def _fts_or_query(query: str) -> str:
        """FTS5's default bareword syntax ANDs every term together, which
        makes natural-language queries match almost nothing (a query with
        one word not in any fact returns zero rows, even if the other
        eleven words are a great match). Quote each token as a literal and
        OR them instead, so any shared term counts, ranked by bm25 — bm25's
        own IDF weighting already discounts common words like "the"."""
        tokens = re.findall(r"\w+", query.lower())
        return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)

    def _keyword_scores(self, query: str) -> dict[int, float]:
        fts_query = self._fts_or_query(query)
        if not fts_query:
            return {}
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(facts_fts) AS score FROM facts_fts WHERE facts_fts MATCH ?",
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}  # belt-and-suspenders — shouldn't trigger now tokens are quoted literals
        if not rows:
            return {}
        scores = {r["rowid"]: r["score"] for r in rows}
        if len(scores) == 1:
            # A single match can't be min-max normalized meaningfully —
            # it's the best (and only) one, so score it as such.
            return {next(iter(scores)): 1.0}
        # bm25() is lower-is-better; min-max normalize with the sign flipped.
        lo, hi = min(scores.values()), max(scores.values())
        spread = hi - lo
        if spread == 0:
            return {rid: 1.0 for rid in scores}
        return {rid: (hi - s) / spread for rid, s in scores.items()}

    def _nearest_by_cosine(self, vec: list[float]) -> tuple[int, float] | None:
        vec_blob = struct.pack(f"{len(vec)}f", *vec)
        row = self.conn.execute(
            "SELECT rowid, distance FROM vec_facts WHERE embedding MATCH ? ORDER BY distance ASC LIMIT 1",
            (vec_blob,)
        ).fetchone()
        
        if not row:
            return None
            
        # sqlite-vec distance is L2 distance for vec0 queries. For normalized vectors, L2^2 = 2 - 2 * cosine.
        # So cosine similarity = 1.0 - (L2^2) / 2
        return row["rowid"], 1.0 - (row["distance"] ** 2) / 2.0

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            text=row["text"],
            confidence=row["confidence"],
            source_type=row["source_type"],
            topic=row["topic"],
            created_at=row["created_at"],
        )
