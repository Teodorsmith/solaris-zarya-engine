"""Tier 4: Project Codebase Memory (projects.db)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.config import IGNORED_DIRS, IGNORED_EXTS, MAX_FILE_SIZE_BYTES
from agent.brains.base import BaseBrain
from agent.brains.mock_brain import MockBrain
from agent.memory.embeddings import EmbeddingEngine
from agent.models import Project, ProjectFile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _heuristic_summary(path: Path, content: str) -> str:
    """Local fallback summary if AI brain is unavailable."""
    ext = path.suffix.lower()
    
    if ext == ".py":
        classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
        funcs = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
        elements = classes[:3] + funcs[:3]
        desc = "Python module"
        if elements:
            desc += f" containing {', '.join(elements)}"
        return f"{path.name} is a {desc}."
    
    if ext in {".md", ".txt"}:
        first_line = content.splitlines()[0] if content else ""
        return f"Documentation file {path.name} starting with '{first_line[:50]}'"
        
    return f"A {ext or 'text'} file named {path.name}."


class ProjectMemory:
    def __init__(self, db_path: str | Path, embedder: EmbeddingEngine):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self.embedder = embedder
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    root_path TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, path),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pf_project ON project_files(project_id)")

    def get_or_create_project(self, root_path: Path) -> Project:
        path_str = str(root_path.resolve().as_posix())
        name = root_path.name
        
        with self.conn:
            row = self.conn.execute("SELECT * FROM projects WHERE root_path=?", (path_str,)).fetchone()
            if row:
                return Project(**dict(row))
                
            cur = self.conn.execute(
                "INSERT INTO projects (name, root_path, created_at, updated_at) VALUES (?,?,?,?)",
                (name, path_str, _now(), _now())
            )
            project_id = cur.lastrowid
            
        return Project(id=project_id, name=name, root_path=path_str)

    def _is_ignorable(self, path: Path) -> bool:
        if path.name.startswith("."):
            return True
        if any(part in IGNORED_DIRS for part in path.parts):
            return True
        if path.suffix.lower() in IGNORED_EXTS:
            return True
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return True
        return False

    def index_workspace(self, directory: Path, brain: BaseBrain) -> int:
        """Scan workspace, hash files, and summarize new/changed files."""
        project = self.get_or_create_project(directory)
        
        # Load existing hashes to minimize re-indexing
        existing = {}
        for row in self.conn.execute("SELECT path, sha256_hash FROM project_files WHERE project_id=?", (project.id,)):
            existing[row["path"]] = row["sha256_hash"]
            
        indexed_count = 0
        current_paths = set()
        
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            if self._is_ignorable(root_path):
                continue
                
            for file in files:
                file_path = root_path / file
                if self._is_ignorable(file_path):
                    continue
                    
                rel_path = file_path.relative_to(directory).as_posix()
                current_paths.add(rel_path)
                
                file_hash = _hash_file(file_path)
                
                # Skip if unmodified
                if rel_path in existing and existing[rel_path] == file_hash:
                    continue
                    
                # New or modified file -> summarize
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue  # skip binaries not caught by ext
                    
                summary = self._summarize_file(rel_path, content, brain)
                vec = self.embedder.embed(f"File {rel_path}: {summary}")
                
                with self.conn:
                    self.conn.execute(
                        """
                        INSERT INTO project_files (project_id, path, sha256_hash, summary, embedding, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(project_id, path) DO UPDATE SET 
                            sha256_hash=excluded.sha256_hash,
                            summary=excluded.summary,
                            embedding=excluded.embedding,
                            updated_at=excluded.updated_at
                        """,
                        (project.id, rel_path, file_hash, summary, json.dumps(vec), _now(), _now())
                    )
                indexed_count += 1

        # Delete files that no longer exist
        missing_paths = set(existing.keys()) - current_paths
        if missing_paths:
            with self.conn:
                self.conn.executemany(
                    "DELETE FROM project_files WHERE project_id=? AND path=?",
                    [(project.id, p) for p in missing_paths]
                )

        return indexed_count

    def _summarize_file(self, rel_path: str, content: str, brain: BaseBrain) -> str:
        """Use Gemini if available, else local heuristic."""
        if isinstance(brain, MockBrain):
            return _heuristic_summary(Path(rel_path), content)
            
        prompt = (
            f"Write a single, concise sentence explaining the role of the file `{rel_path}` based on its contents.\n"
            "Do not include quotes or conversational filler. Be extremely brief.\n\n"
            f"```\n{content[:4000]}\n```"
        )
        try:
            summary = brain.generate(prompt).strip()
            # Clean up potential markdown formatting from LLM
            if summary.startswith("`") and summary.endswith("`"):
                summary = summary.strip("`")
            return summary
        except Exception:
            # Fallback if API fails
            return _heuristic_summary(Path(rel_path), content)

    def search(self, query: str, top_k: int = 5) -> list[ProjectFile]:
        """Dense cosine similarity search on project files."""
        rows = self.conn.execute("SELECT * FROM project_files").fetchall()
        if not rows:
            return []
            
        qvec = self.embedder.embed(query)
        scored = [
            (
                row,
                EmbeddingEngine.similarity(qvec, json.loads(row["embedding"]))
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        
        results = []
        for row, _score in scored[:top_k]:
            results.append(
                ProjectFile(
                    id=row["id"], project_id=row["project_id"], path=row["path"], 
                    sha256_hash=row["sha256_hash"], summary=row["summary"], 
                    created_at=row["created_at"], updated_at=row["updated_at"]
                )
            )
        return results

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM project_files").fetchone()[0]
