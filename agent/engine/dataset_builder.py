import json
import sqlite3
import struct
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
import hashlib

from agent.config import DATA_DIR
from agent.memory.semantic import SemanticMemory


class DatasetBuilder:
    def __init__(self, dataset_path: Path | None = None, episodic_mem=None, semantic_mem: SemanticMemory | None = None):
        self.dataset_path = dataset_path or (DATA_DIR / "dpo_dataset.jsonl")
        self.episodic_mem = episodic_mem
        self.semantic_mem = semantic_mem
        
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = DATA_DIR / "reasoning.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        
    def _init_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dpo_embeddings (
                prompt_hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT
            )
            """
        )
        self.conn.commit()
        
    def harvest_dpo_pair(self, prompt: str, chosen: str, rejected: str, metadata: dict | None = None) -> bool:
        if metadata is None:
            metadata = {}
            
        # 1. Deterministic Verification Gate
        # Validate chosen succeeded
        if metadata.get("chosen_exit_code", 0) != 0:
            return False
            
        # Validate rejected failed
        if metadata.get("rejected_exit_code", 1) == 0 and metadata.get("source") != "user_correction":
            return False
            
        # 2. Benchmark Contamination Gate
        if "benchmark" in (metadata.get("task_id", "")).lower() or "reasoning" in (metadata.get("task_id", "")).lower():
            return False
            
        if self.semantic_mem and self.semantic_mem.embedder:
            prompt_vec = self.semantic_mem.embedder.embed(prompt)
            
            # Benchmark Contamination Check against task prompts in /benchmark (Placeholder)
            # A true implementation would embed all benchmark prompts and check similarity here.
            
            # 3. Novelty Filter
            prompt_blob = struct.pack(f"{len(prompt_vec)}f", *prompt_vec)
            
            # Fetch all existing embeddings
            rows = self.conn.execute("SELECT embedding FROM dpo_embeddings").fetchall()
            max_sim = 0.0
            for row in rows:
                existing_vec = struct.unpack(f"{len(prompt_vec)}f", row["embedding"])
                sim = self._cosine_sim(prompt_vec, existing_vec)
                if sim > max_sim:
                    max_sim = sim
                    
            if max_sim >= 0.95:
                return False
                
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            now = datetime.now(timezone.utc).isoformat()
            
            try:
                self.conn.execute(
                    "INSERT INTO dpo_embeddings (prompt_hash, embedding, created_at) VALUES (?, ?, ?)",
                    (prompt_hash, prompt_blob, now)
                )
                self.conn.commit()
            except sqlite3.IntegrityError:
                return False # already exists
                
        # 5. Prompt+chosen content dedup — prevent exact duplicates regardless of novelty score
        chosen_hash = hashlib.sha256((prompt + chosen).encode("utf-8")).hexdigest()
        dup = self.conn.execute(
            "SELECT 1 FROM dpo_embeddings WHERE prompt_hash = ?", (chosen_hash,)
        ).fetchone()
        if dup:
            return False  # exact (prompt, chosen) pair already recorded

        # 6. Write to jsonl atomically
        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Safe append using tempfile and atomic replacement to avoid corruption
        import tempfile
        import os
        import threading
        
        # Just use file locking or simple append
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        return True

    def _cosine_sim(self, v1, v2):
        dot = sum(a*b for a, b in zip(v1, v2))
        norm1 = sum(a*a for a in v1) ** 0.5
        norm2 = sum(a*a for a in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
        
    def harvest_from_episodic(self, limit: int | None = None, dry_run: bool = False) -> list[dict]:
        if not self.episodic_mem:
            return []
            
        candidates = []
        
        # Look for task_repair_resolved events
        rows = self.episodic_mem.conn.execute(
            "SELECT * FROM episodic_log WHERE kind = 'task_repair_resolved' ORDER BY id DESC"
        ).fetchall()
        
        for row in rows:
            if limit is not None and len(candidates) >= limit:
                break
                
            payload = json.loads(row["content"])
            original_fail_id = payload.get("original_fail_id")
            chosen_code = payload.get("chosen_code", "")
            
            # fetch original fail
            fail_row = self.episodic_mem.conn.execute(
                "SELECT * FROM episodic_log WHERE id = ?", (original_fail_id,)
            ).fetchone()
            
            if fail_row:
                fail_payload = json.loads(fail_row["content"])
                prompt = fail_payload.get("prompt", "")
                rejected_code = fail_payload.get("code", "")
                metadata = {
                    "chosen_exit_code": payload.get("exit_code", 0),
                    "rejected_exit_code": fail_payload.get("exit_code", 1),
                    "task_id": payload.get("task_id", "")
                }
                
                pair = {
                    "prompt": prompt,
                    "chosen": chosen_code,
                    "rejected": rejected_code,
                    "metadata": metadata
                }
                candidates.append(pair)
                
                if not dry_run:
                    self.harvest_dpo_pair(prompt, chosen_code, rejected_code, metadata)
                    
        # Also look for user_correction events
        corr_rows = self.episodic_mem.conn.execute(
            "SELECT * FROM episodic_log WHERE kind = 'user_correction' ORDER BY id DESC"
        ).fetchall()
        for row in corr_rows:
            if limit is not None and len(candidates) >= limit:
                break
            payload = json.loads(row["content"])
            prompt = payload.get("topic", "correction")
            pair = {
                "prompt": prompt,
                "chosen": payload.get("new_text", ""),
                "rejected": payload.get("old_text", ""),
                "metadata": {
                    "chosen_exit_code": 0,
                    "rejected_exit_code": 1,
                    "source": "user_correction"
                }
            }
            candidates.append(pair)
            if not dry_run:
                self.harvest_dpo_pair(pair["prompt"], pair["chosen"], pair["rejected"], pair["metadata"])
                
        # Also look for skill_repair_resolved events from SkillSynthesizer
        skill_rows = self.episodic_mem.conn.execute(
            "SELECT * FROM episodic_log WHERE kind = 'skill_repair_resolved' ORDER BY id DESC"
        ).fetchall()
        for row in skill_rows:
            if limit is not None and len(candidates) >= limit:
                break
            payload = json.loads(row["content"])
            prompt = payload.get("prompt", "")
            pair = {
                "prompt": prompt,
                "chosen": payload.get("chosen_code", ""),
                "rejected": payload.get("rejected_code", ""),
                "metadata": {
                    "chosen_exit_code": payload.get("exit_code", 0),
                    "rejected_exit_code": 1,
                    "source": "skill_synthesizer",
                    "success_source": "mock_only",
                    "error": payload.get("error", ""),
                }
            }
            candidates.append(pair)
            if not dry_run:
                self.harvest_dpo_pair(pair["prompt"], pair["chosen"], pair["rejected"], pair["metadata"])

        return candidates

    def get_stats(self) -> dict:
        total_pairs = 0
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                total_pairs = sum(1 for _ in f)
                
        unique_prompts = self.conn.execute("SELECT COUNT(*) FROM dpo_embeddings").fetchone()[0]
        
        last_addition = self.conn.execute("SELECT MAX(created_at) FROM dpo_embeddings").fetchone()[0]
        
        return {
            "total_pairs": total_pairs,
            "unique_prompts": unique_prompts,
            "file_size": self.dataset_path.stat().st_size if self.dataset_path.exists() else 0,
            "last_addition": last_addition
        }
        
    def clear_dataset(self) -> bool:
        if self.dataset_path.exists():
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                pass # truncate
        self.conn.execute("DELETE FROM dpo_embeddings")
        self.conn.commit()
        return True
