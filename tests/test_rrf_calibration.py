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

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from agent.memory.semantic import SemanticMemory
from agent.models import Fact
from agent.memory.embeddings import EmbeddingEngine

DATA_DIR = Path(__file__).parent / "data"
QUERIES_FILE = DATA_DIR / "rrf_queries.json"

@pytest.fixture
def temp_semantic_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    engine = EmbeddingEngine(force_fallback=False)
    mem = SemanticMemory(path, embedder=engine)
    
    yield mem
    
    mem.conn.close()
    try:
        os.remove(path)
    except OSError:
        pass

def test_rrf_calibration(temp_semantic_db):
    """
    Evaluates formal RRF blending (dense + sparse) on 25 diverse queries.
    Asserts Precision@5 >= 0.90.
    """
    assert QUERIES_FILE.exists(), f"{QUERIES_FILE} not found."
    
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    assert len(queries) == 25, f"Expected 25 queries, found {len(queries)}"
    
    # 1. Seed 100 distractors
    for i in range(1, 101):
        distractor = Fact(
            topic="distractor",
            text=f"This is a randomly generated distractor fact {i} about programming, artificial intelligence, and databases. It contains keywords like SQLite, Python, and neural networks but is mostly noise.",
            source_type="seed"
        )
        temp_semantic_db.add_fact(distractor)
        
    # 2. Seed the 25 target facts and map query to target concept for verification
    target_mapping = {}
    for i, q in enumerate(queries):
        target_fact = Fact(
            topic="calibration",
            text=q["fact"],
            source_type="seed"
        )
        temp_semantic_db.add_fact(target_fact)
        target_mapping[q["query"]] = q["fact"]
        
    # 3. Evaluate Precision@5
    hits = 0
    total = len(queries)
    
    for query_obj in queries:
        q_text = query_obj["query"]
        target_text = target_mapping[q_text]
        
        # RRF is integrated into search() which calls _ranked()
        results = temp_semantic_db.search(q_text, top_k=5)
        
        found = any(r.text == target_text for r in results)
        if found:
            hits += 1
            
    precision_at_5 = hits / total
    
    assert precision_at_5 >= 0.90, f"Precision@5 was {precision_at_5:.2f}, expected >= 0.90"

