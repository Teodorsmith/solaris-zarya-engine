# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
import re
import sqlite3
import ast
from typing import List, Tuple

# 1. Initialize FastEmbed Embedding Engine
print("=" * 70)
print("1. LOADING EMBEDDING MODEL & MEASURING REAL SIMILARITIES")
print("=" * 70)

from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed_one(text: str) -> List[float]:
    vec = list(model.embed([text]))[0]
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

# 2. Benchmark Knowledge Corpus (5 sample facts)
facts = [
    "To create a release with GitHub CLI: gh release create <tag> --title <title> --notes <notes>",
    "To open a pull request with GitHub CLI: gh pr create --title <title> --body <body>",
    "git stash pop applies the top stashed changes and immediately removes them from the stash list",
    "git stash apply applies the top stashed changes but keeps them preserved in the stash list",
    "git rebase rewrites branch history by moving the base of your branch to the target branch"
]

fact_vectors = [embed_one(f) for f in facts]

# 3. Test Queries across the 5 categories
queries = [
    ("EXACT HIT", "How do I create a release using the GitHub CLI?"),
    ("PARAPHRASE", "CLI command to publish a new tag version on GitHub"),
    ("ADJACENT/GREY", "How do I open a pull request on GitHub?"),
    ("OPPOSING CLI 1", "How do I apply a stash and delete it from the stash list?"),
    ("OPPOSING CLI 2", "How do I apply a stash but keep it in the stash list?"),
    ("UNRELATED", "What is the best recipe for homemade pizza dough?")
]

print(f"{'QUERY TYPE':<16} | {'QUERY TEXT':<45} | {'TOP MATCH FACT (TRUNCATED)':<30} | {'COSINE':<7}")
print("-" * 105)

for q_type, q_text in queries:
    q_vec = embed_one(q_text)
    scores = [(cosine_similarity(q_vec, f_vec), facts[i]) for i, f_vec in enumerate(fact_vectors)]
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_fact = scores[0]
    print(f"{q_type:<16} | {q_text[:45]:<45} | {best_fact[:30]:<30}... | {best_score:.4f}")

# 4. Measure CLI Opposition & Dedup similarity
print("\n" * 1)
print("=" * 70)
print("2. MEASURING DEDUPLICATION & OPPOSING COMMAND CONFUSION")
print("=" * 70)

v_pop = embed_one("git stash pop applies stashed changes and deletes them from stash list")
v_apply = embed_one("git stash apply applies stashed changes and keeps them in stash list")
sim_opposing = cosine_similarity(v_pop, v_apply)
print(f"Cosine similarity between 'git stash pop' vs 'git stash apply' facts: {sim_opposing:.4f}")

v_dup1 = embed_one("To create a release with GitHub CLI: gh release create <tag>")
v_dup2 = embed_one("Create a release in GitHub CLI with command: gh release create <tag>")
sim_dup = cosine_similarity(v_dup1, v_dup2)
print(f"Cosine similarity between minor paraphrases of same fact (dedup test): {sim_dup:.4f}")

# 5. Measure Hybrid Search Separation (FTS5 + Cosine)
print("\n" * 1)
print("=" * 70)
print("3. MEASURING HYBRID SEARCH SEPARATION ON OPPOSING COMMANDS")
print("=" * 70)

conn = sqlite3.connect(":memory:")
conn.execute("CREATE VIRTUAL TABLE facts_fts USING fts5(statement)")
for f in facts:
    conn.execute("INSERT INTO facts_fts(statement) VALUES (?)", (f,))

def hybrid_search(query: str, dense_w: float = 0.6, sparse_w: float = 0.4):
    q_vec = embed_one(query)
    # Dense scores
    dense_scores = [cosine_similarity(q_vec, f_vec) for f_vec in fact_vectors]
    
    # Sparse FTS scores
    terms = [re.sub(r'[^a-zA-Z0-9_]', '', t) for t in query.split() if len(t) > 2]
    clean_query = " OR ".join(terms) if terms else query
    
    fts_cursor = conn.execute("SELECT rowid, bm25(facts_fts) FROM facts_fts WHERE facts_fts MATCH ?", (clean_query,))
    fts_results = {row[0] - 1: -row[1] for row in fts_cursor.fetchall()}
    
    max_bm25 = max(fts_results.values()) if fts_results else 1.0
    min_bm25 = min(fts_results.values()) if fts_results else 0.0
    bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

    print(f"\nQuery: '{query}'")
    for i, fact in enumerate(facts):
        d_score = dense_scores[i]
        raw_s = fts_results.get(i, min_bm25)
        s_score = (raw_s - min_bm25) / bm25_range if fts_results else 0.0
        final_score = (dense_w * d_score) + (sparse_w * s_score)
        if "stash" in fact or ("release" in query and "release" in fact):
            print(f"  Fact: {fact[:55]}...")
            print(f"    Dense: {d_score:.4f} | Sparse (norm BM25): {s_score:.4f} | Hybrid: {final_score:.4f}")

hybrid_search("git stash pop command")
hybrid_search("git stash apply command")

# 6. Test Tiered AST Allowlist on Safe vs Malicious code
print("\n" * 1)
print("=" * 70)
print("4. TESTING TIERED AST ALLOWLIST")
print("=" * 70)

safe_code = """
import json
from typing import Dict, Any

def parse_data(raw: str) -> Dict[str, Any]:
    import subprocess
    return json.loads(raw)
"""

malicious_code_global_subprocess = """
import subprocess
subprocess.run(["rm", "-rf", "/"])
"""

malicious_code_blocked_module = """
import socket
s = socket.socket()
"""

malicious_code_dynamic_eval = """
def evil():
    eval("__import__('os').system('del /f C:*')")
"""

TIER_1 = {"json", "re", "math", "typing", "dataclasses", "datetime", "collections", "pathlib", "shlex", "argparse"}
TIER_2 = {"subprocess", "os", "shutil", "requests", "httpx"}
TIER_3 = {"socket", "ctypes", "importlib", "eval", "exec", "compile", "pickle"}

def check_ast(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
        
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            for name in names:
                if name in TIER_3:
                    return False, f"Hard-blocked Tier 3 import: '{name}'"
                if name in TIER_2:
                    if node in tree.body:
                        return False, f"Tier 2 module '{name}' imported at global/module scope (must be function-scoped)"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "__import__"}:
                return False, f"Hard-blocked function call: '{node.func.id}()'"
    return True, "Passed AST Security Check"

for name, code in [("Safe Code", safe_code), 
                   ("Global Subprocess", malicious_code_global_subprocess), 
                   ("Blocked Socket", malicious_code_blocked_module), 
                   ("Dynamic Eval", malicious_code_dynamic_eval)]:
    valid, msg = check_ast(code)
    print(f"{name:<22}: Valid={valid:<5} | {msg}")

print("\n" * 1)
print("=" * 70)
print("BENCHMARK PROBE COMPLETE")
print("=" * 70)
