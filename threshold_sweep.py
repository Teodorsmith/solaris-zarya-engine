# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
import re
import sqlite3

from fastembed import TextEmbedding

# 1. Initialize Embedding Model
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_one(text: str) -> list[float]:
    vec = list(model.embed([text]))[0]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def cosine(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


# 2. Benchmark Corpus (Standard Git & GitHub Knowledge Base)
facts = [
    "To create a release with GitHub CLI: gh release create <tag> --title <title> --notes <notes>",
    "To view release details in GitHub CLI: gh release view <tag>",
    "To open a pull request with GitHub CLI: gh pr create --title <title> --body <body>",
    "git stash pop applies the top stashed changes and immediately removes them from the stash list",
    "git stash apply applies the top stashed changes but keeps them preserved in the stash list",
    "git rebase rewrites branch history by moving the base of your branch to the target branch",
    "git merge combines changes from one branch into the current branch, creating a merge commit",
]
fact_vectors = [embed_one(f) for f in facts]

# Set up SQLite FTS5 for Hybrid Search
conn = sqlite3.connect(":memory:")
conn.execute("CREATE VIRTUAL TABLE facts_fts USING fts5(statement)")
for f in facts:
    conn.execute("INSERT INTO facts_fts(statement) VALUES (?)", (f,))


def query_corpus(
    query: str, dense_w: float = 0.6, sparse_w: float = 0.4
) -> tuple[float, float, float, str]:
    q_vec = embed_one(query)
    dense_scores = [cosine(q_vec, f_vec) for f_vec in fact_vectors]

    terms = [re.sub(r"[^a-zA-Z0-9_]", "", t) for t in query.split() if len(t) > 2]
    clean_query = " OR ".join(terms) if terms else query

    try:
        fts_cursor = conn.execute(
            "SELECT rowid, bm25(facts_fts) FROM facts_fts WHERE facts_fts MATCH ?",
            (clean_query,),
        )
        fts_results = {row[0] - 1: -row[1] for row in fts_cursor.fetchall()}
    except Exception:
        fts_results = {}

    max_bm25 = max(fts_results.values()) if fts_results else 1.0
    min_bm25 = min(fts_results.values()) if fts_results else 0.0
    bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

    best_hybrid = -1.0
    best_dense = -1.0
    best_sparse = -1.0
    best_fact = ""

    for i, fact in enumerate(facts):
        d = dense_scores[i]
        raw_s = fts_results.get(i, min_bm25)
        s = (raw_s - min_bm25) / bm25_range if fts_results else 0.0
        h = (dense_w * d) + (sparse_w * s)
        if h > best_hybrid:
            best_hybrid = h
            best_dense = d
            best_sparse = s
            best_fact = fact

    return best_dense, best_sparse, best_hybrid, best_fact


print("=" * 110)
print("EXPERIMENT 1: THE CRITICAL UNMEASURED CASE — RELATED-BUT-UNKNOWN QUERIES (M29)")
print(
    "Facts present: Release create, Release view, PR create, Stash pop/apply, Rebase, Merge"
)
print("=" * 110)

related_unknown_queries = [
    (
        "What is the maximum file upload size for GitHub releases?",
        "GitHub Release Limits",
    ),
    (
        "How do I delete an asset from a GitHub release using CLI?",
        "GitHub Release Asset Delete",
    ),
    ("What is the rate limit for GitHub CLI requests?", "GitHub CLI Rate Limits"),
    ("How do I resolve a 3-way merge conflict during rebase?", "Git Rebase Conflict"),
    ("How do I drop a specific stash by index number?", "Git Stash Drop"),
]

print(
    f"{'QUERY TEXT':<60} | {'DENSE':<7} | {'SPARSE':<7} | {'HYBRID':<7} | {'LANDS IN ZONE (Hybrid)'}"
)
print("-" * 110)
for q, desc in related_unknown_queries:
    d, s, h, f = query_corpus(q)
    zone = (
        "PASS (>=0.80) [DANGER]"
        if h >= 0.80
        else ("GREY [0.65, 0.80)" if h >= 0.65 else "REJECT (<0.65)")
    )
    print(f"{q[:60]:<60} | {d:.4f} | {s:.4f} | {h:.4f} | {zone}")

print("\n" * 1)
print("=" * 110)
print(
    "EXPERIMENT 2: POP VS APPLY PHRASING SPREAD (DEDUPLICATION THRESHOLD 0.95 STRESS TEST)"
)
print("=" * 110)

pop_phrasings = [
    "git stash pop applies the top stashed changes and immediately removes them from the stash list",
    "git stash pop restores your latest stash and deletes it from storage",
    "git stash pop applies and drops the latest stash entry",
    "run git stash pop to retrieve and clear the topmost stashed change",
]

apply_phrasings = [
    "git stash apply applies the top stashed changes but keeps them preserved in the stash list",
    "git stash apply restores your latest stash without deleting it from storage",
    "git stash apply applies stashed changes while preserving the stash entry",
    "run git stash apply to retrieve the topmost stashed change and keep it",
]

max_cross_sim = 0.0
min_self_sim = 1.0

print(f"{'POP PHRASING':<45} | {'APPLY PHRASING':<45} | {'SIMILARITY'}")
print("-" * 110)
for p in pop_phrasings:
    v_p = embed_one(p)
    for a in apply_phrasings:
        v_a = embed_one(a)
        sim = cosine(v_p, v_a)
        max_cross_sim = max(max_cross_sim, sim)
        if sim >= 0.90:
            print(
                f"{p[:45]:<45} | {a[:45]:<45} | {sim:.4f} {'[HIGH!]' if sim >= 0.93 else ''}"
            )

print(
    f"\nMax cross-similarity between pop & apply across all phrasing combinations: {max_cross_sim:.4f}"
)
print(f"Safety margin below 0.95 dedup threshold: {0.95 - max_cross_sim:.4f}")

print("\n" * 1)
print("=" * 110)
print("EXPERIMENT 3: 20-ITEM LABELED DATASET — THRESHOLD SWEEP & ZONE DISTRIBUTION")
print("=" * 110)

labeled_test_set = [
    # True Hits (Should PASS >= 0.80)
    ("TRUE HIT", "How do I create a release using the GitHub CLI?", True),
    ("TRUE HIT", "gh release create command syntax", True),
    ("TRUE HIT", "How to make a pull request with gh cli", True),
    ("TRUE HIT", "How do I pop a git stash?", True),
    ("TRUE HIT", "How to apply git stash and keep it?", True),
    ("TRUE HIT", "What does git rebase do?", True),
    ("TRUE HIT", "How to merge a branch in git", True),
    # Related-but-Unknown (Should land in GREY [0.65, 0.80) -> Trigger Closed-World or Discriminator)
    ("UNKNOWN", "What is the max file size for a GitHub release?", False),
    ("UNKNOWN", "How to configure GitHub CLI OAuth token manually?", False),
    ("UNKNOWN", "How to rebase interactively with squash?", False),
    ("UNKNOWN", "How to view the reflog in git?", False),
    ("UNKNOWN", "How to cherry-pick a commit in git?", False),
    # Opposing / Syntax Distinctions (Should match specific fact via hybrid)
    ("OPPOSING", "Retrieve stash and remove it from stash list", True),
    ("OPPOSING", "Retrieve stash and retain it in stash list", True),
    # Completely Unrelated (Should REJECT < 0.65)
    ("UNRELATED", "Best recipe for sourdough pizza crust", False),
    ("UNRELATED", "How does quantum entanglement work in physics?", False),
    ("UNRELATED", "Current price of Ethereum in USD", False),
    ("UNRELATED", "How to train a dog to sit on command", False),
    ("UNRELATED", "What are the symptoms of seasonal allergies?", False),
    ("UNRELATED", "How to fix a leaky faucet in the kitchen", False),
]

print(
    f"{'CATEGORY':<12} | {'QUERY TEXT':<45} | {'DENSE':<7} | {'SPARSE':<7} | {'HYBRID':<7} | {'CLASSIFICATION'}"
)
print("-" * 110)

dense_scores_all = []
hybrid_scores_all = []

for cat, q_text, is_answerable in labeled_test_set:
    d, s, h, f = query_corpus(q_text)
    dense_scores_all.append((cat, is_answerable, d))
    hybrid_scores_all.append((cat, is_answerable, h))

    # Classification zone with hybrid
    if h >= 0.80:
        c_zone = "PASS (>=0.80)"
    elif h >= 0.65:
        c_zone = "GREY [0.65, 0.80)"
    else:
        c_zone = "REJECT (<0.65)"
    print(f"{cat:<12} | {q_text[:45]:<45} | {d:.4f} | {s:.4f} | {h:.4f} | {c_zone}")

# 4. Precision / Recall Sweep across thresholds
print("\n" * 1)
print("=" * 110)
print("THRESHOLD SWEEP: PASS GATE (Evaluating True Hits Precision & False-Pass Rate)")
print("=" * 110)

print(
    f"{'PASS THRESHOLD':<15} | {'TRUE HITS PASS RATE':<22} | {'UNKNOWN QUERIES LEAKING AS PASS (FALSE POSITIVES)':<45}"
)
print("-" * 110)

for t in [0.70, 0.73, 0.75, 0.78, 0.80, 0.82, 0.85]:
    true_hits = [
        h for cat, is_ans, h in hybrid_scores_all if cat in ("TRUE HIT", "OPPOSING")
    ]
    unknowns = [
        h for cat, is_ans, h in hybrid_scores_all if cat in ("UNKNOWN", "UNRELATED")
    ]

    hit_pass_rate = sum(1 for h in true_hits if h >= t) / len(true_hits) * 100
    unknown_leak_rate = sum(1 for h in unknowns if h >= t) / len(unknowns) * 100
    leaked_count = sum(1 for h in unknowns if h >= t)

    print(
        f"{t:<15.2f} | {hit_pass_rate:<6.1f}% ({sum(1 for h in true_hits if h >= t)}/{len(true_hits)})           | {unknown_leak_rate:<6.1f}% ({leaked_count}/{len(unknowns)} leaked into pass zone)"
    )

print("\n" * 1)
print("=" * 110)
print("SWEEP COMPLETE")
print("=" * 110)
