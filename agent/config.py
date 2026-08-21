"""Central configuration: paths, constants, thresholds."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_DATA_DIR = PROJECT_ROOT / "seed_data"
SKILLS_DIR = DATA_DIR / "skills"  # Phase 2+, directory reserved now

SEMANTIC_DB = DATA_DIR / "semantic.db"
EPISODIC_DB = DATA_DIR / "episodic.db"
PROCEDURAL_DB = DATA_DIR / "procedural.db"

SEED_VERSION = 1  # bump when seed_data/facts.json changes shape, to force a reseed

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model name (384-dim)
EMBEDDING_DIM = 384

# Confidence gate thresholds. Simplified from ARCHITECTURE.md's Stage-1/Stage-2
# design: there's no Stage-2 LLM discriminator in Phase 0 (MockBrain can't
# discriminate anything), so this is a direct two-threshold gate on the
# blended retrieval score.
CONFIDENT_THRESHOLD = 0.80
TENTATIVE_THRESHOLD = 0.65

# Hybrid retrieval blend: cosine (semantic) + FTS5 BM25 (keyword)
COSINE_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

EPISODIC_RETENTION_DAYS = 90


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
