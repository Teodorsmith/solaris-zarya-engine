"""Central configuration: paths, constants, thresholds."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_DATA_DIR = PROJECT_ROOT / "seed_data"
SKILLS_DIR = DATA_DIR / "skills"  # Phase 2+, directory reserved now

SEMANTIC_DB = DATA_DIR / "semantic.db"
EPISODIC_DB = DATA_DIR / "episodic.db"
PROCEDURAL_DB = DATA_DIR / "procedural.db"
PROJECTS_DB = DATA_DIR / "projects.db"
GOALS_DB = DATA_DIR / "goals.db"
ACTIVE_TASK_JSON = DATA_DIR / "active_task.json"
STATE_MANIFEST_JSON = DATA_DIR / "state_manifest.json"

SEED_VERSION = 1  # bump when seed_data/facts.json changes shape, to force a reseed

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model name (384-dim)
EMBEDDING_DIM = 384

# Project Indexing Filters
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB
IGNORED_DIRS = {".git", "__pycache__", ".venv", "node_modules", "data", ".pytest_cache", ".idea", ".vscode"}
IGNORED_EXTS = {
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".o", ".a", 
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".sqlite", ".db", ".db-wal", ".db-shm", ".zip", ".tar", ".gz"
}

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

def load_env() -> None:
    """Minimal zero-dependency .env loader."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    import os
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val:
            os.environ[key] = val


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
