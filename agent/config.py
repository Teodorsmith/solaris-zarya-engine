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

# Phase 4A: Self-Model
SELF_MODEL_JSON = DATA_DIR / "self_model.json"
SELF_MODEL_BAK_JSON = DATA_DIR / "self_model.bak.json"

# Phase 4A: Heartbeat Daemon (Mitigations #41, #46)
HEARTBEAT_INTERVAL_SECS = 900  # 15-min idle check; no-op if nothing actionable
HEARTBEAT_MAX_PER_HOUR = 3  # hard ceiling on autonomous actions per rolling hour
HEARTBEAT_DAILY_CALL_CAP = 50  # reserved for future LLM-calling background tasks
STALE_FACT_DAYS = 180  # flag facts older than this for review

# Phase 4B: Reasoning Memory (Mitigation #61)
REASONING_DB = DATA_DIR / "reasoning.db"
REASONING_SUITE_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "reasoning_suite"
)

# Phase 4B: Lateral Critic (Mitigation #63)
CRITIC_SIMILARITY_THRESHOLD = 0.75  # cosine; below = divergent
CRITIC_BRAIN_B_TEMPERATURE = 0.85  # fallback temperature for single-provider mode

# Phase 4B: ZPD Benchmark (Mitigation #66)
ZPD_CATEGORIES = [
    "decomposition",
    "hypothesis_testing",
    "causal_reasoning",
    "counterexample_gen",
    "planning",
    "adversarial",
]
ZPD_MAX_ROUNDS = 5
ZPD_DIFFICULTY_MIN = 1
ZPD_DIFFICULTY_MAX = 5

SEED_VERSION = 1  # bump when seed_data/facts.json changes shape, to force a reseed

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model name (384-dim)
EMBEDDING_DIM = 384

# Project Indexing Filters
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "data",
    ".pytest_cache",
    ".idea",
    ".vscode",
}
IGNORED_EXTS = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".sqlite",
    ".db",
    ".db-wal",
    ".db-shm",
    ".zip",
    ".tar",
    ".gz",
}

# Confidence gate thresholds. Simplified from ARCHITECTURE.md's Stage-1/Stage-2
# design: there's no Stage-2 LLM discriminator in Phase 0 (MockBrain can't
# discriminate anything), so this is a direct two-threshold gate on the
# blended retrieval score.
CONFIDENT_THRESHOLD = 0.80
TENTATIVE_THRESHOLD = 0.65

# Hybrid retrieval blend: RRF Config
RRF_DENSE_WEIGHT = 0.6
RRF_FTS5_WEIGHT = 0.4

EPISODIC_RETENTION_DAYS = 90

# Phase 6: Evolutionary Loop & DPO Pipeline (Mitigations #68, #69, #70)
MOA_ROUTING_THRESHOLD = 0.5
DPO_MIN_CORPUS_SIZE = 50
DPO_NOVELTY_THRESHOLD = 0.7
DPO_DATASET_DIR = DATA_DIR / "datasets"


# Phase 5: Domain Validation & Engine Integration
UNITY_EXE_PATH_ENV = "UNITY_EXE"
BLENDER_EXE_PATH_ENV = "BLENDER_EXE"
UNITY_PROJECT_PATH_ENV = "UNITY_PROJECT_PATH"
UNITY_DAEMON_PORT = 8080

def get_unity_exe() -> Path | None:
    import os
    import shutil
    # 1. Environment variable
    if UNITY_EXE_PATH_ENV in os.environ:
        return Path(os.environ[UNITY_EXE_PATH_ENV])
    # 2. PATH
    which_unity = shutil.which("Unity")
    if which_unity:
        return Path(which_unity)
    # 3. Common fallback (Windows)
    fallback = Path(r"C:\Program Files\Unity\Hub\Editor")
    if fallback.exists():
        # Just grab the first version found, not ideal but better than nothing
        for exe in fallback.rglob("Unity.exe"):
            return exe
    return None

def get_blender_exe() -> Path | None:
    import os
    import shutil
    # 1. Environment variable
    if BLENDER_EXE_PATH_ENV in os.environ:
        return Path(os.environ[BLENDER_EXE_PATH_ENV])
    # 2. PATH
    which_blender = shutil.which("blender")
    if which_blender:
        return Path(which_blender)
    # 3. Common fallback (Windows)
    fallback = Path(r"C:\Program Files\Blender Foundation")
    if fallback.exists():
        for exe in fallback.rglob("blender.exe"):
            return exe
    return None

def get_unity_project_path() -> Path | None:
    import os
    if UNITY_PROJECT_PATH_ENV in os.environ:
        return Path(os.environ[UNITY_PROJECT_PATH_ENV])
    return None

# Phase 6: OS-Level Docker Sandbox (Mitigation #72)
DOCKER_SANDBOX_IMAGE: str = "python:3.11-slim"
DOCKER_SANDBOX_TIMEOUT: float = 5.0
DOCKER_SANDBOX_MEMORY: str = "256m"
DOCKER_SANDBOX_CPUS: str = "1.0"

# Phase 6: QLoRA Fine-Tuning Constants
TRAINING_BASE_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAINING_MAX_SEQ_LENGTH: int = 1024
TRAINING_MAX_PROMPT_LENGTH: int = 512


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
    DPO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

