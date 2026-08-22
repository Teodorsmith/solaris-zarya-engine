# Phase 4 Technical Readiness & Substrate Audit

> **Status**: Phase 0–3 Substrate Verified. Ready for Phase 4 (Autonomous Maintenance & Metacognitive Reasoning).

---

## 1. Substrate Verification (Phases 0–3 Completed)

Before introducing autonomous background loops or metacognitive self-models, all core deterministic substrates have been implemented, hardened, and verified:

| Component | Status | Substrate Verification |
| :--- | :--- | :--- |
| **Memory Hierarchy** | ✅ Verified | 4 SQLite WAL stores (`episodic.db`, `semantic.db`, `procedural.db`, `projects.db`, `goals.db`) with atomic writes, schema migrations, and deduplication gates. |
| **Embedding & Gating** | ✅ Verified | FastEmbed ONNX (`bge-small-en-v1.5`) with 0.65 tentative and 0.80 confident thresholds. Project file hits are calibrated so unrelated queries refuse cleanly. |
| **Skill Safety** | ✅ Verified | Host-side AST allowlist (`agent/engine/validator.py`) blocking unsafe primitives (`eval`, `exec`, `os.system`, `subprocess`, `socket`, `ctypes`) with subprocess test execution and 2-retry repair. |
| **Planning & FSM** | ✅ Verified | Goal DAG decomposition with prerequisite ordering, depth caps, deterministic file-write tier overrides, and crash-resilient `active_task.json` + `state_manifest.json`. |
| **Governor Centralization** | ✅ Verified | `PermissionGovernor` intercepts all file writes, skill synthesis, and multi-step DAG actions, logging structured audit events to `episodic.db`. |
| **Multi-Brain Switch** | ✅ Verified | `BrainManager` dynamically hot-swaps between Gemini, Groq, OpenAI, Local Ollama/vLLM, and MockBrain without process restart. |

---

## 2. Phase 4 Target Scope & Specifications

Phase 4 introduces proactive self-maintenance, reasoning traces, and metacognitive calibration:

### 2.1 Autonomous Heartbeat Daemon (`agent/engine/heartbeat.py`)
- Background daemon running an idle **Perceive $\to$ Evaluate $\to$ Plan $\to$ Act** cycle.
- **Cadence & Rate Limits**: Maximum 3 autonomous actions per hour.
- **Safety**: Automatically pauses when interactive user input is detected in the REPL.
- **Tasks**: Stale fact refresh (facts > 180 days old), dead-link verification, and background project re-indexing.

### 2.2 Persistent Self-Model (`data/self_model.json` & `agent/memory/self_model.py`)
- Maintains cross-cutting profile: `global_scores`, `domain_deltas`, `strategy_index`, and `zpd_ceilings`.
- Startup tamper detection verified against `data/state_manifest.json` with automatic rollback on checksum mismatch.

### 2.3 Tier 2.5 Reasoning Memory (`data/reasoning.db` & `agent/memory/reasoning.py`)
- Permanent retention of structured reasoning episodes (SHyAOEDRGL tuples):
  - **S**ituation, **Hy**pothesis, **A**ction, **O**utcome, **E**valuation, **D**ivergence, **R**eflection, **G**oal, **L**esson.

### 2.4 Lateral Critic & ZPD Difficulty Calibration (`agent/engine/critic.py`)
- Parallel two-solver dispatch on high-complexity reasoning tasks.
- Skips critic on consensus; triggers divergence arbitration on disagreement.
- 5-round Zone of Proximal Development (ZPD) binary search to benchmark and discover difficulty ceilings per category.

---

## 3. Exit Gates for Phase 4

1. **ZPD Benchmark**: `agent calibrate-reasoning` discovers difficulty ceilings across 6 reasoning categories and records them to `self_model.json`.
2. **Heartbeat Idle Test**: Detects stale facts while idle and runs refresh within the 3 actions/hour ceiling.
3. **Tamper Test**: Manual corruption of `self_model.json` is caught on boot and safely rolled back.
4. **Critic Consensus Test**: Confirms zero latency overhead on unanimous solver solutions and full arbitration on divergent solver traces.
5. **Pytest Coverage**: All new tests in `tests/test_heartbeat.py`, `tests/test_reasoning.py`, and `tests/test_reflection.py` pass.