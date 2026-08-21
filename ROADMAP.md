# Engineering Implementation Roadmap
## Autonomous Research & Skill Synthesis Engine with 4-Tier Memory & Metacognitive Reasoning

> **Document Role**: Tactical build specification and phased execution guide.
> **North Star Specification**: For comprehensive architectural diagrams, formal mathematical invariants, and complete details of all 70 mitigations, see [ARCHITECTURE.md](file:///e:/AI%20double/ARCHITECTURE.md).

> [!IMPORTANT]
> ### ROADMAP SCOPE & CORE PURPOSE
> **The product is the autonomous self-learning agent.**
> 
> The phase order prioritizes the agent's core cognitive substrate first:
> $$\text{Memory Core} \longrightarrow \text{Project Grounding} \longrightarrow \text{Skill Execution} \longrightarrow \text{Planning \& FSM} \longrightarrow \text{Metacognitive Autonomy}$$
> 
> **Unity / Blender / Game Development is a secondary validation milestone (Phase 5)** — a rigorous stress-test to prove the agent can learn, plan, execute, synthesize tools, and self-correct in complex real-world external environments. The agent is a general-purpose intelligent double, capable of programming, research, game design, systems automation, and complex problem-solving.

---

## Roadmap Overview & Phase Ladder

Development proceeds along an incremental, verifiable ladder where each phase solves one concrete capability barrier before adding complexity:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 0: Memory Core & Offline Loop (Zero Setup, MockBrain)            │
│ (3 SQLite WAL Stores, FastEmbed ONNX, Closed-World Grounding)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 1: Real Brain & Project Codebase Memory (Grounded Retrieval)     │
│ (Live Brains, Tier 4 Project Indexer, Hash Diffs, Symbol Grounding)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 2: Real Skill Execution & Safety (Subprocess & Revision Loop)    │
│ (AST Security Validator, Dry-Run Execution, 2-Retry Error Repair)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 3: Supervised Planning, Task FSM & Chat/Stream Interaction       │
│ (Hierarchical Goal DAG, On-Disk Task FSM, HITL Tiers, Crash Resume)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 4: Autonomous Maintenance & Metacognitive Reasoning (V2)        │
│ (Heartbeat Daemon, Self-Model, Reasoning Memory, Lateral Critic)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 5: Domain Validation & Engine Integration (Unity / Blender MCP)  │
│ (Unity MCP C#/Scenes, Headless UTF Test Runner, Blender 3D Export)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 6: Evolutionary Loop & Model Fine-Tuning (MoA / DPO Pipeline)    │
│ (Mixture-of-Agents Router, Novelty/Entropy Filter, DPO Dataset Builder)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Memory Core & Offline Loop

### Objective
Build the foundational local memory loop with **zero external API dependencies** and zero configuration hurdles. Establish the 3 primary memory stores, filetree skill persistence, and deterministic offline testing.

### Scope & Files to Implement
- `agent/config.py`: Directory auto-creation (`data/`, `skills/`, `seed_data/`), base paths, `SEED_VERSION`, default similarity thresholds (0.65 / 0.80).
- `agent/models.py`: Core Pydantic schemas (`Fact`, `Passage`, `Skill`, `EpisodicLog` with telemetry columns: `prompt_hash`, `strategy_label`, `novelty_score`, `reasoning_domain`, `outcome_class`, `hypothesis_count`).
- `agent/brains/`:
  - `base.py`: Abstract `BaseBrain` interface (`generate()`, `embed()`, `extract_json()`).
  - `mock_brain.py`: Deterministic offline brain with known-topics declaration and predictable outputs (Mitigations #11, #28).
  - `factory.py`: Brain loader with clean fallback to `MockBrain` (Mitigation #56).
- `agent/memory/`:
  - `embeddings.py`: FastEmbed ONNX embedding generator (L2-normalized vectors) + basic cosine similarity and keyword matching (Mitigations #2, #30).
  - `episodic.py`: Tier 1 SQLite WAL store (`episodic.db`) with 90-day TTL and explicit strategy telemetry columns (Mitigations #18, #20, #67).
  - `semantic.py`: Tier 2 SQLite WAL store (`semantic.db`) managing `semantic_facts` and `context_passages` with deduplication (>0.95 similarity check) and contradiction gating (Mitigations #6, #7, #31, #32, #33).
  - `procedural.py`: Tier 3 SQLite WAL store (`skills.db`) storing tool metadata and syncing with real, human-readable `.py` files in `skills/` (Mitigations #15, #35).
  - `seeder.py`: Foundational bootloader populating initial facts from `seed_data/facts.json` (Mitigation #17).
- `agent/engine/retriever.py`: Cosine similarity + keyword retriever with confidence gating (0.65 / 0.80) and closed-world grounding prompt assembly (Mitigations #29, #37).
- `agent/cli.py` & `agent/main.py`: Interactive CLI shell supporting `ask`, `correct`, `forget`, and `--demo`.

### Key Safeguards Enforced
- **Mitigation #1**: One-directional import graph (no circular imports).
- **Mitigation #20**: SQLite WAL mode on all connections.
- **Mitigation #29**: Closed-world prompt grounding (honest refusal on missing knowledge).
- **Mitigation #67**: Episodic telemetry schema locked before first `INSERT`.

### Concrete Proof Milestone (Exit Gate)
1. Run `python -m agent.main --demo --reseed`.
2. Seed data loads without errors into `semantic.db` and `episodic.db`.
3. Query: `"How do I check my git status?"` $\to$ Returns grounded answer from seed facts (retrieval score $\ge 0.80$).
4. Query: `"What is the maximum upload size for GitHub releases?"` $\to$ Returns honest refusal (`"I found related information about GitHub releases, but my stored memory does not contain the specific upload size limit."`).
5. **Disk Reuse Test**: Run the demo twice. A skill learned in run 1 loads from `skills/` on disk and is reused in run 2 instead of re-synthesizing.
6. Run unit tests: `pytest tests/test_memory.py tests/test_retriever.py` $\to$ All pass with 100% green.

---

## Phase 1: Real Brain + Project Codebase Memory

### Objective
Connect a real LLM brain and give the agent grounded awareness of the user's actual codebase, enabling file-aware code collaboration with strict anti-hallucination refusal paths.

### Scope & Files to Implement
- `agent/brains/`:
  - `gemini_brain.py` / `claude_brain.py` / `openai_brain.py`: Direct API brain integrations using environment variables (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or local Ollama) with 3-retry exponential backoff and robust JSON extraction (Mitigations #8, #9).
- `agent/memory/project.py`: Tier 4 Project Memory manager (`projects.db`):
  - Full-tree scan with `.gitignore`-style exclusions.
  - `project_files` table: relative paths, SHA-256 content hashes, and concise role summaries.
  - `project_decisions` table: records architecture decisions with `related_files_json`.
  - Incremental indexing via hash diffs (re-embed and re-summarize only changed files; Mitigations #50, #52).
- `agent/engine/retriever.py` (Extension): Cross-tier retrieval searching `project_files` alongside semantic facts and skills for code-aware queries.
- `agent/cli.py` (Extension): Add `project index <path>` and `project decision <title> <content>` commands.

### Key Safeguards Enforced
- **Mitigation #8 & #9**: 3-stage JSON extraction with exponential backoff on live API calls.
- **Mitigation #50**: Project Memory write-protection (agent cannot directly write to `projects.db`).
- **Mitigation #29**: Grounded answers citing specific indexed files; refusal on non-existent symbols.

### Concrete Proof Milestone (Exit Gate)
1. Set API key in `.env` and run `python -m agent.main --brain gemini`.
2. Index codebase: `project index .` $\to$ Files are hashed and indexed into `projects.db`.
3. Ask a real question about your actual codebase: `"Where is configuration handled?"` $\to$ Returns grounded answer citing `agent/config.py` with its exact role summary.
4. Ask about a non-existent module: `"Where is the payment gateway handled?"` $\to$ Honestly refuses instead of hallucinating a file.
5. Run unit tests: `pytest tests/test_brains.py tests/test_project.py` $\to$ All pass.

---

## Phase 2: Real Skill Execution & Safety

### Objective
Enable the agent to synthesize, test, and execute real Python tools on the machine safely within an isolated subprocess, handling errors honestly through an iterative revision loop.

### Scope & Files to Implement
- `agent/models.py`: Add `Skill` and immutable `SkillResultSchema` (Mitigation #49).
- `agent/engine/validator.py`:
  - Subprocess runner with strict execution timeout (5s) and captured `stdout`/`stderr`.
  - Fast AST validation blocking dangerous primitives (`os.system`, `eval`, raw `exec` of untrusted strings, `__builtins__` reflection, `socket`, `ctypes`).
  - Import resolution checking against allowed stdlib modules (Tier 1 & Tier 2 allowlist).
  - Host-side typed result validation against `SkillResultSchema` (Mitigation #49).
- `agent/engine/synthesizer.py`: Fact-anchored tool generator (Mitigation #12), test generator using `unittest.mock`, and 2-retry revision loop that parses real stderr feedback to repair code (Mitigation #35).
- `skills/`: Real Python tool files saved to disk (`skills/<topic>_<name>.py`).

### Key Safeguards Enforced
- **Mitigation #12**: Fact-anchored synthesis prompts with syntax assertions.
- **Mitigation #25 (Core)**: AST allowlist blocking execution of unsafe dynamic patterns.
- **Mitigation #35**: Real dry-run execution loop + 2-retry revision on compiler/runtime errors.
- **Mitigation #49**: Host-side typed output parsing (self-reported exit codes not trusted).

### Concrete Proof Milestone (Exit Gate)
1. Command: `learn "JSON parsing and formatting utility"`.
2. Agent writes `skills/stdlib_json_formatter.py`.
3. Validator runs tool in subprocess; passes tests $\to$ Registered in `skills.db` as `real_local`.
4. Inject a synthetic syntax error into a generated tool $\to$ Validator catches stderr $\to$ Revision loop automatically repairs syntax within 2 retries.
5. Inject an unsafe command (`os.system('rm ...')`) $\to$ Validator rejects before execution.
6. Run unit tests: `pytest tests/test_validator.py` $\to$ All pass.

---

## Phase 3: Supervised Planning, Task FSM & Chat/Stream Interaction

### Objective
Implement multi-step goal decomposition, on-disk crash-resilient Task FSM, Human-in-the-Loop (HITL) safety governors, and a live chat adapter (Twitch / Discord) with persistent viewer memory.

### Scope & Files to Implement
- `agent/memory/goals.py`: SQLite store (`goals.db`) for hierarchical DAG prerequisite trees with completion criteria (Mitigations #42, #47, #54).
- `agent/engine/planner.py`: Curriculum Planner decomposing complex goals into conceptual, practical, and skill targets, computing `novelty_score` and `complexity_score` (Mitigations #26, #62, #67, #70).
- `agent/engine/state_machine.py`: Deterministic Task FSM managing `data/active_task.json` through atomic transitions (`PENDING` $\to$ `RUNNING` $\to$ `VERIFYING` $\to$ `COMMITTED` $\to$ `FAILED`), enforcing a 5-step cap and 2-failure circuit breaker (Mitigation #49).
- `agent/engine/governor.py`: Permission Governor enforcing Tier 0 (read/search), Tier 1 (sandboxed test), and Tier 2 (terminal `[Y/n]` confirmation for file writes and system actions; Mitigations #46, #54).
- `agent/chat/`:
  - `twitch_adapter.py` / `discord_adapter.py`: Chat connection adapter reading messages and streaming responses.
  - Per-viewer interaction tracking in `episodic.db` (username, interaction history, preferences).

### Key Safeguards Enforced
- **Mitigation #42**: Directed Acyclic Goal Graph with prerequisite validation.
- **Mitigation #46**: HITL Permission Tiers and step/token circuit breakers.
- **Mitigation #47 & #54**: Goal tree depth caps (depth 2 autonomous; depth 3–4 requiring per-expansion `[Y/n]` approval).
- **Mitigation #49**: Deterministic on-disk task state machine with single-atomic-action turns.

### Concrete Proof Milestone (Exit Gate)
1. Multi-Step Task: `task "Research Docker CLI, extract facts, and synthesize a container inspect tool"`.
2. Planner creates goal tree and starts `active_task.json`.
3. Crash Simulation: Kill process at step 3 $\to$ Restart $\to$ Agent reads disk state, resumes at step 3 without repeating prior steps, prompts `[Y/n]` before final write, and commits.
4. Chat Persistence Test: Viewer sends chat message $\to$ Agent answers $\to$ Viewer returns later $\to$ Agent remembers viewer context from `episodic.db`.
5. Run unit tests: `pytest tests/test_goals.py tests/test_state_machine.py tests/test_governor.py` $\to$ All pass.

---

## Phase 4: Autonomous Maintenance & Metacognitive Reasoning (V2)

### Objective
Expand the system with proactive self-maintenance and true reasoning improvement: autonomous Heartbeat Daemon, write-protected Self-Model, Tier 2.5 Reasoning Memory (`reasoning.db`), Lateral Critic, and ZPD Binary Search difficulty calibration.

### Scope & Files to Implement
- `agent/memory/self_model.py`: Persistent `data/self_model.json` manager with `reasoning_profile` (`global_scores`, `domain_deltas`, `strategy_index`, `zpd_ceilings`) and startup tamper detection against `state_manifest.json` (Mitigations #40, #52, #62).
- `agent/engine/heartbeat.py`: Autonomous background daemon executing Perceive $\to$ Evaluate $\to$ Plan $\to$ Act idle loop with 3 actions/hour rate limit, pausing on active user input (Mitigations #41, #46).
- `agent/engine/reflection.py`: Metacognitive reflection engine consolidating episodic logs and running weekly SQL aggregation jobs to compute `domain_deltas` (Mitigations #43, #55, #62, #67).
- `agent/memory/reasoning.py`: Tier 2.5 SQLite store (`reasoning.db`) storing `reasoning_episodes` with full SHyAOEDRGL tuples and permanent retention (Mitigation #61).
- `agent/engine/strategy_injector.py`: Injects reasoning prompt templates from `strategy_index` and stamps `strategy_label` + `prompt_hash` into telemetry (Mitigations #62, #67).
- `agent/engine/critic.py`: Lateral Critic executing parallel two-solver dispatch, skipping critic on consensus and arbitrating divergences (Mitigation #63).
- `agent/engine/verifier.py`: Symbolic Verifier for Structured Reasoning Traces (SRTs; Mitigation #65).
- `agent/engine/benchmark.py` & `tests/reasoning_suite/`: 6-category reasoning benchmark suite with 5-round ZPD binary search difficulty calibration (Mitigation #66).

### Key Safeguards Enforced
- **Mitigation #40 & #52**: Self-Model persistence with tamper detection and rollback.
- **Mitigation #41**: Autonomous Heartbeat Daemon with silent no-op cadence.
- **Mitigation #61 & #62**: Tier 2.5 Reasoning Memory + Cross-cutting Bayesian reasoning profile.
- **Mitigation #63**: Lateral Critic (two-solver consensus + divergence arbiter).
- **Mitigation #65 & #66**: SRT Symbolic Verifier + ZPD Binary Search difficulty calibration.

### Concrete Proof Milestone (Exit Gate)
1. Run `agent calibrate-reasoning` $\to$ 5-round ZPD binary search discovers difficulty ceilings per category $\to$ Updates `zpd_ceilings` in `self_model.json`.
2. Heartbeat idle test: Detects 180-day stale facts on idle $\to$ Triggers background refresh within 3 actions/hour ceiling.
3. Tamper test: Modify `self_model.json` manually $\to$ Restart agent $\to$ Tamper watchdog detects mismatch and rolls back.
4. Divergent problem: Lateral critic dispatches two solvers $\to$ Detects disagreement $\to$ Arbitrates and logs full SHyAOEDRGL episode to `reasoning.db`.
5. Run unit tests: `pytest tests/test_heartbeat.py tests/test_reflection.py tests/test_reasoning.py` $\to$ All pass.

---

## Phase 5: Domain Validation & Engine Integration (Unity / Blender MCP)

### Objective
Stress-test the agent in a complex external domain by connecting it to the game development workflow via Unity and Blender MCP servers. Validate that the agent's core memory, planning, and revision loops enable it to autonomously author C# scripts, inspect scene hierarchies, create/export 3D assets, and pass headless test suites.

```
                  ┌───────────────────────────────────────────────┐
                  │ Phase 5: Domain Stress-Test & Environment     │
                  └───────────────────────┬───────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌─────────────────────────┐                     ┌─────────────────────────┐
     │ Unity MCP Server        │                     │ Blender MCP Server      │
     │ - Read/write C# scripts │                     │ - Create/import assets  │
     │ - Inspect scenes/prefabs│                     │ - Headless scripts (.py)│
     │ - Run Unity Test Runner │                     │ - Export FBX/glTF       │
     └────────────┬────────────┘                     └────────────┬────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ Headless Unity Compile & Play Smoke Tests     │
                  │ (Unity.exe -runTests -testPlatform EditMode)  │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ Git Branch Isolation (ai-feat/*) & Regression │
                  └───────────────────────────────────────────────┘
```

### Scope & Files to Implement
- `agent/integrations/unity_mcp.py`:
  - Unity MCP client connecting to Unity Editor bridge daemon.
  - Scene and prefab introspection (GameObjects, components, serialized properties, layer masks).
  - C# script authoring, patching, and Unity Roslyn compiler error feedback (`CS0246`, `CS1061`; Mitigation #36).
  - Headless test execution runner (`Unity.exe -runTests -testPlatform EditMode/PlayMode -testResults results.xml`).
- `agent/integrations/blender_mcp.py`:
  - Headless Blender execution bridge (`blender --background --python <script>`).
  - Procedural 3D mesh generator, material assignment, UV unwrapping, and automated export (FBX / glTF) into Unity `Assets/` folders.
  - Asset sanity checks (vertex count, scale normalization, `.meta` file integrity).
- `agent/engine/synthesizer.py` (C# Multi-Runtime Extension):
  - Specialized templates for Unity `MonoBehaviour`, `ScriptableObject`, State Machines, and Editor Scripts.
  - 2-retry revision loop against real Unity Roslyn compiler errors and Test Runner output.
- `agent/engine/vcs_manager.py`:
  - Automated task branching (`git checkout -b ai-feat/<task-name>`).
  - Runs regression smoke tests in headless Unity before staging commits.

### Key Safeguards Enforced
- **Mitigation #36**: Multi-runtime synthesis & error feedback loop (`dotnet` / Unity C# compiler feedback).
- **Mitigation #46 & #50**: Project Memory write-protection and Tier-2 HITL approval before writing to host Unity assets or committing changes.
- **Mitigation #49**: Deterministic task execution with atomic rollback if Unity compilation fails or UTF test suite breaks.

### Concrete Proof Milestone (Exit Gate)
1. Connect agent to local Unity project and Blender.
2. Dispatch domain task: `"Add a HealthComponent with unit tests and a simple low-poly potion bottle model"`.
3. Agent:
   - Synthesizes `HealthComponent.cs` and `HealthComponentTests.cs`.
   - Runs headless Blender script to generate and export `potion_bottle.fbx` into `Assets/Models/`.
   - Triggers Unity Test Runner in headless EditMode; captures test results XML $\to$ all pass.
   - Summarizes changes and stages git commit on feature branch `ai-feat/health-component`.
4. Run integration tests: `pytest tests/test_game_engine_integration.py` $\to$ All pass.

---

## Phase 6: Evolutionary Loop & Model Fine-Tuning (V3/V4)

### Objective
Close the evolutionary loop: deploy the Mixture-of-Agents (MoA) router for LoRA co-processing in V3, and build the automated Novelty/Entropy filtered DPO dataset pipeline for true Level 3 model weight fine-tuning in V4.

### Scope & Files to Implement
- `agent/brains/moa_router.py`: Mixture-of-Agents Router routing on `complexity_score` between cloud API (routine) and fine-tuned LoRA reasoning adapter (Mitigation #70).
- `agent/engine/dataset_builder.py`: Dataset Builder enforcing Novelty gate (`novelty_score > 0.7`), Solution-Path Entropy gate ($>2\sigma$ from centroid), and Symbolic Verification gate (`verified = true`), formatting accepted episodes into DPO preference pairs (Mitigations #68, #69).
- `agent/engine/trainer.py` (V4): Automated fine-tuning coordinator with benchmark promotion gate (must improve $\ge 3/6$ ZPD ceilings before deploying new weights; Mitigation #69).

### Key Safeguards Enforced
- **Mitigation #68**: Novelty & Entropy Filter preventing training set distribution collapse.
- **Mitigation #69**: Verified Experience $\to$ Training Data DPO pipeline with rollback benchmark gate.
- **Mitigation #70**: MoA Router separating task-intrinsic `complexity_score` from task-relative `novelty_score`.

### Concrete Proof Milestone (Exit Gate)
1. Filter test: Ingest 1,000 reasoning episodes $\to$ `dataset_builder.py` discards low-novelty and unverified episodes, retaining only $\ge 500$ high-novelty, verified DPO pairs.
2. MoA test: Task with `complexity_score = 0.8` routes to LoRA adapter; task with `complexity_score = 0.2` routes to base cloud API.
3. Promotion gate test: Simulated fine-tuned model failing benchmark suite is rejected and rolled back to previous checkpoint.
4. Run unit tests: `pytest tests/test_dataset_builder.py tests/test_moa_router.py` $\to$ All pass.

---

## Backlog

Everything below is fully documented in [ARCHITECTURE.md](file:///e:/AI%20double/ARCHITECTURE.md) and represents long-term engineering depth — pull an item into an active phase only when real build constraints force the issue:

- **OS-Level Containment**: Full gVisor sealed sandbox with read-only squashfs rootfs and 4 prebuilt Docker images (`sandbox-stdlib`, `sandbox-web`, `sandbox-scientific`, `sandbox-full`; Mitigations #38, #57–#60).
- **Advanced Retrieval**: Hybrid Reciprocal Rank Fusion (dense ONNX + BM25 FTS5), calibrated confidence thresholds via labeled evaluation set (`calibration/queries.json`; Mitigations #30, #37, #53).
- **Web Ingestion Resiliency**: Multi-provider fetch chain (Jina Reader $\to$ Trafilatura $\to$ BeautifulSoup $\to$ Playwright) with domain authority scoring and abort guard (Mitigations #4, #5, #27, #39).
- **Arbitrary Provider Registry**: Dynamic user-editable `brains.json` supporting custom local/cloud OpenAI-compatible endpoints (Mitigation #56).

---

## Build Execution Matrix

| Phase | Milestone Name | Primary Test Target | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Memory Core & Offline Loop** | `tests/test_memory.py`, `tests/test_retriever.py` | ✅ **Completed (11/11 tests pass)** |
| **Phase 1** | **Real Brain + Project Memory** | `tests/test_brains.py`, `tests/test_project.py` | 🟡 **Active Build Target** |
| **Phase 2** | **Real Skill Execution & Safety** | `tests/test_validator.py` | ⚪ Queued |
| **Phase 3** | **Supervised Planning & Chat** | `tests/test_goals.py`, `tests/test_state_machine.py` | ⚪ Queued |
| **Phase 4** | **Autonomous Maintenance & Reasoning (V2)** | `tests/test_reasoning.py`, `tests/test_heartbeat.py` | ⚪ Queued |
| **Phase 5** | **Domain Validation & Engine Integration** | `tests/test_game_engine_integration.py` | ⚪ Queued |
| **Phase 6** | **Evolutionary Loop & Model Fine-Tuning** | `tests/test_dataset_builder.py` | ⚪ Queued |
