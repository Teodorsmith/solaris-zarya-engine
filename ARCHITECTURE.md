# Autonomous Research & Skill Synthesis Engine with 4-Tier Memory & Metacognitive Reasoning
## Master System Architecture & End-State Specification (North Star Vision)

> **Document Role**: Comprehensive North Star system architecture, formal mathematical invariants, full 70-mitigation catalog, and complete end-state technical reference.
> **Tactical Build Plan**: For day-to-day phased implementation steps, specific files to code per milestone, and concrete verification gates, follow [ROADMAP.md](file:///e:/AI%20double/ROADMAP.md).

> [!IMPORTANT]
> ### PRIMARY PROJECT SCOPE
> **This document defines the agent itself.**
> 
> Unity / Blender / game development is an **EXAMPLE DOMAIN** used to stress-test whether the agent can autonomously learn, plan, execute, and self-correct against real environment feedback.
> 
> The agent is not merely a game-development harness. It is a **general self-learning autonomous agent with persistent multi-tier memory** that can assist with game development, general software engineering, deep research, system automation, and arbitrary complex workflows.

> [!IMPORTANT]
> ### IMPLEMENTATION STATUS: SUBSTRATE VS. EXTENDED HORIZONS
> - **Active Implemented Core (Phases 0–4 & Sandbox & QLoRA)**:
>   - **4-Tier SQLite WAL Memory**: `episodic.db`, `semantic.db`, `procedural.db`, `projects.db`, `goals.db`.
>   - **Hybrid Retrieval & Gate**: FastEmbed ONNX (`bge-small-en-v1.5`) dense cosine + SQLite FTS5 BM25 weighted blend with direct 0.65 / 0.80 two-threshold confidence gate and closed-world refusal.
>   - **Skill Execution Safety**: Host-side Python AST allowlist validator (`agent/engine/validator.py`) and strict Docker OS-Level Sandbox containment.
>   - **Task DAG & FSM**: Goal DAG decomposition, deterministic file-write tier overrides, and crash-resilient `data/active_task.json` + `data/state_manifest.json` state machine.
>   - **Permission Governor**: Centralized `PermissionGovernor` enforcing Tier 0/1/2 HITL approval, depth caps, and episodic audit logging.
>   - **Brain Hot-Swapping**: `BrainManager` hot-swapping Gemini, Groq, OpenAI, Local Ollama/vLLM, and MockBrain.
>   - **Conversational REPL**: `ChatEngine` fallback maintaining 10-turn context, persona injection, and semantic project grounding.
>   - **Metacognitive Maintenance (Phase 4)**: Asynchronous Heartbeat daemon, `data/self_model.json`, Tier 2.5 `reasoning.db`, and academic paper ingestion.
>   - **QLoRA & MoA (Phase 6)**: Mixture-of-Agents routing separating complexity and automated `trl.DPOTrainer` fine-tuning pipeline for LoRA adapters.
> - **Extended Target Specifications (Phases 5 & 6 Remainder)**:
>   - Unity and Blender MCP bridges and headless UTF test runner (Phase 5).
>   - Automated Dataset Builder enforcing Novelty/Entropy filters for DPO pair generation (Phase 6).

---

## 1. Overview & Core Philosophy

The **Autonomous Research & Skill Synthesis Engine** provides an AI Agent with the ability to:
1. **Learn any topic on demand** via autonomous curriculum decomposition and multi-source ingestion.
2. **Distill atomic, verifiable knowledge** into a persistent semantic memory store with non-volatile fact versioning.
3. **Crystallize actionable, reusable skills** as standalone tools with sandboxed validation and a layered verification tier (`mock` → `real_local` → `real_external`). V1 targets Python; additional runtimes (C#, GDScript, C++, shaders) are an explicit extension (Mitigation #36).
4. **Achieve high-fidelity, hallucination-resistant answers** through a strict **Two-Stage Confidence-Gated Retrieval** engine with closed-world prompting that honestly admits ignorance when knowledge is missing or unverified.
5. **Compound capabilities over time** through **Hierarchical Skill Composition (Voyager Loop)**, allowing new tools to import and chain existing verified skills.
6. **Exercise proactive agency and metacognition** via an asynchronous **Heartbeat Daemon** (autonomous background loop), a persistent **Self-Model** (`self_model.json`), a **Directed Acyclic Goal Graph** (`goals.db`), and periodic **Metacognitive Reflection** consolidation cycles.
7. **Operate under strict mechanical governors & HITL permission tiers** (Mitigations #45–#49), anchoring competence strictly in immutable benchmark test suites rather than LLM self-deception, enforcing circuit breakers against runaway loops, and running deterministic state machines over context windows.
8. **Maintain a persistent Project Memory (Tier 4, `projects.db`)** indexing the user's actual codebase — files, hashes, roles, summaries, and project decisions — so it can collaborate on real software (e.g., a Unity project) instead of only answering general questions. Project Memory participates in retrieval alongside Semantic and Procedural Memory (Mitigation #50).

> **One-Sentence Summary**:
> *"A self-teaching AI agent that autonomously researches topics, writes and tests reusable tools, maintains a persistent self-model and goal graph, proactively self-improves during idle cycles, and strictly refuses to extrapolate when knowledge is missing."*

> [!IMPORTANT]
> **"Zero hallucinations" is not an achievable goal.** (Note: Ensure "hallucination-resistant grounded retrieval" is the terminology used). The closed-world prompt reduces hallucination dramatically, but an LLM can still ignore instructions, paraphrase retrieved facts incorrectly, infer a plausible missing detail, or retrieve a wrong fact that scores above threshold. The stated goal of this system is **hallucination-resistant** answers achieved through: (1) closed-world grounding (Mitigation #29), (2) confidence gating (Mitigation #37), (3) honest refusal paths, and (4) user-corrected top-authority facts (Mitigation #31). Hallucination is treated as a residual, measured risk — not a solved problem. Any benchmark or release note claiming otherwise is incorrect.

> [!NOTE]
> **Broad general knowledge comes from the Brain, not the Memory.** This system does **not** replace the underlying LLM's pretrained knowledge. With Gemini/Claude/OpenAI as the brain, the agent already has broad general knowledge; the memory system *augments* it with persistent, private, sourced, and user-corrected facts, verified skills, and project state. With MockBrain or a small local model, the agent only knows what it has explicitly ingested — there is no latent broad knowledge. Feature comparisons ("knows about history like ChatGPT") depend entirely on which brain is configured, not on the memory tier.
>
> **Which brain is used is entirely user-configurable** via the `brains.json` provider registry (Mitigation #56). Any OpenAI-compatible, Google GenAI, Anthropic, or local (Ollama/vLLM) endpoint can be registered by hand — e.g., OpenCodeZen DeepSeek v4, Google Gemini, or Codex — and selected with `--brain <provider>`. This is **not** limited to the built-in `mock`/`gemini`/`claude` backends.

---

## 2. High-Level Architecture Diagram

```
                     ┌──────────────────────────────────────────────────┐
                     │          DUAL-MODE COGNITIVE ARCHITECTURE        │
                     └────────────────────────┬─────────────────────────┘
                                              │
          ┌───────────────────────────────────┴───────────────────────────────────┐
          │                                                                       │
          ▼ FOREGROUND (Reactive Mode)                                            ▼ BACKGROUND (Proactive Mode)
┌───────────────────────────────────────────┐                 ┌───────────────────────────────────────────┐
│ User Directive / Query                    │                 │ Heartbeat Daemon (Autonomous Idle Loop)   │
│ "Learn GitHub" OR "How to create Release?"│                 │ Perceive ──► Evaluate ──► Plan ──► Act    │
└─────────────────────┬─────────────────────┘                 └─────────────────────┬─────────────────────┘
                      │                                                             │
                      ▼                                                             ▼
┌───────────────────────────────────────────┐                 ┌───────────────────────────────────────────┐
│ 1. RESEARCH & SYNTHESIS PIPELINE          │                 │ 2. METACOGNITIVE & GOAL SUBSTRATE         │
│ - Curriculum Planner (Decomposition)      │                 │ - Goal Graph (data/goals.db)              │
│ - 4-Stage Ingestion (Abort if <100 chars) │                 │ - Self-Model (data/self_model.json)       │
│ - 3 Core Fact Types (Concept/Syntax/Fix)  │                 │ - Active Task FSM (data/active_task.json) │
│ - RAG-Selected Skill Context (Top 3)      │                 │ - Periodic Reflection Engine (Sleep cycle)│
│ - AST Reflection Ban + Subprocess Sandbox │                 │ - Autonomous Fact Refresh & Self-Testing  │
│ - HITL Permission Tiers (0, 1, 2 [Y/n])   │                 │ - Objective Benchmark Suite (Immutable)   │
└─────────────────────┬─────────────────────┘                 │ - Tier 4: Project Memory (projects.db)    │
                      │                                       └─────────────────────┬─────────────────────┘
                      │                                                             │
                      ▼                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. TWO-STAGE CONFIDENCE GATE & GROUNDED RETRIEVAL                                                       │
│ - Candidate Ranking: Reciprocal Rank Fusion (RRF) on Dense Cosine (BGE) + SQLite FTS5 BM25              │
│ - Gating Metric: L2-normalized Dense Cosine Similarity of Top-1 Candidate against Dynamic Thresholds    │
│ - Stage 1A (<0.65): Honest refusal ──► "I haven't learned this yet. Study it?"                          │
│ - Stage 1B (>=0.80): Closed-world grounded answer (Mitigation #29)                                      │
│ - Stage 2 (0.65-0.80): Fast LLM Discriminator validation                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 4 Pillars of Controlled Autonomy

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. HUMAN-IN-THE-LOOP (HITL) PERMISSION TIERS                                                            │
│    - Tier 0 (Autonomous): Read docs, query vector memory, read-only AST linting                         │
│    - Tier 1 (Guarded Autonomous): Execute tests in Docker/Job-Object sandbox; no network by default,    │
│                                    allowlisted hosts only for real_external verification (Mitigation #51)│
│    - Tier 2 (Explicit Approval Required): Install packages, modify workspace code, write host files,    │
│                                           dispatch live external network requests ([Y/n] confirmation) │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DETERMINISTIC STATE MACHINES OVER CONTEXT-WINDOW MEMORY                                              │
│    - Active plans persisted to data/active_task.json: PENDING ──► RUNNING ──► VERIFYING ──► COMMITTED   │
│    - Each turn executes exactly 1 atomic action, updates state on disk, and exits prompt context        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. GROUND-TRUTH TELEMETRY & TRACE IDs                                                                   │
│    - Every cycle stamped with UUID trace_id logged to episodic.db:                                      │
│      Goal ──► Prompt ──► Tool Invocation ──► Sandbox Stderr ──► State Mutation                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. OBJECTIVE BENCHMARKING (ANTI-SELF-DECEPTION HARNESS)                                                 │
│    - Immutable test suite in tests/benchmark_suite/ (agent forbidden from modifying)                    │
│    - Competence Score = (Passed Benchmark Tests) / (Total Benchmark Tests)                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 Machine Learning Topology & Extension Hierarchy

The system is fundamentally an **ML-native cognitive architecture**. Machine Learning is not an afterthought or a single feature; it forms the core substrate at every level of perception, memory retrieval, reasoning, and tool execution.

### Where ML Already Exists (Foundational Substrate)

| Core Component | ML Technique / Model | Role in Engine |
| :--- | :--- | :--- |
| **Dense Embeddings** | FastEmbed ONNX (`bge-small-en-v1.5`, 384d) | Vector representation of facts, passages, project files, and skills |
| **Semantic Retrieval** | L2-normalized Cosine Similarity | Dense vector similarity scoring against query vectors |
| **Sparse Retrieval** | SQLite FTS5 BM25 Term Ranking | Lexical relevance and Inverse Document Frequency (IDF) scoring |
| **Cognitive Brain** | Frontier API / Local LLM (Gemini, Claude, Ollama) | Reasoning, semantic distillation, planning, code synthesis, reflection |
| **Fact Distillation** | LLM Few-Shot JSON Extraction | Distills atomic, falsifiable facts from raw text inputs |
| **Skill Synthesis** | LLM Code Generation + AST Validation | Synthesizes executable tools with iterative compiler-error feedback |
| **Metacognitive Planning** | LLM Curriculum Decomposition & Reflection | Decomposes goals into DAGs and consolidates episodic logs |

---

### Value-Ranked ML Extension Hierarchy (Incremental Capabilities)

As the engine scales, additional specialized ML models can be layered into the architecture in rough order of operational leverage:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Level 1: Cross-Encoder Re-Ranking (bge-reranker-base ONNX)             │
│ (Retrieve Top 50 ──► Cross-Encoder Rerank ──► Feed Top 3-5 to LLM)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 2: Query-Type & Intent Classification (FastText / Logistic Reg)   │
│ (Routes: codebase_lookup | factual_qa | tool_exec | goal_planning)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 3: Bayesian Reinforcement Feedback for Skill Selection           │
│ (Beta-Bernoulli historical success weighting on tool retrieval)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 4: Domain-Adapted Embedding Fine-Tuning                          │
│ (Specialized vector fine-tuning for Unity/C# & domain vocabularies)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 5: Probabilistic Competence & Retrieval Sufficiency Prediction   │
│ (Model-free classifier predicting retrieval sufficiency before LLM)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 6: Local Brain Fine-Tuning (DPO / LoRA Pipeline — Phase 6)       │
│ (LoRA reasoning adapter fine-tuned on verified novelty-filtered traces)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Level 7: Vision ML for Visual Regression & Game Testing (Phase 5+)     │
│ (YOLO / CLIP / SAM for viewport inspection & UI regression tests)      │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Cross-Encoder Neural Re-Ranking (Level 1)**:
   - *Problem*: Dense cosine similarity + BM25 returns high recall, but top-3 ranking can still contain subtle misorderings.
   - *ML Solution*: Retrieve Top 50 candidates via Hybrid RRF, then re-rank with a local ONNX cross-encoder (e.g. `BAAI/bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) to extract the Top 3–5 highest-fidelity context items. Directly sharpens the Two-Stage Confidence Gate (Mitigation #37).
2. **Query-Type & Intent Classifier (Level 2)**:
   - *ML Solution*: A micro-classifier (logistic regression on embedding vectors or fastText) that categorizes queries before retrieval: `codebase_question`, `factual_question`, `tool_execution`, `multi_topic_comparison`, `goal_planning`. Allows routing straight to the optimal memory index without running all search paths.
3. **Bayesian Reinforcement for Skill Selection (Level 3)**:
   - *ML Solution*: Track verified execution outcomes in `skills.db`. Update a Beta-Bernoulli posterior on tool reliability ($(\alpha, \beta)$ parameters based on real-world execution success/failure) to dynamically boost proven skills during procedural retrieval.
4. **Domain-Adapted Embedding Fine-Tuning (Level 4)**:
   - *ML Solution*: Fine-tune sentence-transformer representations using contrastive loss on project-specific query/code pairs when standard embeddings struggle with domain-specific APIs (e.g., Unity `MonoBehaviour` lifecycle methods, custom shaders).
5. **Calibrated Competence & Sufficiency Prediction (Level 5)**:
   - *ML Solution*: Train a lightweight probabilistic model on retriever scores and lexical overlap to predict $P(\text{answerable} \mid \text{query}, \text{retrieved\_facts})$, making the honest refusal decision faster and more reliable before invoking the full LLM.
6. **Local Brain Fine-Tuning / LoRA (Level 6 — Phase 6)**:
   - *ML Solution*: Automated pipeline extracting verified, high-entropy, symbolically-validated episodes (Mitigation #68) and training LoRA/QLoRA adapters via Direct Preference Optimization (DPO; Mitigation #69).
7. **Vision ML for Visual Testing (Level 7 — Phase 5 Extension)**:
   - *ML Solution*: Use object detection and visual segmentation models (YOLO, CLIP, SAM) to inspect rendered game frames in Unity/Godot, detecting missing textures, broken UI bounding boxes, and visual regressions.

---

## 3. Persistent Memory Architecture & Meta-Cognitive Substrates

```
                                  ┌──────────────────────────────────┐
                                  │   PERSISTENT COGNITIVE STATE     │
                                  └─────────────────┬────────────────┘
                                                    │
        ┌──────────────┬─────────────┬──────────────┼──────────────┬────────────────────────────┐
        │              │             │              │              │                            │
        ▼ TIER 1       ▼ TIER 2      ▼ TIER 2.5     ▼ TIER 3       ▼ TIER 4: PROJECT
  EPISODIC           SEMANTIC     REASONING       PROCEDURAL
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
│episodic.db   │  │semantic.db   │  │reasoning.db  │  │skills.db     │  │projects.db           │
│- Interaction │  │- Distilled   │  │- Reasoning   │  │- /skills     │  │- project registry    │
│  logs        │  │  Facts       │  │  episodes    │  │  filetree    │  │- project_files index │
│- Audit trail │  │- Passages    │  │- Hypotheses  │  │- Tiered      │  │  (path, hash, role)  │
│- 90-day TTL  │  │- Vectors     │  │- Root causes │  │  Verification│  │- project_decisions   │
│- trace_id +  │  │- Dedup       │  │- Generalized │  │- Runtime     │  │- Write-protected     │
│  strategy    │  │  Gating      │  │  rules       │  │  params      │  │  (M50)               │
│  telemetry   │  │  Tracking    │  │- SRTs (M65)  │  │              │  │                      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘
       │                 │                 │                 │                      │
       └─────────────────┼─────────────────┼─────────────────┼──────────────────────┘
                         │
                         ▼
       ┌─────────────────────────────────────────────────────────────────────┐
       │                  META-COGNITIVE & GOAL SUBSTRATES                   │
       ├──────────────────────────────┬──────────────────────────────────────┤
       │ data/self_model.json         │ data/goals.db                        │
       │ - Identity & Focal Areas     │ - Directed Acyclic Graph             │
       │ - Competence Matrix          │ - Prerequisite Trees                 │
       │ - Known Strengths/Gaps       │ - Active Subgoals & State            │
       │ - reasoning_profile:         │                                      │
       │     global_scores            │                                      │
       │     domain_deltas            │                                      │
       │     strategy_index (M62)     │                                      │
       ├──────────────────────────────┼──────────────────────────────────────┤
       │ data/active_task.json        │ tests/benchmark_suite/               │
       │ - Deterministic Task FSM     │ - Immutable Benchmark Tests          │
       │ - strategy_label / prompt_   │ - Reasoning Benchmarks (M66)         │
       │   hash per-task (M67)        │ - Objective Ground Truth             │
       └──────────────────────────────┴──────────────────────────────────────┘
```

### Tier 1: Episodic & Working Memory (`data/episodic.db`)
- **Schema**: `id`, `session_id`, `trace_id`, `event_type`, `input_text`, `output_text`, `confidence_score`, `details_json`, `timestamp`, `prompt_hash`, `strategy_label`, `novelty_score`, `reasoning_domain`, `outcome_class`, `hypothesis_count`.
- **Extended fields (Mitigation #67 — Telemetry Hotfix)**:
  - `prompt_hash TEXT` — SHA-256 of the exact system prompt template injected for this turn. Enables exact-equality grouping of traces that ran under provably identical reasoning instructions. Indexable; the full prompt text is NOT stored (cost/size).
  - `strategy_label TEXT` — `NULL` | `'decomposition'` | `'hypothesis_competition'` | `'counterfactual'` | `'causal'` | `'planning'`. Written by the Strategy Injector at prompt-dispatch time, providing **ground truth** for domain delta computation rather than behavioral inference.
  - `novelty_score REAL` — `0.0–1.0`. Computed by the Planner as `1 - max_cosine_similarity_to_historical_task_embeddings`. Drives strategy selection and MoA routing; kept orthogonal from `complexity_score` (see M#66).
  - `reasoning_domain TEXT` — `'game_dev'` | `'cli_tools'` | `'web_apis'` | `'general'`. Propagated from the curriculum planner's domain tag.
  - `outcome_class TEXT` — `'success'` | `'failure'` | `'partial'` | `'inconclusive'`. Written by the result relayer after FSM transition (Mitigation #49); never self-reported by the skill.
  - `hypothesis_count INTEGER` — number of competing hypotheses generated (0 = no hypothesis competition active). Used by domain delta aggregation to identify hypothesis_competition episodes.
- **Purpose**: Tracks chronological interactions, user query turns, research milestones, skill execution logs, end-to-end trace telemetry (`Goal -> Prompt -> Tool -> Stderr -> State Mutation`), and explicit reasoning-strategy ground truth for domain delta computation (Mitigation #62).
- **Retention Policy**: Query logs auto-pruned after 90 days. Learning events, audit traces, benchmark logs, and reasoning episodes retained permanently.

### Tier 2: Semantic Memory (`data/semantic.db`)

Semantic memory stores two primary, complementary knowledge structures:

**Table: `semantic_facts`** (Atomic Knowledge Units)
- **Schema**: `id`, `topic`, `knowledge_type`, `statement`, `confidence`, `source_type`, `source_url`, `version`, `is_superseded`, `superseded_by`, `staleness_days`, `ingested_at`, `timestamp`, `embedding_json`.
- **`source_type` field**: `"seed"` | `"learned"` | `"user_corrected"`. Learned facts can supersede seed facts via contradiction gating. User-corrected facts have top authority (confidence: 1.0) and cannot be overwritten by automatic re-ingestion. `forget <topic>` preserves seed facts by default (use `forget <topic> --include-seeds` to remove all).
- **`knowledge_type` field** — 3 core categories (streamlined for V1 reliability):
  | Type | What It Captures | Example |
  |------|-----------------|--------|
  | `concept` | Definitions, core mental models, and "what/why/when" | *"A pull request is a proposal to merge changes from one branch into another."* |
  | `syntax` | Exact CLI command/API syntax with flags and parameters | *"`gh release create <tag> [--title <string>] [--notes <string>]`"* |
  | `troubleshooting` | Symptom → root cause → fix chains | *"If git push fails with 'non-fast-forward', the remote has commits you lack. Fix: git pull --rebase origin main."* |
- **Search**: Hybrid retrieval combining L2-normalized dense vector cosine similarity with sparse keyword/token matching (SQLite FTS5) via Reciprocal Rank Fusion (RRF).
- **Deduplication**: Facts with statement similarity >0.95 to an existing active fact are rejected as duplicates.

**Table: `context_passages`** (Long-Form Understanding)
- **Schema**: `id`, `topic`, `title`, `content` (200–500 words), `source_url`, `ingested_at`, `embedding_json`.
- **Purpose**: Preserves reasoning chains, nuanced multi-step explanations, and paragraph context that cannot be captured in atomic facts. Retrieved alongside facts when queries require depth.
- **Example**: A 300-word passage explaining Git's staging area lifecycle — how `git add` moves changes from working tree to index, why the index exists as an intermediate step, and how it relates to `git commit`.

### Tier 3: Procedural Memory (`skills/` + `data/skills.db`)
- **Schema**: `id`, `name`, `topic`, `runtime`, `language`, `description`, `usage_example`, `parameters_json`, `code`, `test_code`, `confidence`, `source_type`, `is_verified`, `verification_tier`, `verification_output`, `dependencies_json`, `created_at`, `embedding_json`.
- **Storage**: Source files stored directly in `skills/<topic>_<name>.<ext>` (topic-prefixed to prevent collisions). V1 default is Python (`.py`); see **Mitigation #36** for the multi-runtime extension.
- **Purpose**: Crystallizes verified, runnable tools that the agent can execute on demand or compose hierarchically. `verification_tier` records how the skill was proven: `mock` (static + mocked unit tests only), `real_local` (executed against the real local CLI/engine in a non-networked sandbox), or `real_external` (executed against an allowlisted external API or its deterministic local fixture — see **Mitigations #35/#51**).
- **Search**: Hybrid retrieval combining L2-normalized dense vector cosine similarity with sparse keyword/token matching (SQLite FTS5). Exact CLI syntax matches are boosted above pure semantic similarity.

### Tier 4: Project Memory (`data/projects.db`)
- **Purpose**: A persistent, structured model of the user's actual codebase(s) — the single largest gap between "a very smart encyclopedia" and "a software collaborator". Enables the agent to answer questions like *"Where is player movement handled in my Unity project?"* by searching actual files, and to make grounded edits with full context. Populated automatically whenever the agent reads or writes workspace files (Mitigation #50).
- **Tables**:

  **`projects`**
  - **Schema**: `project_id`, `name`, `root_path`, `description`, `runtime`, `created_at`, `updated_at`.
  - One row per indexed workspace (e.g., `e:\AI double`, a Unity project, a Godot project).

  **`project_files`**
  - **Schema**: `file_id`, `project_id`, `relative_path`, `absolute_path`, `file_hash`, `language`, `role_summary`, `semantic_summary`, `embedding_json`, `last_indexed_at`.
  - `file_hash` (SHA-256) detects changes; `role_summary` captures the file's architectural role (e.g., *"player controller MonoBehaviour"*); `semantic_summary` is embedded for semantic search. `UNIQUE(project_id, absolute_path)`.
  - Re-indexing is incremental: on read/write of a file, recompute the hash; only re-embed and re-summarize when the hash changed.

  **`project_decisions`**
  - **Schema**: `decision_id`, `project_id`, `title`, `decision`, `rationale`, `related_files_json`, `timestamp`.
  - Documents architecture decisions (e.g., *"Use JSON over binary for save files because Unity's JsonUtility is deterministic and debuggable"*) so the agent honors prior choices across sessions.

- **Write Protection**: The agent is **forbidden from directly editing `projects.db`** (same rule as `tests/benchmark_suite/`). All writes flow through `agent/memory/project.py` only. Direct modification by the agent is treated as a security violation and rolled back (Mitigation #50).
- **Search**: Participates in hybrid retrieval alongside `semantic_facts`, `context_passages`, and skills. Query intent that mentions a file, class, or project concept retrieves matching `project_files` rows (by path/keyword FTS5 and by semantic summary embedding), returning grounded, code-aware answers. (Mitigation #50, Subsystem 15.)

### Tier 2.5: Reasoning Memory (`data/reasoning.db`)
- **Purpose**: Bridges episodic logs (what happened) and semantic facts (what is true) with a new substrate: **how the agent thinks**. Records structured reasoning episodes — the full trajectory from initial hypothesis through failure diagnosis to generalized rule — enabling the Heartbeat to replay past failures, build a personal reasoning curriculum, and supply verified trajectories to the training-data pipeline (Mitigations #61–#70).

  **Table: `reasoning_episodes`**
  - **Schema**: `episode_id`, `trace_id` (FK → episodic), `domain`, `problem`, `initial_hypothesis`, `actions_json`, `observations_json`, `failure_mode`, `root_cause`, `corrected_strategy`, `generalized_rule`, `confidence`, `novelty_score`, `outcome_class`, `srt_json` (Structured Reasoning Trace, M#65), `verified` (bool — passed symbolic verifier), `timestamp`.
  - **Example record**:
    ```json
    {
      "domain": "game_dev",
      "problem": "Unity player movement jitter at high frame rates",
      "initial_hypothesis": "physics timestep mismatch",
      "actions": ["moved Rigidbody logic to FixedUpdate", "disabled interpolation", "tested varying frame rates"],
      "observations": ["jitter remained with interpolation disabled", "jitter disappeared after removing duplicate movement update"],
      "root_cause": "movement applied in both Update and FixedUpdate simultaneously",
      "generalized_rule": "Never drive Rigidbody movement from two update loops",
      "confidence": 0.94,
      "verified": true
    }
    ```
  - Only episodes with `verified = true` are eligible as training data (Mitigation #69). Unverified episodes are stored for debugging but flagged.

- **Retention**: Reasoning episodes are **retained permanently** (no TTL). High-novelty, verified episodes are the highest-value artifact in the system.

### Meta-Cognitive, Goal & Safety Substrates

#### 1. Persistent Self-Model (`data/self_model.json`)
- **Schema**:
  ```json
  {
    "identity": "Autonomous-Agent-v1",
    "boot_count": 42,
    "last_reflection_at": "2026-08-20T00:00:00Z",
    "current_focal_areas": ["improve git workflows", "docker networking"],
    "empirical_competence_matrix": {
      "git": {"skills_verified": 6, "skills_failed": 0, "pass_ratio": 1.0, "confidence": 0.95},
      "docker": {"skills_verified": 3, "skills_failed": 1, "pass_ratio": 0.75, "confidence": 0.78}
    },
    "known_strengths": ["git", "python_stdlib", "cli_scripting"],
    "known_knowledge_gaps": ["docker_compose_v2", "kubernetes_helm"],
    "reasoning_profile": {
      "global_scores": {
        "decomposition": 0.82,
        "hypothesis_testing": 0.61,
        "causal_reasoning": 0.74,
        "counterexample_gen": 0.47,
        "planning": 0.65
      },
      "domain_deltas": {
        "game_dev":  {"causal_reasoning": 0.12, "planning": 0.09},
        "cli_tools": {"decomposition": 0.07, "hypothesis_testing": -0.08},
        "web_apis":  {"planning": -0.11}
      },
      "strategy_index": {
        "novel_problem":      ["counterfactual", "hypothesis_competition"],
        "structured_problem": ["decomposition", "causal"],
        "debugging":          ["hypothesis_competition", "discriminating_test"]
      },
      "zpd_ceilings": {
        "decomposition": 0.82,
        "hypothesis_testing": 0.61
      }
    },
    "user_preferences": {
      "prefer_rebase_over_merge": true
    }
  }
  ```
- **Purpose**: Gives the agent persistent self-awareness across session boots, tracking empirical strengths, knowledge gaps, and focus areas (Mitigation #40). Anchored strictly in external metrics (Mitigation #45). Extended with `reasoning_profile` (cross-cutting cognitive profile) — **not** attached to individual skills to prevent domain overfitting (Mitigation #62). `domain_deltas` are posterior corrections on global priors computed weekly from `reasoning_domain` + `outcome_class` + `strategy_label` columns in episodic logs (Mitigation #67). `zpd_ceilings` record the highest difficulty percentile each reasoning skill can reliably clear, updated by the ZPD binary search (Mitigation #66).

#### 2. Directed Acyclic Goal Graph (`data/goals.db`)
- **Schema**: `id`, `title`, `description`, `priority` (1-10), `depth` (autonomous: max 2; supervised: max 4), `supervised` (bool, set when a subtree is approved to exceed depth 2), `status` (`pending`|`active`|`blocked`|`completed`|`failed`), `prerequisites_json`, `subgoals_json`, `completion_criteria`, `created_at`, `completed_at`.
- **Purpose**: Maintains long-term objectives and sub-goal prerequisite trees with hard tree-depth caps to prevent goal drift (Mitigation #42, #47). Autonomous background goals stay at depth 2; user-supervised goals may reach depth 4 **only** with explicit `[Y/n]` approval at each expansion past depth 2 (Mitigation #54).

#### 3. Deterministic Task State Machine (`data/active_task.json`)
- **Schema**: `task_id`, `trace_id`, `goal_id`, `state` (`PENDING` | `RUNNING` | `VERIFYING` | `COMMITTED` | `FAILED`), `step_index`, `max_steps` (5), `consecutive_failures` (max 2), `atomic_action`, `result_payload`, `strategy_label`, `prompt_hash`, `updated_at`.
- **Extended fields (Mitigation #67)**: `strategy_label` — the reasoning strategy injected for this task (copied from episodic at task spawn); `prompt_hash` — SHA-256 of the injected system prompt template. Both fields are written at task spawn time before the first action, so they are available to the result relayer when writing `outcome_class` back to episodic.
- **Purpose**: Persists active execution on disk to eliminate multi-hour context drift. On every turn, the agent reads current state, executes 1 atomic action, persists state mutation, and exits prompt (Mitigation #49).

#### 4. Immutable Benchmark Suite (`tests/benchmark_suite/`)
- **Purpose**: Immutable set of developer-written integration tests that the agent is strictly forbidden from modifying. Anchors the competence score objectively: $\text{Competence} = \frac{\text{Passed Benchmark Tests}}{\text{Total Benchmark Tests}}$ (Mitigation #45).

---

## 4. Production Pillars, Edge Cases & All 70 Mitigations

### Layer A: Build & Import Phase (Mitigations #1–#3)

#### Mitigation #1: Circular Import Prevention
- **Problem**: Python circular imports between `config → models → memory → brains → engine → orchestrator` cause `ImportError` on first run.
- **Fix**: Enforce strict one-directional dependency graph:
  ```
  config → models → memory/embeddings → memory/* → brains/base → brains/* → engine/* → orchestrator → cli/main
  ```
  No module may import from a module that depends on it. No backward imports. `config.py` and `models.py` import nothing from the agent package.

#### Mitigation #2: Embedding Dimension Upgrade Migration
- **Problem**: If the user changes `EMBEDDING_MODEL_NAME` in `config.py` (e.g. from MiniLM 384d to BGE 768d), the new embeddings will crash when compared against the old 384d vectors in SQLite, or when loading the `numpy` matrix.
- **Fix**: Store the `EMBEDDING_MODEL_NAME` in `episodic.db` (or a `metadata` table). On startup, if the active config model differs from the stored model, wipe the vectors and trigger a `re-seed` (for seed facts) and prompt the user to `refresh` learned topics. For V1 MVP, just check dimensions during matrix load:
  ```python
  if len(embedding) != self.expected_dim:
      raise ValueError("Model dimension mismatch. Please clear semantic.db.")
  ```

#### Mitigation #3: Workspace Paths with Spaces
- **Problem**: The workspace path `e:\AI double` contains a space. String-concatenated paths break `subprocess.run`, `PYTHONPATH`, and imports on Windows.
- **Fix**: Always use `pathlib.Path` for all path operations. Always quote paths in subprocess environment variables. Never use string concatenation for paths. Integration tests must explicitly test with space-containing paths.

---

### Layer B: Web Ingestion & Content (Mitigations #4–#7)

#### Mitigation #4: Ingestion Fetch Fallback & Abort Guard (No Synthesized Fallback Text)
- **Problem**: Historically, when Cloudflare or JavaScript hydration blocked the fetch stages, a final fallback "synthesized content from curriculum description" injected LLM guesses into semantic memory as ground truth.
- **Fix**: Implement a resilient fetch chain, but abort the curriculum unit if web content is <100 characters instead of generating synthetic fallback text. **The abort is a guard, not a fetch stage** — numbering is: Stage 1 Jina, Stage 2 Trafilatura, Stage 3 BeautifulSoup, Stage 4 (optional) Playwright JS render (Mitigation #39), then the abort guard:
  ```
  Stage 1: Jina Reader (r.jina.ai/<url>) → clean markdown from JS-rendered pages
  Stage 2: Direct HTTP GET + Trafilatura → article-level extraction
  Stage 3: Direct HTTP GET + BeautifulSoup → raw text extraction
  Stage 4: Optional Playwright/Chromium headless render (only if available, ≤30s/page — M39)
  ABORT GUARD: if all stages fail OR len(extracted_content.strip()) < 100 → abort unit
  ```
  ```python
  if len(extracted_content.strip()) < 100:
      logger.warning(f"Failed to fetch external ground truth for {url}.")
      return IngestionResult(success=False, error="Doc blocked or unreadable")
  ```

#### Mitigation #5: DuckDuckGo Search Rate Limiting
- **Problem**: DuckDuckGo returns empty results after ~20 rapid queries in succession.
- **Fix**: Cap search to 2 queries per curriculum unit. Add `time.sleep(1.0)` between search calls. If search returns zero results, proceed with the offline fallback (the learning loop must never stall on search failure).

#### Mitigation #6: Stale or Incorrect Web Content
- **Problem**: Ingested web content could be outdated, wrong, or irrelevant. Facts distilled from bad sources pollute semantic memory.
- **Fix**: Every `SemanticFact` carries mandatory `source_url`, `ingested_at` timestamp, and `staleness_days` (configurable, default: 365). Facts older than `staleness_days` are flagged for re-verification on next query. CLI command `refresh <topic>` re-runs ingestion and updates facts.

#### Mitigation #7: Long Documents Exceeding LLM Context Windows
- **Problem**: Full API reference pages (e.g., Docker CLI manual) can be 50,000+ characters, exceeding context limits.
- **Fix**: Chunk documents to **max 4000 characters** per distillation call. Process chunks sequentially through the brain. Deduplicate extracted facts: if a new fact has >0.95 cosine similarity to an existing active fact on the same topic, discard it as a duplicate.

---

### Layer C: LLM/Brain Integration (Mitigations #8–#11)

#### Mitigation #8: Malformed JSON from LLM Responses
- **Problem**: LLMs frequently return JSON wrapped in markdown fences, with trailing commas, or truncated mid-object.
- **Fix**: Implement robust `_extract_json()` method with 3-stage parsing:
  ```
  Stage 1: Strip ``` and ```json fences, try json.loads()
  Stage 2: Regex-extract outermost { ... } or [ ... ], try json.loads()
  Stage 3: On final failure, log the raw response and return a safe empty default
           (empty fact list, generic skill template) rather than crashing
  ```
  Never let a JSON parse error abort the entire learning loop.

#### Mitigation #9: API Rate Limits and Transient Failures
- **Problem**: 429 (rate limit), 500 (server error), and network timeouts during multi-step learning sessions.
- **Fix**: Wrap all brain API calls in a retry decorator: **3 retries, exponential backoff (1s, 2s, 4s)**. On final failure after 3 retries, log the error and continue the learning loop with partial results (e.g., skip one curriculum unit) rather than crashing the entire session.

#### Mitigation #10: Token Cost Awareness
- **Problem**: A single `learn` command on a broad topic could trigger 10+ API calls, generating unexpected costs.
- **Fix**:
  - Add `--dry-run` CLI flag that generates and displays the curriculum plan without executing ingestion or API calls.
  - After each learning session, print a summary: *"Learning session used ~X API calls across Y curriculum units."*
  - MockBrain is always free and should be the default.

#### Mitigation #11: MockBrain Knowledge Boundary
- **Problem**: MockBrain has hand-crafted fixtures only for GitHub, Docker, and generic topics. Other topics produce shallow, generic results.
- **Fix**: MockBrain must declare its known topics list. For unknown topics, return clearly labeled generic templates and print a warning: *"MockBrain has limited coverage for '{topic}'. Register a real provider in `brains.json` and select it with `--brain <provider>` for full autonomous research."* The learning loop still completes, but with lower-quality generic facts and skills.

---

### Layer D: Skill Synthesis & Testing (Mitigations #12–#16)

#### Mitigation #12: Tautological Mock Test False Positives
- **Problem**: Because the LLM writes both the tool code and the unit test mock, if the LLM hallucinated a non-existent CLI flag (e.g., `gh release make`), the mock assertion asserts that same bad flag. The test passes 100% despite the tool being broken.
- **Fix**: Validate synthesized command tokens against official CLI schemas extracted during Step 2 ingestion or run a fast dry-run check:
  ```python
  def validate_command_flags(synthesized_cmd: list[str], allowed_flags: set[str]):
      for token in synthesized_cmd:
          if token.startswith("--") and token not in allowed_flags:
              raise ValidationError(f"Synthesized unrecognized flag: {token}")
  ```

#### Mitigation #13: Generated Skills Depend on Uninstalled Packages
- **Problem**: A synthesized skill might `import requests` or `import yaml` when the user's environment doesn't have them installed.
- **Fix**: After AST parsing, extract all `import X` and `from X import Y` statements. Verify each top-level module exists via `importlib.util.find_spec(X)`. If a dependency is missing, flag it in `verification_output`: *"Warning: skill requires package 'X' which is not installed. Run `pip install X`."* The skill can still be registered but marked with a dependency warning.

#### Mitigation #14: Cross-Platform Shell & Executable Differences
- **Problem**: Skills using `subprocess.run(["gh", ...])` work on systems where `gh` is in PATH but fail on others. Path separators differ between Windows and Unix.
- **Fix**: Generated skills should use `shutil.which("gh")` to locate executables and raise a clear error if not found. All path operations must use `pathlib.Path`. Store platform requirements in skill metadata `dependencies_json`: `{"cli_tools": ["gh"], "packages": []}`.

#### Mitigation #15: Skill Name Collisions
- **Problem**: Two topics could both try to create `auth_login.py` (e.g., GitHub auth and Docker auth).
- **Fix**: All skill names are topic-prefixed: `github_auth_login`, `docker_auth_login`. The `name` column in `skills.db` has a `UNIQUE` constraint. If a name collision is detected, append a numeric suffix (`_2`, `_3`).

#### Mitigation #16: Orphaned Child Subprocesses on Windows (TimeoutExpired Leak)
- **Problem**: On Windows, `subprocess.run(timeout=5.0)` raises `TimeoutExpired` and terminates the parent Python process, but child processes spawned by CLI tools (e.g., background `gh` or docker daemons) remain orphaned in memory, locking open files and ports.
- **Fix**: Kill the entire process tree on timeout:
  ```python
  import os, subprocess


  def run_sandboxed_test(test_cmd: list[str], timeout: float = 5.0):
      try:
          return subprocess.run(test_cmd, capture_output=True, text=True, timeout=timeout)
      except subprocess.TimeoutExpired as exc:
          if os.name == "nt" and exc.pid:
              # Kill the entire process tree on Windows
              subprocess.run(
                  ["taskkill", "/F", "/T", "/PID", str(exc.pid)], capture_output=True
              )
          raise
  ```

---

### Layer E: Memory & Storage Scaling (Mitigations #17–#20)

#### Mitigation #17: In-Memory NumPy Vector Cache Invalidation Race
- **Problem**: If a background async worker runs `learn docker` and writes 25 facts to SQLite in WAL mode, the foreground query thread (`ask`) continues computing dot-product similarities against the stale in-memory NumPy matrix, causing newly learned knowledge to be invisible.
- **Fix**: Query SQLite's internal version counter (`PRAGMA data_version;`). If it changes, reload the matrix before running vector cosine search.
  ```python
  def _sync_vector_cache(self):
      cur = self.db.cursor()
      cur.execute("PRAGMA data_version;")
      current_ver = cur.fetchone()[0]
      if current_ver != self._cached_db_version or self._embeddings_matrix is None:
          self._reload_numpy_matrix()
          self._cached_db_version = current_ver
  ```

#### Mitigation #18: Unbounded Episodic Memory Growth
- **Problem**: After months of usage, `episodic.db` grows to hundreds of MB with routine query logs.
- **Fix**: Configurable retention policy in `config.py`:
  - `EPISODIC_RETENTION_DAYS = 90` (default).
  - On startup, auto-prune: `DELETE FROM episodic_records WHERE event_type = 'query' AND timestamp < (now - retention_days)`.
  - Learning events (`learned_topic`, `execute_skill`) are **never pruned** — they are permanent audit records.

#### Mitigation #19: Superseded Facts Accumulate
- **Problem**: Re-learning topics multiple times creates dead superseded rows that bloat `semantic.db`.
- **Fix**: Add `purge-superseded` CLI command that runs `DELETE FROM semantic_facts WHERE is_superseded = 1`. Periodic `VACUUM` after purge to reclaim disk space. Default behavior: keep superseded facts for audit trail (they are excluded from active queries by default).

#### Mitigation #20: SQLite FTS5 Concurrent Batch Lock Spikes
- **Problem**: Concurrent async curriculum ingestion workers writing to FTS5 shadow tables cause `sqlite3.OperationalError: database is locked`.
- **Fix**: Set a busy timeout of 15 seconds and `synchronous = NORMAL` on all connections:
  ```python
  def create_sqlite_conn(db_path: str) -> sqlite3.Connection:
      conn = sqlite3.connect(db_path, timeout=15.0)
      conn.execute("PRAGMA journal_mode = WAL;")
      conn.execute("PRAGMA busy_timeout = 15000;")
      conn.execute("PRAGMA synchronous = NORMAL;")
      return conn
  ```

---

### Layer F: Retrieval & Confidence Quality (Mitigations #21–#23)

#### Mitigation #21: Cold Start Detection
- **Problem**: On first ever interaction (empty memory), every question gets a meaningless `Confidence: 0.00` refusal.
- **Fix**: Detect cold start state (zero facts and zero skills in memory). Return a dedicated message:
  ```
  "My knowledge base is empty. Tell me what to learn first with `learn <topic>`."
  ```
  Do not display a misleading confidence score of 0.00.

#### Mitigation #22: Multi-Topic Queries
- **Problem**: A query like *"Compare GitHub Actions vs Docker CI"* spans two learned topics. Topic-filtered retrieval would miss half the context.
- **Fix**: Run retrieval in two modes:
  1. **Unfiltered query**: Retrieve top-k facts across all topics.
  2. **Topic-detected query**: If the query contains a known topic keyword, also retrieve topic-filtered results.
  Merge and deduplicate results. The discriminator check evaluates whether the combined context is sufficient to answer.

#### Mitigation #23: Topic Disambiguation
- **Problem**: *"learn python"* could mean the programming language or the snake. Broad topics like *"learn security"* are ambiguous.
- **Fix**: Brain's `plan_curriculum()` prompt includes an explicit instruction: *"Interpret the topic in its technical/software engineering context unless clearly otherwise."* For real brains (Gemini/Claude), add: *"If the topic is genuinely ambiguous, state your interpretation in the overview."* MockBrain defaults to the technical interpretation.

---

### Layer G: User Experience & Resilience (Mitigation #24)

#### Mitigation #24: Selective Forgetting (`forget <topic>`)
- **Problem**: No way to undo a learning session or remove bad/unwanted knowledge from memory.
- **Fix**: Add `forget <topic>` CLI command that:
  1. Deletes all `SemanticFact` rows where `topic = <topic>` from `semantic.db`.
  2. Deletes all `SkillDefinition` rows where `topic = <topic>` from `skills.db`.
  3. Deletes corresponding `skills/<topic>_*.py` files from disk.
  4. Records the deletion event in `episodic.db` for audit: `event_type = "forgot_topic"`.
  5. Prints confirmation: *"Forgot all knowledge about '{topic}': X facts removed, Y skills deleted."*

---

### Layer H: Critical Security & Quality Traps (Mitigations #25–#27)

#### Mitigation #25: Zero-Trust Positive-Match Skill Compiler (Replacing AST Scanning)
- **Problem**: Blacklist-based AST inspection is fundamentally broken against Python's dynamic nature. The original `SafeASTVisitor` banned specific reflection attributes (`__bases__`, `__subclasses__`, etc.), but Python provides an unbounded number of bypass vectors: `getattr(__builtins__, 'eval')`, `(lambda: __import__('os'))()`, `type('', (), {'__del__': lambda s: __import__('os').system('rm -rf /')})()`, `vars()`, `locals()['__builtins__']`, monkey-patching, descriptor protocol abuse, and any future Python syntax addition. A blacklist must enumerate all attacks; a whitelist must enumerate all *operations*. The whitelist is finite; the attack surface is not.
- **Fix**: Replace the AST scanner with a **zero-trust positive-match compiler** that inverts the security model. Instead of walking the AST looking for *bad* patterns (implicit allowlist — anything not banned passes), the compiler walks every AST node looking for *good* patterns only (explicit allowlist — anything not matched is rejected). The key mechanism: `generic_visit()` **defaults to rejection**.

  | Property | Old (Scanner) | New (Compiler) |
  |----------|---------------|----------------|
  | Default behavior for unknown nodes | **Pass** (implicit allow) | **Reject** (explicit deny) |
  | Security invariant | Must enumerate all attacks | Must enumerate all operations |
  | New Python syntax (e.g., `match`/`case`) | Silently allowed (potential bypass) | Rejected until explicitly handled |
  | Maintenance burden | Unbounded (whack-a-mole) | Bounded (finite opcode set) |

  ```python
  class SkillCompiler(ast.NodeVisitor):
      """Zero-trust positive-match compiler.
      Every allowed AST node type has an explicit visit_ method.
      Any node without a handler is rejected by generic_visit.
      The LLM never gets a retry — the proposal is rejected at compile-time."""

      ALLOWED_BUILTINS = frozenset(
          {
              "len",
              "int",
              "str",
              "float",
              "bool",
              "list",
              "dict",
              "set",
              "tuple",
              "range",
              "enumerate",
              "zip",
              "map",
              "filter",
              "sorted",
              "reversed",
              "min",
              "max",
              "sum",
              "abs",
              "round",
              "isinstance",
              "hasattr",
              "print",
              "repr",
              "type",
              "any",
              "all",
              "hex",
              "oct",
              "bin",
              "ord",
              "chr",
          }
      )

      ALLOWED_IMPORTS = ...  # Populated from Tiered Import Allowlist (Tier 1 + Tier 2)

      def generic_visit(self, node: ast.AST) -> None:
          """DEFAULT: REJECT. This is the entire security model."""
          raise CompilationError(
              f"Rejected: unrecognized AST node '{type(node).__name__}' "
              f"at line {getattr(node, 'lineno', '?')}. "
              f"Node type has no explicit compiler handler."
          )

      # --- Explicitly allowed structural nodes ---
      def visit_Module(self, node):
          self._visit_children(node)

      def visit_FunctionDef(self, node): ...  # validates name, args, decorators
      def visit_AsyncFunctionDef(self, node):
          raise CompilationError("async not allowed")

      def visit_Return(self, node):
          self._visit_children(node)

      def visit_Assign(self, node):
          self._visit_children(node)

      def visit_AugAssign(self, node):
          self._visit_children(node)

      def visit_AnnAssign(self, node):
          self._visit_children(node)

      def visit_If(self, node):
          self._visit_children(node)

      def visit_For(self, node):
          self._visit_children(node)

      def visit_While(self, node):
          self._visit_children(node)

      def visit_With(self, node):
          self._visit_children(node)

      def visit_Raise(self, node):
          self._visit_children(node)

      def visit_Try(self, node):
          self._visit_children(node)

      def visit_ExceptHandler(self, node):
          self._visit_children(node)

      def visit_Pass(self, node):
          pass

      def visit_Break(self, node):
          pass

      def visit_Continue(self, node):
          pass

      # --- Expression nodes (each explicitly validated) ---
      def visit_Call(self, node):
          """Only allow calls to ALLOWED_BUILTINS or previously-compiled functions."""
          if isinstance(node.func, ast.Name):
              if (
                  node.func.id not in self.ALLOWED_BUILTINS
                  and node.func.id not in self._compiled_function_names
              ):
                  raise CompilationError(
                      f"Call to unrecognized function: '{node.func.id}'"
                  )
          elif isinstance(node.func, ast.Attribute):
              self._validate_method_call(node)  # checks module.method against allowlist
          else:
              raise CompilationError(
                  f"Dynamic call expression not allowed at line {node.lineno}"
              )
          self._visit_children(node)

      def visit_Attribute(self, node):
          """Reject ALL dunder attributes — no exceptions."""
          if node.attr.startswith("__") and node.attr.endswith("__"):
              raise CompilationError(f"Dunder attribute access rejected: '.{node.attr}'")
          self._visit_children(node)

      def visit_Import(self, node):
          self._validate_imports(node)

      def visit_ImportFrom(self, node):
          self._validate_imports(node)

      # ... visit_Constant, visit_Name, visit_BinOp, visit_Compare, etc.
      # Every primitive expression type gets an explicit handler.

      def _visit_children(self, node):
          """Recursively visit children — each child hits generic_visit if unhandled."""
          for child in ast.iter_child_nodes(node):
              self.visit(child)
  ```

  **Critical properties**:
  1. The compiler outputs the **same Python code** (or a sanitized subset) — no IR or bytecode compilation required. The security comes from the positive-match walk, not from a target language change.
  2. The Tiered Import Allowlist (Tier 1/2/3) survives as input to `_validate_imports()`. Tier 1 modules pass; Tier 2 modules pass only inside `FunctionDef` nodes; Tier 3 and unknown modules are rejected.
  3. **No retry on rejection**: if the compiler rejects a proposal, the skill is marked `compilation_failed` and the synthesizer's revision loop (Mitigation #35) may attempt a new generation (up to 2 retries), but the *same code* is never re-submitted — the LLM must produce a fundamentally different skill.
  4. Future Python syntax additions (e.g., new AST node types in Python 3.14+) are **automatically rejected** until the compiler is updated with an explicit handler. This is the correct default — new syntax is untrusted until audited.
- **Enforcement Location**: `engine/validator.py` (replaces the old `SafeASTVisitor` and `check_ast()`).

#### Mitigation #26: RAG-Based Skill Context Injection (Fixing Voyager Context Collapse)
- **Problem**: The original design injects **all** existing skill signatures from `skills/` into the synthesis prompt. When the skill library grows to 40+ skills, this consumes 8,000–12,000 tokens of irrelevant context. The LLM loses track of the main synthesis instruction, forgets parameter requirements, and produces degraded code quality.
- **Fix**: Replace `get_available_skill_signatures()` (which loads all skills) with a **RAG retrieval pipeline** in `synthesizer.py`:
  1. Before synthesizing a new skill, query `skills.db` vector store with the current `CurriculumUnit.title + description` as the search query.
  2. Retrieve the **top 3 most relevant** skills by cosine similarity.
  3. Only include skills with similarity score **> 0.50** (below that, the skill is irrelevant to the task).
  4. Inject **compressed signatures only**: function name + one-line description + parameter types. No full docstrings or implementation code.
  5. Enforce a hard **≤800 token budget** for skill context injection.
  6. If zero skills score above 0.50, inject no skill context — the new skill is standalone.
- **Enforcement Location**: `engine/synthesizer.py` — the `synthesize_skills_for_curriculum()` method must call `self.skill_library.find_skill(query, k=3)` instead of `self.skill_library.get_available_skill_signatures()`.
- **Context Format**:
  ```
  # You may import and reuse these previously verified skills:
  # from skills.github_auth_login import github_auth_login(token: str) -> bool  # Authenticates via gh auth
  # from skills.github_create_pr import github_create_pr(title: str, body: str, base: str) -> dict  # Creates a PR
  ```

#### Mitigation #27: Trusted Docs Registry & Domain Authority Scoring (Fixing Poisoned Web Search)
- **Problem**: DuckDuckGo search for technical queries (e.g., "Docker API") frequently returns SEO-optimized spam blogs, outdated Medium articles from 2018, or scraped content farms before official documentation. If the agent ingests a 2018 tutorial, it synthesizes outdated tools that may pass sandbox tests (since both code and tests are generated from the same bad source) but fail in the real world. The fact deduplicator (>0.95 similarity) then fiercely protects this garbage from being overwritten.
- **Fix**: Implement a **2-stage source quality pipeline** in `ingest.py`:

  **Stage 1: Trusted Docs Registry (Direct Fetch, No Search)**
  Maintain a `TRUSTED_DOCS` dictionary mapping known topics to their official documentation URLs:
  ```python
  TRUSTED_DOCS = {
      "github": ["https://cli.github.com/manual/", "https://docs.github.com/en"],
      "docker": ["https://docs.docker.com/reference/cli/docker/"],
      "python": ["https://docs.python.org/3/library/"],
      "kubernetes": ["https://kubernetes.io/docs/reference/"],
      "git": ["https://git-scm.com/docs"],
      "nodejs": ["https://nodejs.org/api/"],
      "rust": ["https://doc.rust-lang.org/book/"],
      "aws": ["https://docs.aws.amazon.com/cli/latest/reference/"],
  }
  ```
  For topics with entries in `TRUSTED_DOCS`, fetch these URLs **directly** via the Jina Reader / Trafilatura fallback chain — **skip web search entirely**. This guarantees official, current documentation.

  **Stage 2: Domain Authority Scoring (Fallback for Unknown Topics)**
  If the topic has **no entry** in `TRUSTED_DOCS`, run DuckDuckGo search and score each result:

  | Domain Pattern | Score Modifier | Rationale |
  |---------------|---------------|----------|
  | `*.gov`, `*.edu`, `*.org` | **+0.3** | Institutional authority |
  | Official `docs.*` subdomains | **+0.3** | Official documentation |
  | `github.com/<org>/` repos | **+0.2** | Source code & READMEs |
  | `stackoverflow.com` | **+0.1** | Community-vetted answers |
  | `medium.com`, `dev.to` | **−0.3** | Unvetted blog content |
  | `w3schools.com`, `geeksforgeeks.org` | **−0.3** | Often inaccurate/outdated |
  | `blogspot.*`, `wordpress.com` | **−0.5 (filtered)** | Generic blogs, SEO farms |

  Additionally:
  - **Freshness Check**: If Trafilatura / `htmldate` extracts a publication date older than **2 years** for fast-moving tools (Docker, K8s, Node.js, Rust), apply an additional **−0.2 penalty** or skip the page entirely.
  - Take the **top 2 highest-scored** results and proceed with the fallback fetch chain.

- **Enforcement Location**: `engine/ingest.py` — the `ingest_curriculum()` method must check `TRUSTED_DOCS` first, then fall back to scored DDG search.
- **Extensibility**: Users can extend `TRUSTED_DOCS` by adding entries to a `trusted_docs.json` file in the `data/` directory, which is loaded at startup and merged with the built-in registry.

---

### Layer I: Bootstrap & Cold Start (Mitigation #28)

#### Mitigation #28: Foundational Knowledge Pre-Seeding
- **Problem**: Even with cold start detection (Mitigation #21), the agent has zero useful capability on first boot. The user must immediately run `learn git` and `learn python` for basic developer tasks — requiring web access, API calls, and time — before the agent can answer even trivial questions like "How do I check git status?" or "How do I read a JSON file in Python?"
- **Fix**: Ship a curated `seed_data/` directory containing pre-written foundational knowledge and skills. On first boot (or when `semantic.db`/`skills.db` is empty), `agent/memory/seeder.py` automatically ingests this baseline.

  **Seed Facts (`seed_data/facts.json`)**:
  A curated set of **50–100 high-accuracy, atomic semantic facts** covering foundational developer knowledge:

  | Category | Example Facts | Count |
  |----------|-------------|-------|
  | **Git & Version Control** | `git add -A` stages all changes, `git log --oneline` shows compact history, `git stash` saves uncommitted work | ~15 |
  | **Python Standard Library** | `json.loads()` parses a JSON string, `pathlib.Path` handles cross-platform paths, `subprocess.run()` executes external commands | ~20 |
  | **CLI & Shell Fundamentals** | `--help` shows command usage, `|` pipes stdout to another command, `>` redirects output to a file | ~10 |
  | **HTTP & JSON** | `GET` retrieves resources, `POST` creates resources, HTTP 200 = success / 404 = not found | ~10 |
  | **File System & OS** | `os.getcwd()` returns current directory, `shutil.copy2()` preserves metadata | ~10 |

  Each fact carries:
  - `source_type: "seed"` — distinguishes from learned facts.
  - `source_url: "builtin://seed_data/v1"` — clearly labeled as built-in.
  - `confidence: 0.95` — high but not 1.0, allowing learned facts to supersede.
  - `topic`: Categorized as `"git"`, `"python"`, `"cli"`, `"http"`, or `"filesystem"`.

  **Seed Skills (`seed_data/skills/`)**:
  8 pre-written, standalone, tested Python tools for essential developer tasks:

  | Skill File | Purpose | Key Imports (Tier 2) |
  |-----------|---------|---------------------|
  | `stdlib_read_file.py` | Read file contents with encoding detection | `pathlib` (Tier 1 only) |
  | `stdlib_write_file.py` | Write/append to files with backup option | `pathlib`, `shutil` (Tier 2) |
  | `stdlib_list_directory.py` | List directory contents with filtering | `pathlib` (Tier 1 only) |
  | `stdlib_system_info.py` | Get OS, Python version, environment info | `os` (Tier 2) |
  | `stdlib_git_status.py` | Run `git status`, `git log`, `git diff` | `subprocess` (Tier 2) |
  | `stdlib_http_get.py` | Perform HTTP GET with timeout and error handling | `requests` (Tier 2) |
  | `stdlib_json_parse.py` | Parse, query, and validate JSON data | `json` (Tier 1 only) |
  | `stdlib_run_command.py` | Safe subprocess wrapper with timeout and output capture | `subprocess` (Tier 2) |

  Each seed skill:
  - Includes a companion `test_<name>.py` with mocked unit tests.
  - Passes the full Tiered Import Allowlist validation (Mitigation #25) — **no security backdoor**.
  - Is registered with `source_type: "seed"` in `skills.db`.

  **Seeder Module (`agent/memory/seeder.py`)**:
  - Runs automatically on startup if `semantic.db` has zero non-superseded facts OR `skills.db` has zero verified skills.
  - **Does NOT ship pre-computed embeddings** — computes L2-normalized vectors at boot time using the active embedding model. This prevents model-mismatch bugs if the user changes embedding backends.
  - Validates each seed skill through `validator.py` before registration (same Tiered Import Allowlist, same sandbox test).
  - Idempotent: uses fact deduplication (>0.95 similarity) and `INSERT OR IGNORE` to prevent double-seeding.
  - Records a `seeded_knowledge` event in episodic memory with `seed_version`.

  **Seed Versioning**:
  - `config.py` defines `SEED_VERSION = "1.0"`. If `SEED_VERSION` changes (e.g., updated facts or new skills), the seeder re-runs on next startup and supersedes outdated seed facts.
  - Old seed facts are marked `is_superseded = 1, superseded_by = "seed_v<new>"` — not deleted.

  **Impact on Core Architecture**:
  - **Retriever (Mitigation #21)**: Cold start now means "seeded but no learned topics" instead of "completely empty". The welcome message changes to: *"I have foundational knowledge about Git, Python, CLI, and HTTP. Ask me anything, or teach me a new topic with `learn <topic>`."*
  - **Forget command (Mitigation #24)**: `forget git` removes only `source_type: "learned"` facts by default. Use `--include-seeds` flag to also remove seed facts.
  - **Stats command**: Shows breakdown: *"52 seed facts, 28 learned facts, 8 seed skills, 2 learned skills."*

---

### Layer J: Answer Integrity & Memory Correction (Mitigations #29–#31)

#### Mitigation #29: Closed-World Generation Guard (Fixing the ≥0.80 Hard-Pass Hallucination)
- **Problem**: When vector similarity clears ≥0.80, the system directly passes retrieved facts to the brain for answer generation. But the brain can extrapolate beyond the provided facts — if a user asks *"What's the max upload size for GitHub releases?"* and the retrieved facts discuss releases but don't contain the specific size limit, the brain may hallucinate a plausible number.
- **Fix**: Even in the hard-pass zone, enforce a **strict closed-world constraint** in every answer generation prompt:
  ```
  "Answer the question using ONLY the provided facts. If the facts discuss
  the topic but do NOT contain the exact specific detail asked (e.g., specific
  limits, parameters, exact version numbers, or precise flags), state:
  'I found related information about [Topic], but my stored memory does not
  contain that specific detail.' NEVER extrapolate, estimate, or guess."
  ```
  This makes the retrieval pipeline three-stage: **similarity gate → fact retrieval → closed-world grounded generation**.
- **Enforcement**: `engine/retriever.py` must inject the closed-world constraint into every brain call, regardless of confidence zone. The brain's response must be grounded, never generative.

#### Mitigation #30: Hybrid Search Formula Penalty on Paraphrased Queries
- **Problem**: The additive formula `0.6 * dense + 0.4 * sparse` produces false penalties (e.g. `0.54`) when BM25 score is 0.0 (e.g. asking "How to ship local commits to remote?" for `git push`), which drops the score below the 0.65 threshold and triggers a false refusal. It also allows sparse matches to falsely elevate irrelevant dense scores.
- **Fix**: Decouple Candidate Ranking from Confidence Gating. Use RRF strictly as a Rank Aggregator to select the Top-K candidates. Use the Top-1 candidate's L2-normalized dense cosine similarity for the Gating Metric ($0.65$ / $0.80$):
  ```python
  def reciprocal_rank_fusion(
      dense_ranks: dict[str, int], sparse_ranks: dict[str, int], k: int = 60
  ) -> list[tuple[str, float]]:
      rrf_scores = {}
      for doc_id, rank in dense_ranks.items():
          rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
      for doc_id, rank in sparse_ranks.items():
          rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
      return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
  ```

#### Mitigation #31: In-Place Memory Correction (`correct` Command)
- **Problem**: If the agent gives a wrong answer, the only fix is `forget <topic>` → `learn <topic>`, which destroys all knowledge for the topic and re-crawls everything. No way to surgically fix a single fact.
- **Fix**: Add `correct <statement>` and `correct <topic> "<old_query>" -> "<new_statement>"` CLI command:
  1. Finds the fact that was most recently used in an answer (from episodic memory's last retrieval log) or by semantic search for the old statement.
  2. Marks the old fact `is_superseded = 1, superseded_by = "user_correction"`.
  3. Inserts the new fact with `confidence = 1.0` and `source_type = "user_corrected"`.
  4. `user_corrected` facts have **top authority** — they can never be automatically superseded by `learn` re-ingestion or seed updates. Only another `correct` command or explicit deletion can override them.
  5. Records the correction event in episodic memory for audit.
- **Enforcement**: `cli.py` parses the correction syntax. `engine/orchestrator.py` executes the supersede + insert. `memory/semantic.py` respects `user_corrected` authority during contradiction gating.

---

### Layer K: Deep Understanding & Knowledge Depth (Mitigations #32–#34)

#### Mitigation #32: 3-Category Over-Engineering Prune (Taxonomy Distillation)
- **Problem**: Extracting 7 categories $\times$ 3 depth levels causes the LLM to hallucinate boundaries, split atomic concepts unnecessarily, and bloat the prompt. Most queries only need Concepts, Syntax, or Troubleshooting.
- **Fix**: Prune to 3 core types and drop `depth_level` for V1.
  ```
  1. CONCEPT (Definitions, architectures, and "what/why/when")
  2. SYNTAX (Exact CLI commands, code snippets, or procedures)
  3. TROUBLESHOOTING (Symptom -> Cause -> Fix)
  ```

#### Mitigation #33: Context Passages (Long-Form Semantic Memory)
- **Problem**: Some knowledge cannot be atomized without losing meaning. A 300-word explanation of Git's staging area — how `git add` moves changes from working tree to index, why the index exists as an intermediate step, and how it relates to `git commit` — loses its coherence when split into 6 atomic facts. The relationships between sentences carry the understanding.
- **Fix**: During ingestion, alongside atomic fact extraction, the synthesizer also identifies and stores **context passages** — self-contained 200–500 word explanatory blocks that preserve reasoning chains, nuance, and narrative flow.
  - The distillation prompt includes: *"If a section contains a coherent multi-paragraph explanation, decision framework, or troubleshooting walkthrough that would lose meaning if split into atomic facts, also extract it as a CONTEXT_PASSAGE with a descriptive title."*
  - Context passages are stored in the `context_passages` table with their own embeddings.
  - During retrieval, both `semantic_facts` and `context_passages` are searched. If a context passage scores above the confidence threshold, it is included in the answer context alongside atomic facts.
  - This gives the brain the full reasoning chain needed to generate deep, accurate answers.
- **Storage budget**: Max 10 context passages per curriculum unit (to prevent bloat).

#### Mitigation #34: 1-Hop Knowledge Graph Removal (V1 Stabilization)
- **Problem**: The `concept_relationships` table and 1-hop graph traversal look great on paper but in practice, they pull in too much noisy context during retrieval, confusing the LLM and breaking the "closed-world" guarantee. It is over-engineered for an MVP.
- **Fix**: Completely remove the `concept_relationships` table, edge extraction, and graph traversal logic from the V1 build. Rely entirely on hybrid FTS5/Dense search over `semantic_facts` and `context_passages`.

---

### Layer L: Verification Integrity & Real-World Grounding (Mitigations #35–#39)

> These mitigations address the documented V1 weaknesses: tautological mock tests, the missing real-environment feedback loop, Python-only skill synthesis, brittle fixed thresholds, and the fact that AST checks alone are not OS-level isolation.

#### Mitigation #35: Real-Environment Verification Loop (Breaking the Tautological Mock Test)
- **Problem**: The same LLM generates both the tool code and its mocked unit tests. If the LLM hallucinates a non-existent command (e.g., `gh release make`), the mock test asserts that same bad command, passes 100%, and registers a skill that is broken in the real world. Mock verification is *evidence of internal consistency, not evidence of real-world correctness*. This is also why the Voyager loop cannot converge: Voyager works because it receives feedback from a real environment (Minecraft), while this system's skills were only ever proven against their own mocks.
- **Fix**: Add a **verification tier ladder** — a skill earns a higher tier only when it is proven against a real tool, not just its own test. The `real` tier is split into `real_local` and `real_external` because an external API cannot be verified in a non-networked sandbox (Mitigation #51):
  | Tier | Evidence | Meaning |
  |------|----------|---------|
  | `compiled` | Zero-trust positive-match compiler passes (Mitigation #25) | Every AST node explicitly handled; no unrecognized constructs |
  | `static` | Compiled + tiered import checks pass (Mitigation #25, #13) | Safe to parse, all imports in allowlist, dependencies available |
  | `mock` | Mocked `unittest` suites pass (existing V1 path) | Internally consistent; **not** real-world proven |
  | `real_local` | Skill executes against a **real local** CLI/compiler/engine in a gVisor sealed sandbox (git, docker, python, dotnet, godot...) against deterministic fixtures | Proven correct-by-execution locally |
  | `fixture_verified` | Executes against an offline schema-validating local HTTP fixture server | Proven against official schema |
  | `real_external` | Executes against a live external network endpoint via an allowlisted proxy with test credentials (GitHub, Docker Hub, email, Wikipedia...) through an allowlisted network domain + sandbox/test account, **or** against a deterministic local fixture server that emulates the API under schema validation | Proven against the external service (or a schema-faithful emulation) |
  | `pure_deterministic` | *(Optional, for side-effect-free skills only)* Two runs with different PRNG seeds produce identical output hashes | Verified deterministic; no time-based side-channels or pointer leaks |

  1. **Official schema anchors (Mitigation #12, extended)**: During Step 2 ingestion, capture the official command surface (`gh --help`, `gh release --help`, `docker --help`) into the fact store. `validate_command_flags()` rejects any synthesized `--flag` not present in the official schema before tests even run.
  2. **Real dry-run execution**: When a compatible runtime is available (Mitigation #38), run the skill against the real tool in a **read-only gVisor sealed sandbox** (Mitigation #38) against deterministic fixtures (a freshly initialized local git repo, a local mock HTTP server, a temp staging directory). Local CLIs verify as `real_local` with no network. External-API skills verify as `real_external` only when a network allowlist entry exists for the target domain (Mitigation #51) and a sandbox/test account is configured; otherwise they remain at `mock` and are labeled *"requires network verification"*.
  3. **Feedback-driven revision (the actual Voyager loop)**: If real execution fails, capture the real stderr/return code, feed it back to the synthesizer with the failing test output, and regenerate the skill. A skill is retried at most **2** times; if it still fails real execution it is registered as `is_verified = 0` with `verification_tier = failed` and is **never** presented as verified.
  4. **Confidence semantics**: Skills at `mock` tier are listed with a persistent *"mock-verified, not real-executed"* label; only `real_local`/`real_external`-tier skills are eligible for hierarchical composition as dependencies (Mitigation #26).
  5. **Pure-deterministic verification (optional sublevel)**: Skills with **no Tier 2 imports** (`subprocess`, `os`, `shutil`, `requests`) whose compiler analysis confirms all operations are side-effect-free are eligible for deterministic verification. The host runs the skill **twice** inside identical gVisor sandboxes with different pseudo-random seeds on the same input, then compares the cryptographic hash (SHA-256) of the output buffer between runs. If hashes match, the skill earns `determinism_verified: true`. If hashes differ, the skill is **flagged for human review** with `determinism_warning: "Non-deterministic output detected"` — it is **not** auto-quarantined (auto-quarantine on hash mismatch is a denial-of-service vector: even benign sources of non-determinism like `dict` iteration order or floating-point rounding would block the entire skill registry). Quarantine is always a human decision.
- **Enforcement Location**: `engine/validator.py` (real dry-run executor, determinism checker), `engine/synthesizer.py` (revision loop), `engine/orchestrator.py` (tier assignment and gating).

#### Mitigation #36: Multi-Runtime & Multi-Language Skill Synthesis (Game-Dev Ready)
- **Problem**: V1 synthesizes Python CLI tools only. Game programming requires C# (Unity), GDScript (Godot), C++ (Unreal), and shader languages (HLSL/GLSL). A game-developer user would be stuck synthesizing Python wrappers that cannot touch engine APIs, and — critically — would get **no real environment feedback** (does the script compile? does the level load?) from a Python-only pipeline.
- **Fix**: Parameterize synthesis by runtime via a `RUNTIMES` registry in `config.py`:
  | Runtime | Language | Extension | Compile/Run Smoke Test |
  |---------|----------|-----------|------------------------|
  | `python` (V1 default) | Python | `.py` | `python -m unittest` (mock) + real dry-run |
  | `dotnet` (V2) | C# | `.cs` | `dotnet build` + `dotnet run` smoke |
  | `godot` (V2) | GDScript | `.gd` | headless `godot --headless --script` load check |
  | `cpp` (V2) | C++ | `.cpp` | `g++/cl` compile + run smoke |
  | `shader` (V2) | HLSL/GLSL | `.hlsl/.glsl` | `glslc`/`dxc` compile check |
  1. `skills.db` gains `runtime` and `language` columns (Mitigation #3 schema update); skill files are stored as `skills/<topic>_<name>.<ext>`.
  2. The synthesizer prompt and code templates are per-runtime; RAG skill-context (Mitigation #26) filters by matching runtime.
  3. **Real-environment feedback loop is the core requirement**: for game engines the smoke test is the environment feedback — *"does the script compile, does the player move, does the level load"*. Skills that fail the engine smoke test are fed back to the synthesizer for revision (same loop as Mitigation #35).
  4. V1 ships `python` only; `dotnet`, `godot`, `cpp`, `shader` are gated behind `--runtime` with explicit *"experimental, requires engine toolchain on PATH"* warnings.
- **Enforcement Location**: `engine/synthesizer.py` (runtime templates), `engine/validator.py` (per-runtime smoke tests), `memory/procedural.py` (runtime columns + indexing).

#### Mitigation #37: Calibrated Confidence Thresholds (Replacing Hard-Coded Gates)
- **Problem**: The gates 0.65 (grey floor) and 0.80 (hard pass) are arbitrary. Cosine similarity distributions depend heavily on the embedding model (MiniLM 384d vs BGE 768d vs multilingual models) and on corpus vocabulary. The same thresholds can produce false refusals (paraphrased queries dropping below 0.65) or false confidence (unrelated-but-similar queries crossing 0.80). There is no evaluation set, so the gates are unvalidated.
- **Fix**: Make the gates **calibrated data, not constants**:
  1. **Ship a labeled evaluation set** `calibration/queries.json` with the four categories from `threshold_sweep.py`: `true_hit` (must pass), `related_unknown` (must land in grey → closed-world refusal), `opposing` (must match the correct specific fact), `unrelated` (must reject).
  2. **`calibrate-thresholds` CLI command** runs the precision/recall sweep (reference: `threshold_sweep.py`, `probe_bench.py`) against the active embedding model and current corpus, then writes the chosen `THRESHOLD_GREY` / `THRESHOLD_PASS` values to `data/calibration.json`.
  3. **Model-keyed calibration**: calibration results are stored per `EMBEDDING_MODEL_NAME` (ties into Mitigation #2 — changing the embedding model invalidates old calibration and triggers re-calibration at seed time).
  4. `set-threshold <val>` (existing CLI) overrides the calibrated values at runtime for experimentation, but the calibrated defaults persist.
  5. Until calibration has run, the system uses the documented starting points (0.65 / 0.80) **and labels them as uncalibrated** in `stats`.
- **Enforcement Location**: `cli.py` (`calibrate-thresholds`), `memory/semantic.py` (calibrated thresholds), `engine/retriever.py` (zone checks read from `data/calibration.json`).

#### Mitigation #38: gVisor Sealed Execution Envelope (Zero-Trust OS-Level Isolation)
- **Problem**: The positive-match compiler (Mitigation #25) catches code-level attacks at the AST, but it is not OS-level isolation. Tier-2 `subprocess` calls still need to execute somewhere, and the compiler cannot prevent runtime exploits in CPython itself, native `.so` libraries, or kernel 0-days. Docker/containers share the host kernel; a single 0-day in seccomp or runc (e.g., CVE-2024-21626, CVE-2019-5736) is catastrophic. The previous specification mentioned "Job Objects / bwrap / Docker" without defining the actual isolation boundary, resource limits, or kill semantics.
- **Fix**: Layer five independent isolation mechanisms in a **zero-trust execution envelope**. The **real-verification tier (Mitigation #35) and `run-skill` must execute inside these layers**, never bare on the host:

  - **Layer 0 — Compilation Gate**: Zero-trust positive-match compiler (Mitigation #25). Rejects any AST node without an explicit handler. This is the first gate — most malicious proposals die here without spawning a sandbox at all.

  - **Layer 1 — gVisor Process Isolation (Linux) / Job Object (Windows Fallback)**:
    Run skill execution inside **gVisor** (`runsc`), which virtualizes the entire Linux syscall table via a userspace kernel (Sentry). Unlike raw Docker/containers that share the host kernel with seccomp filtering, gVisor intercepts syscalls at the application-kernel boundary — a container escape hits another sandboxed kernel, not the host. This provides ~80% of WebAssembly's memory isolation guarantees without rewriting the skill pipeline.

    **gVisor gofer hardening** (critical — the gofer process mediates file I/O and introduces a subtle attack surface: file descriptor leakage and host mount traversal):
    - **No host bind-mounts. Period.** The gVisor rootfs is a **read-only squashfs** (or overlay) built at container build time from a tiered sandbox image (see Mitigation #60). A malicious skill using `os.open("../../../etc/passwd")` cannot escape the intended root because there is no host directory to traverse into.
    - **Input via stdin only**: If the skill needs input data, the host pipes it via stdin as a JSON payload (within the `RLIMIT_AS` memory cap). The host never bind-mounts a data directory.
    - **Output via stdout only**: The skill writes structured results to stdout. The host captures this via `runsc exec` after the container exits — never via a bind-mounted file. The typed result relayer (Mitigation #49) parses the captured stdout deterministically.
    - **Gofer flags**: `--host-uds=false` (prevents the skill from creating Unix domain sockets that could trick the gofer), `--net-raw=false` (prevents raw socket creation), `--file-access=exclusive` (prevents host filesystem caching bugs that could leak state between sandbox runs).
    - **Explicitly disabled**: gofer host-network proxying, gofer host-filesystem proxying.

    **Windows fallback**: A Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and a restricted token (no admin, no high-integrity). Killing the Job Object on timeout kills the whole tree (extends Mitigation #16). Windows Job Objects **cannot** enforce filesystem mount options (`noexec`, `nosuid`) or network namespaces; on Windows the sealed envelope is weaker and requires Docker Desktop or WSL2 for full isolation. This is a documented V1 boundary.

  - **Layer 2 — Sealed Resource Envelope**:
    Hard resource limits enforced by the OS/gVisor before the skill process starts. The skill is **SIGKILL'd** on any violation — no graceful shutdown, no cleanup, no chance to exfiltrate on the way out.

    | Resource | gVisor Limit (Primary) | Host RLIMIT (Failsafe) | Enforcement | Kill Behavior |
    |----------|----------------------|----------------------|-------------|---------------|
    | CPU time | 2 seconds wall-clock | 4 seconds (`RLIMIT_CPU`) | gVisor `--watchdog-action=panic` | SIGKILL — host RLIMIT is 2× gVisor limit as failsafe only |
    | Address space | 128 MB (`--memory-limit`) | 256 MB (`RLIMIT_AS`) | gVisor memory cgroup (primary enforcer) | OOM kill by gVisor; host RLIMIT is 2× as failsafe to avoid confusing double-enforcement SIGKILL propagation |
    | Filesystem | gVisor internal tmpfs, `noexec,nosuid,nodev`, 64 MB cap | N/A (no host mount) | `mount -t tmpfs -o size=64m,noexec,nosuid,nodev` inside gVisor | Write beyond cap returns `ENOSPC` |
    | Network | `--network=none` (loopback only, all outbound dropped) | `iptables -A OUTPUT -j DROP` (host-level failsafe) | gVisor network stack (primary); host iptables (failsafe) | Packets silently dropped at kernel level |
    | PIDs | 32 max (`--pid-limit`) | `RLIMIT_NPROC` = 32 | gVisor cgroup `pids.max` | Fork bomb → `EAGAIN` on `clone()` |
    | Open files | 64 max | `RLIMIT_NOFILE` = 64 | gVisor enforced | `EMFILE` on `open()` |

    > **RLIMIT / gVisor interaction note**: The host `RLIMIT_AS` and `RLIMIT_CPU` are set **looser** (2×) than gVisor's internal limits. gVisor is the primary enforcer; the host RLIMITs are failsafes only. This prevents confusing double-enforcement where gVisor's OOM killer and the host's SIGKILL race, causing signals that don't propagate cleanly to the typed result relayer. Let gVisor kill first; if gVisor fails to kill, the host RLIMIT catches it.

  - **Layer 3 — Network Isolation**:
    Default: `--network=none`. The sandbox has **no network access** — not even loopback is routable to the host.
    For `real_external` verification only (Mitigation #51): a `socat` or `iptables` rule forwards **only** the allowlisted domain (e.g., `api.github.com:443`) through a transparent proxy that terminates TLS and logs all traffic. Everything else remains dropped. The allowlist is per-skill from `NETWORK_ALLOWLIST` in `config.py`; the skill cannot influence it.

  - **Layer 4 — Capability Stripping & Secret Quarantine**:
    Environment passed to the gVisor sandbox contains **no secrets** whatsoever (`GITHUB_TOKEN`, `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, etc. are stripped). Only dummy credentials are injected for the tool being verified. The skill runs as an unprivileged user inside the gVisor sandbox with no `CAP_*` Linux capabilities.

  1. If no gVisor/container runtime is available, skills may still be mock-verified (`verification_tier = mock`) but **`run-skill` refuses to execute real commands on the host** unless the operator explicitly opts in via `run-skill --unsafe-host` (printed warning).
  2. A skill that passes Layers 0–4 is the only one eligible for `real_local`/`real_external` verification tiers (Mitigation #51).
  3. gVisor is the **V2 Linux target**. V1 may use Docker with `--security-opt=no-new-privileges` and a read-only rootfs as a stepping stone. The gVisor migration is a container runtime swap, not an architectural change.
  4. **WebAssembly (V3 boundary)**: Wasm (Wasmtime/WAMR) provides hardware-level linear memory isolation and native control-flow integrity (CFI), which is structurally superior to gVisor's syscall interception. However, no mature Python→Wasm compilation toolchain exists today; `py2wasm` is experimental, and running CPython inside Wasm still exposes the Python VM as the attack surface. Wasm is deferred to V3 when the skill IR is mature enough to emit non-Python output. See Section 9 (Documented Boundaries).
- **Enforcement Location**: `engine/validator.py` (gVisor executor, sealed envelope), `cli.py` (`run-skill` gating), `config.py` (sandbox runtime selection, `GVISOR_ENABLED`, `NETWORK_ALLOWLIST`).

#### Mitigation #39: Resilient Web Ingestion (Tolerating the Fragile Web)
- **Problem**: Real-world scraping breaks: docs sites change HTML structure, JS-heavy pages resist extraction, Jina/Trafilatura get blocked or rate-limited, search engines throttle, and topics without a `TRUSTED_DOCS` entry depend on fallible search. The existing fallback chain is well-designed but has no caching, no provider health tracking, and no headless-JS escape hatch — so coverage will be uneven and flaky.
- **Fix**:
  1. **Response cache**: Add `data/ingest_cache/` keyed by URL hash with a configurable TTL (default 7 days). Repeat ingestion of the same URL does not re-fetch — reduces rate-limit pressure and flakiness.
  2. **Provider health tracking**: Track per-domain success/failure across sessions. If Jina or Trafilatura fails on a domain repeatedly, auto-downgrade them for that domain and try the next provider first. All providers follow retry-with-backoff (extend Mitigation #9) plus jitter.
  3. **Headless-JS escape hatch**: Add **Stage 4** (optional) — Playwright/Chromium headless render (hard budget ≤30s/page) for JS-only docs, executed after the BeautifulSoup stage and **before** the abort guard (Mitigation #4). If rendering is unavailable, proceed to abort rather than guessing.
  4. **Fetch-quality provenance**: Every ingested fact/passage records `fetch_quality` (`markdown`/`structured`/`raw`/`js_render`) and the provider used. Low-quality provenance (e.g., `raw` HTML text) caps the fact's initial confidence to 0.80 and sets a shorter `staleness_days`, forcing earlier re-verification.
  5. **Coverage gap reporting**: When a unit aborts, the result includes *"No external ground truth found for <topic>. Skipped N facts/skills."* so the user sees what was not learned instead of silently thinner coverage.
- **Enforcement Location**: `engine/ingest.py` (cache, health tracking, JS render), `models.py` (`fetch_quality`), `memory/semantic.py` (confidence cap on low-quality provenance).

---

### Layer M: Proactive Agency & Autonomous Metacognition (Mitigations #40–#44)

> These mitigations transition the system from a passive, command-reactive assistant into an autonomous agent possessing persistent self-awareness, internal goal management, and proactive idle-time self-improvement.

#### Mitigation #40: Persistent Self-Model & Empirical Competence Matrix
- **Problem**: Stateless agents start each session with no concept of what they know well, where their knowledge is brittle, or what projects they are actively assisting with. They treat all learned domains with equal, uncalibrated confidence.
- **Fix**: Maintain a persistent state document `data/self_model.json` updated after every learning session, tool run, and reflection cycle:
  - Tracks identity parameters, boot counter, active focal areas, and known knowledge gaps.
  - Maintains an **empirical competence matrix** per topic calculating real pass/fail ratios of generated skills (`skills_verified / total_skills`), live execution outcomes, and user correction frequency. For knowledge-only domains (no generated skills), competence comes from quiz pass ratios, user ratings, and fact-verification events (Mitigation #55).
  - Injects a compressed summary of the self-model into brain system prompts, giving the agent persistent meta-awareness across session boots.
- **Write protection**: `self_model.json` is never written directly by the agent — only via `memory/self_model.py` from benchmark pass/fail, exit codes, corrections, and ratings (Mitigations #45/#52).
- **Enforcement Location**: `agent/memory/self_model.py`.

#### Mitigation #41: The Heartbeat Daemon (Autonomous Idle Loop)
- **Problem**: Traditional AI assistants sit completely idle until a user enters a prompt. They cannot self-heal stale memory, cannot proactively discover broken skills, and cannot conduct background research on user-assigned long-term goals.
- **Fix**: Implement an asynchronous background daemon (`agent/engine/heartbeat.py`) running on a configurable timer (default: every 15 minutes during user idle cycles) executing a 5-stage loop:
  ```
  1. PERCEIVE: Inspect semantic.db for facts older than staleness_days (M6) and skills.db for unverified skills.
  2. EVALUATE: Query goals.db for highest-priority unblocked subgoals.
  3. PLAN: Formulate an internal learning or verification action without user prompting.
  4. ACT: Execute background ingestion, skill re-testing, or research.
  5. REFLECT: Update data/self_model.json and log audit traces to episodic.db.
  ```
- **Safety Rails**: Hard rate limits (max 3 autonomous actions per hour), strict quiet hours, pause-on-user-input, and zero network calls when offline or `--no-daemon` is specified.
- **No-Op Cadence (7.3 clarification)**: The 15-minute wake is a *cheap check*, not an action. If nothing is actionable (no stale facts, no unverified skills, no ready goals), the cycle performs a metadata-only check and goes back to sleep **without emitting a log line** — no-op cycles are counted silently and never written to `episodic.db`. The wake interval and the 3-actions/hour ceiling are independent: the wake cadence only sets how often the *check* runs, so frequent wakes cause no log noise or meaningful overhead. Operators may raise the interval (e.g., 60 min) via `config.py` when log spam or idle CPU is a concern.
- **Enforcement Location**: `agent/engine/heartbeat.py`, `agent/main.py`.

#### Mitigation #42: Directed Acyclic Goal Graph (Autonomous Task Decomposition)
- **Problem**: Flat "to-do" lists cannot express complex dependencies (e.g. "To build a GitHub release tool, first research Git tags, then verify `gh` CLI auth"). Without prerequisite tracking, autonomous agents attempt tasks out of order.
- **Fix**: Store a DAG in `data/goals.db` with nodes representing long-term objectives and subgoals:
  - Supports dependency resolution: a goal cannot transition to `active` until all its parent prerequisites are `completed`.
  - The Heartbeat Daemon automatically selects the highest-priority root/leaf goal that is ready to execute.
  - Operators can inspect and mutate the graph via `goals add`, `goals list`, `goals complete` CLI commands.
- **Enforcement Location**: `agent/memory/goals.py`, `agent/cli.py`.

#### Mitigation #43: Metacognitive Reflection Engine (Consolidation "Sleep" Cycle)
- **Problem**: Over time, memory accumulates noise, dead superseded facts, stale competence scores, and recurring failure patterns that degrade agent performance.
- **Fix**: Implement a periodic consolidation process (`agent/engine/reflection.py`):
  - Audits `episodic.db` logs to identify frequent user corrections (`source_type = "user_corrected"`).
  - Recalibrates topic confidence scores in `self_model.json` (e.g. downgrading a topic if recent queries yielded low discriminator scores).
  - Flags contradictory or low-confidence facts for re-ingestion on the next heartbeat.
  - Can be triggered manually via `reflect` CLI command or automatically after major learning milestones.
- **Enforcement Location**: `agent/engine/reflection.py`, `agent/cli.py`.

#### Mitigation #44: Headless Runtime Harness (Real Environment Feedback Loops)
- **Problem**: Static AST checks and unit mocks are blind to runtime compiler errors, dynamic linker failures, engine lifecycle bugs, and missing system headers. Without real execution feedback, the Voyager synthesis loop cannot converge on working code.
- **Fix**: Integrate headless execution harnesses for supported toolchains:
  - Runs headless CLI commands, Python subprocesses, or target engine builds inside quarantined sandboxes (Mitigation #38).
  - Captures compiler output, stderr traces, and exit codes as structured environment feedback.
  - Synthesizer revision loop (Mitigation #35) parses real error traces and iteratively refactors code up to 2 retry cycles before committing to `skills.db`.
- **Enforcement Location**: `agent/engine/validator.py`.

---

### Layer N: Autonomous Safety, Circuit Breakers & Mechanical Governors (Mitigations #45–#49)

> When an agent gains the power to plan tasks, reflect on memory, and execute real code, failure modes shift to systemic instability, security breaches, and resource depletion. These mitigations enforce hard mechanical governors.

#### Mitigation #45: External Metric Anchoring (Anti-Self-Deception Competence Governor)
- **Problem**: State and reflection drift: an LLM reflecting on its own episodic logs can hallucinate that broken skills actually worked, reinforcing false self-competence and overconfidence across sessions.
- **Fix**: Competence scores in `self_model.json` are **never updated via self-generated LLM reflection logs**. They are anchored strictly in deterministic external metrics:
  1. Passing runs against the immutable benchmark suite `tests/benchmark_suite/` ($\text{Competence} = \frac{\text{Passed Benchmark Tests}}{\text{Total Benchmark Tests}}$).
  2. Subprocess compiler/runtime exit codes (`code == 0` on real dry-run execution).
  3. Explicit human corrections (`source_type = "user_corrected"`).
  Additionally, `self_model.json` is **write-protected from the agent**: the agent may never edit the file directly; all updates flow through `agent/memory/self_model.py`, which accepts only the three sources above. Direct file modification is treated as a security violation and rolled back (Mitigation #52).
- **Enforcement Location**: `agent/engine/benchmark.py`, `agent/memory/self_model.py`.

#### Mitigation #46: Autonomous Circuit Breakers & Token/Iteration Ceilings
- **Problem**: Runaway loops and cost: an unconstrained background self-improvement loop can burn through hundreds of API calls and CPU cycles while the user is away.
- **Fix**: Enforce hard, non-bypassable mechanical circuit breakers:
  - **Task Step Cap**: Maximum 5 atomic iterations per autonomous goal. If a goal does not complete in 5 steps, transition state to `FAILED` and halt.
  - **Failure Breaker**: 2 consecutive tool or verification failures immediately halts the active task and requests human review.
  - **Heartbeat Throttle**: Heartbeat daemon throttles to configurable intervals (e.g., once every 15 min when active, once per 4 hours during sleep cycles) with a hard ceiling of max 3 actions per hour.
  - **Daily Token & Call Ceiling**: Hard daily budget (configurable in `config.py`, e.g. 50 API calls/day). Once reached, background daemon sleeps until the next UTC day.
- **Enforcement Location**: `agent/engine/governor.py`, `agent/engine/heartbeat.py`.

#### Mitigation #47: Hard-Capped Goal DAG (Anti-Goal-Drift Discriminator)
- **Problem**: Goal drift and context amnesia: left unsupervised, an agent will decompose goals into subgoals, and subgoals into sub-subgoals, wandering completely off-topic and hallucinating irrelevant rabbit holes.
- **Fix**: Enforce structural constraints on `data/goals.db`:
  - **Max Tree Depth — Autonomous = 2**: Tree is strictly restricted to Root Goals $\to$ Subgoals for autonomous heartbeat goals. Subgoals are forbidden from spawning child subgoals.
  - **Max Tree Depth — Supervised = 4**: User-supervised goals may reach depth 4, **only** with explicit `[Y/n]` approval at each expansion past depth 2 (see Mitigation #54). Deeper-than-2 nodes are flagged `supervised = true` and remain visible in the UI.
  - **Parent-Goal Relevance Discriminator**: Before any new subgoal is accepted into `goals.db`, a discriminator evaluates cosine similarity against the parent goal's objective. If similarity $< 0.70$, the subgoal is rejected as goal drift. This discriminator remains active at **every** depth level, including supervised subtrees.
- **Enforcement Location**: `agent/memory/goals.py`.

#### Mitigation #48: Automated Full Regression Suite on Skill Mutation
- **Problem**: Silent behavioral regression: when the agent synthesizes a "fix" for one skill or updates a shared utility, it can silently break 3 previously verified skills without notice.
- **Fix**: Any modification, deletion, or re-synthesis in `skills/` triggers a mandatory, automated background run of the entire regression test harness across all registered skills.
  - If any existing skill fails regression, the mutation is rolled back immediately (`git checkout` or shadow copy restore), and the event is logged as a regression failure in `episodic.db`.
- **Enforcement Location**: `agent/engine/validator.py`, `agent/engine/orchestrator.py`.

#### Mitigation #49: Deterministic Task State Machine, Typed Result Relayer & Ground-Truth Trace Telemetry
- **Problem**: Relying on the LLM's conversation context window to track multi-step execution over hours causes hallucinations and dropped steps due to context limits or session restarts. Additionally, the previous design let the skill self-report its `result_payload` to `active_task.json` via the orchestrator — but if the skill *is* the orchestrator (because it escaped the sandbox), it can write whatever it wants. The FSM's `VERIFYING → COMMITTED` transition must not trust self-reported success.
- **Fix**:
  1. **Finite State Machine (`data/active_task.json`)**: Persistent on-disk state transitions: `PENDING` $\to$ `RUNNING` $\to$ `VERIFYING` $\to$ `COMMITTED` (or `FAILED`). On every turn, the agent reads its current state from disk, executes exactly 1 atomic action, writes the result, and exits prompt context.
  2. **Typed Result Relayer (Host-Side Output Validation)**: The sandbox cannot write to the host database or `active_task.json` directly. Instead, the skill writes its output to **stdout as raw bytes** (the only communication channel — see Mitigation #38, Layer 1 gofer hardening). When the gVisor runtime exits, the **deterministic host-side parser** attempts to decode and validate the output against an immutable schema:
     ```python
     @dataclass(frozen=True)
     class SkillResultSchema:
         """Immutable schema — the skill cannot influence this definition.
         Lives in the host binary (validator.py), NOT inside the sandbox."""

         required_fields: frozenset = frozenset({"exit_code", "stdout", "stderr"})
         optional_fields: frozenset = frozenset({"result", "metrics"})
         max_stdout_bytes: int = 65536  # 64KB cap — truncation attack = reject
         max_stderr_bytes: int = 16384  # 16KB cap


     def parse_skill_output(raw_bytes: bytes, schema: SkillResultSchema) -> dict | None:
         """Parse sandbox output. Returns None on ANY schema violation.
         None → FAILED_VERIFICATION, even if exit_code was 0."""
         try:
             text = raw_bytes.decode("utf-8", errors="strict")
             if len(text) > schema.max_stdout_bytes:
                 return None  # Truncation attack — reject
             result = json.loads(text)
             if not isinstance(result, dict):
                 return None
             if not schema.required_fields.issubset(result.keys()):
                 return None  # Missing required fields
             if not set(result.keys()).issubset(
                 schema.required_fields | schema.optional_fields
             ):
                 return None  # Extra fields = reject (no smuggling)
             return result
         except (UnicodeDecodeError, json.JSONDecodeError):
             return None  # FAILED_VERIFICATION
     ```
     The FSM transitions `VERIFYING → COMMITTED` **only** if `parse_skill_output()` returns a valid dict. If it returns `None`, the FSM transitions to `FAILED_VERIFICATION` regardless of the skill's reported exit code. The skill cannot fake this validation because the schema, serialization logic, and parser all live in the host process (`engine/validator.py`); the skill only outputs raw bytes and has no access to the host's parsing logic.
  3. **End-to-End Trace ID**: Every autonomous action generates a unique UUID `trace_id` propagated across `episodic.db`, linking `Goal -> Prompt -> Tool Invocation -> Sandbox Stderr -> State Mutation` for complete human auditability.
- **Enforcement Location**: `agent/engine/state_machine.py`, `agent/engine/validator.py` (typed result relayer), `agent/memory/episodic.py`.

---

### Layer O: Project Memory, Real-External Verification & Integrity Governors (Mitigations #50–#55)

> These mitigations close the remaining gaps that turn an "encyclopedia" into a software collaborator: persistent project state, honest external-API verification, tamper-proof self-model and project stores, minimum-viable calibration data, human-supervised goal depth, and competence signals for non-skill knowledge domains.

#### Mitigation #50: Persistent Project Memory (Tier 4 — Project-Aware Software Collaboration)
- **Problem**: The system has semantic facts, skills, goals, and a self-model — but no persistent, structured model of the user's actual project: what files exist, what code is in them, what architecture decisions were made, what bugs were found and fixed, what changed between sessions. Without this, the agent cannot reliably help build a game or complex software; it can answer facts about Unity but cannot meaningfully work on a Unity project unless the user pastes code or describes every file.
- **Fix**: Introduce **Tier 4 Project Memory** (`data/projects.db`, `agent/memory/project.py`) with three tables:
  - `projects`: `project_id`, `name`, `root_path`, `description`, `runtime`, `created_at`, `updated_at`.
  - `project_files`: `file_id`, `project_id`, `relative_path`, `absolute_path`, `file_hash` (SHA-256), `language`, `role_summary`, `semantic_summary`, `embedding_json`, `last_indexed_at`. `UNIQUE(project_id, absolute_path)`.
  - `project_decisions`: `decision_id`, `project_id`, `title`, `decision`, `rationale`, `related_files_json`, `timestamp`.
  1. **Automatic indexing**: When the agent reads or writes a workspace file, `memory/project.py` recomputes the hash; re-embeds and re-summarizes only if the hash changed (incremental, cheap).
  2. **Write protection**: The agent is **forbidden from directly editing `projects.db`** (same rule as `tests/benchmark_suite/`). All writes go through `memory/project.py`. Direct modification is treated as a security violation and rolled back.
  3. **Cross-tier retrieval**: The retriever (Subsystem 5) searches `project_files` alongside `semantic_facts`, `context_passages`, and skills. A query like *"Where is player movement handled in my Unity project?"* retrieves the relevant file rows (path/keyword FTS5 + semantic summary embedding) and optionally related Unity-API facts, returning a grounded, code-aware answer.
  4. **Path scanning**: `project index <path>` (CLI) performs an initial full-tree scan (respecting `.gitignore`-style exclusions) and registers the project. Subsequent edits update incrementally.
  5. **Indexing Hardening Guards**:
     - **Symlink Traversal Guard**: Set `follow_symlinks=False` on all directory walks to prevent infinite recursion and directory escapes.
     - **Secret / Credential Sanitizer**: Automatically ignore files matching sensitive patterns (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `credentials.json`, `*secret*`) and run a fast regex scan for standard key patterns before chunking.
     - **File Size Ceiling**: Skip any individual text file > 2 MB to prevent memory exhaustion and context flooding.
     - **Injection Scrubbing**: Strip prompt delimiters (`"""`, `system:`, `<|im_start|>`) before passing file chunks into summarization Brain prompts.
- **Enforcement Location**: `agent/memory/project.py`, `agent/engine/retriever.py`, `agent/cli.py`.

#### Mitigation #51: Real-Local vs Real-External Verification (Network-Aware "Real" Tier)
- **Problem**: Mitigation #35's `real` tier says the skill executes against the real CLI/API "in an isolated sandbox", but the sandbox is described as non-networked. That is a contradiction: local CLIs (`git`, `python`, `docker`, compilers, engines) can be verified offline, but `github_create_release.py` cannot be verified against real GitHub without network access. Without a distinction, external-API skills are either "fake-real" (mock masquerading as real) or impossible to verify.
- **Fix**: Split `real` into `real_local` and `real_external` (Mitigation #35 table), and provide the supporting infrastructure:
  1. **`real_local`**: Non-networked sandbox (Mitigation #38). Verifies local CLIs, compilers, game engines, and any tool with no network dependency. This is the default real tier.
  2. **`real_external`**: For external-API skills, one of two paths:
     - **Network allowlist**: A per-skill `NETWORK_ALLOWLIST` (e.g., `["api.github.com", "hub.docker.com"]`) mounted into the sandbox as the *only* reachable hosts; everything else is blocked. Execution uses a **sandbox/test account** (dummy credentials — Mitigation #38 Layer 3) and never production secrets. Requires explicit operator opt-in in `config.py`.
     - **Deterministic local fixture server**: A local mock HTTP server that emulates the external API using the **official schema captured at ingest time** (Mitigation #12/#35 anchors). The skill is exercised end-to-end against the fixture; requests/responses are schema-validated. This earns `real_external` **only** when a fixture covers the skill's endpoints; otherwise the skill remains at `mock` labeled *"requires network verification"*.
  3. HITL Tier 1 now reads: *"no network by default; allowlisted hosts only for `real_external`"*.
  4. The `skills` listing shows the tier exactly (`mock` / `real_local` / `real_external`) and, for `real_external`, whether it was proven against the live API or an emulated fixture.
- **Enforcement Location**: `agent/engine/validator.py` (network allowlist executor + fixture server), `config.py` (`NETWORK_ALLOWLIST`), `agent/cli.py`.

#### Mitigation #52: Self-Model & Project-Memory Write Protection (Anti-Tamper Integrity)
- **Problem**: `tests/benchmark_suite/` is protected from the agent, but `self_model.json` is not. If the agent has file-write permission for its own workspace, it could edit `self_model.json` directly and inflate its competence scores — turning self-awareness into self-deception. The same risk applies to `projects.db` (it could rewrite project history or decisions).
- **Fix**: Treat `self_model.json` and `projects.db` as **write-protected agent state**, exactly like the benchmark suite:
  1. **Forbidden direct writes**: The agent (including generated skills, synthesizer code, and heartbeat/reflection actions) is strictly forbidden from writing these paths.
     - **V1 Mechanism**: Enforce write protection at the API level (only `project.py` and `self_model.py` expose write handles) combined with a SHA-256 State Manifest (`data/.state_manifest.json`) checked at startup to detect unauthorized file mutations.
     - **V2/V3 Mechanism**: Run the daemon and state stores in a separate OS daemon process communicating exclusively over IPC/UNIX sockets with strict POSIX file permissions (`chmod 600` owned by an isolated daemon user).
  2. **Allowed mutation sources only**: `self_model.py` accepts competence updates from (a) benchmark pass/fail (Mitigation #45), (b) subprocess exit codes, (c) user corrections, and (d) explicit user ratings (Mitigation #55). `project.py` accepts index/decision writes from file-hash diffs and user-provided `project decision` commands.
  3. **Tamper detection & rollback**: A startup watchdog hashes `self_model.json` and `projects.db` (and compares against `data/.state_manifest.json`). If a change was not produced by the owning module's audit log (every write is appended to `episodic.db` with `event_type = "self_model_update"` / `"project_memory_update"` + `trace_id`), the change is treated as a security violation: the file is restored from the last audited state and the event is logged for human review.
  4. Any direct modification by the agent is rolled back and surfaced to the user as a security alert.
- **Enforcement Location**: `agent/memory/self_model.py`, `agent/memory/project.py`, `agent/engine/governor.py`, `agent/main.py` (startup watchdog).

#### Mitigation #53: Calibration Dataset Minimum Viability & Growth Policy
- **Problem**: `calibrate-thresholds` (Mitigation #37) is only as good as `calibration/queries.json`. If that file is empty or has ~10 examples, the calibrated thresholds are noisy and possibly worse than the defaults — and the architecture did not specify who creates the set or how large it must be.
- **Fix**: Define a **minimum-viable calibration set** and a documented growth process:
  1. **Minimum sizes** (per category, per `data/calibration.json` schema):
     | Category | Minimum Items | Must be labeled by |
     |----------|--------------|--------------------|
     | `true_hit` | 50 | human (or curated seed) |
     | `related_unknown` | 30 | human |
     | `opposing` | 20 | human |
     | `unrelated` | 30 | human |
  2. **Refusal to calibrate**: `calibrate-thresholds` **refuses to run** if any category is below its minimum, printing the shortfall: *"Calibration aborted: related_unknown has 12/30 items. Add labeled examples to calibration/queries.json."* Until calibration succeeds, the system uses the uncalibrated defaults (0.65 / 0.80) and labels them as such in `stats`.
  3. **Documented contribution process**: `calibration/queries.json` is a plain, commented JSON with a `meta` section recording per-category counts and provenance (who/when each batch was added). New examples are added by the operator via `calibrate add` (CLI) or direct file edit — the agent never self-adds calibration items (they are ground-truth data, like the benchmark suite).
  4. **Periodic re-calibration**: `calibrate-thresholds` re-runs automatically at a configurable interval (default: every 30 days) and whenever the embedding model changes (ties into Mitigation #2/#37). As the corpus grows, stale thresholds are flagged by the reflection engine (Mitigation #43).
- **Enforcement Location**: `agent/cli.py` (`calibrate add`, `calibrate-thresholds`), `agent/engine/retriever.py` (`data/calibration.json`), `config.py`.

#### Mitigation #54: User-Supervised Deep Goal Trees (Depth 4 with Approval)
- **Problem**: The hard goal-depth cap of 2 (Mitigation #47) prevents goal drift but also blocks real software work. Complex tasks routinely need depth 3–4 (e.g., *Build save system → Design schema → Research Unity serialization → Compare JSON vs binary*). An unconditional depth-2 cap means the agent cannot autonomously pursue the very tasks it is meant to help with.
- **Fix**: Two-tier depth policy on `data/goals.db`:
  | Context | Hard Cap | Approval Gate |
  |---------|----------|---------------|
  | Autonomous background goals (heartbeat) | **2** | No expansion beyond 2 |
  | User-supervised goals | **4** | Explicit `[Y/n]` approval for **each** expansion from depth 2→3 and 3→4 |
  1. Goals (or subtrees) allowed beyond depth 2 are flagged `supervised = true` and remain visible in `goals list` and the UI.
  2. The agent may **never** silently expand depth 2→3 or 3→4. Each proposed expansion must display the proposed subtree and ask for confirmation; a declined expansion keeps the node at its current depth.
  3. The parent-relevance discriminator (Mitigation #47, similarity ≥ 0.70) remains active at **every** depth level, including supervised subtrees.
  4. If a task genuinely needs more than depth 4, the agent must split it into a new root goal (with its own approved subtree) or request manual review — it may not tunnel past depth 4.
- **Enforcement Location**: `agent/memory/goals.py` (depth/supervised enforcement), `agent/cli.py` (approval prompt).

#### Mitigation #55: Competence Signals for Non-Skill Domains (Beyond Skill Benchmarks)
- **Problem**: The empirical competence matrix (Mitigation #40/#45) is anchored in generated-skills pass/fail and benchmark tests. For knowledge-only domains — e.g., WW2 history, design principles — there are no generated skills and no benchmark suite, so the competence matrix stays empty and the agent cannot know whether it is competent in those topics.
- **Fix**: Extend the competence matrix with three non-skill signal sources (stored in `self_model.json`, all through `memory/self_model.py` per Mitigation #52):
  1. **Topic quizzes**: For each learned knowledge topic, the reflection engine (Mitigation #43) generates a short closed-world quiz from stored facts (definition / syntax / troubleshooting questions). The agent answers from memory only; correct ratio feeds `competence[topic].quiz_pass_ratio`.
  2. **User ratings**: `ask` responses carry an optional rating (`rate <topic> good|bad`, or a post-answer prompt). Ratings feed `competence[topic].user_rating` (rolling average). User corrections (Mitigation #31) count as negative signals.
  3. **Fact verification events**: `refresh <topic>`, staleness re-verification (Mitigation #6), and contradiction-gating supersedes (Mitigation #31) produce events: a topic whose facts were re-verified without change earns `competence[topic].fact_stability`; a topic with frequent supersedes/corrections is downgraded.
  The competence matrix therefore reports, per topic: `skills_verified` (where applicable), `quiz_pass_ratio`, `user_rating`, and `fact_stability`. For non-skill topics, the last three provide the signals that skill benchmarks cannot.
- **Enforcement Location**: `agent/engine/reflection.py` (quiz generation), `agent/memory/self_model.py`, `agent/cli.py` (`rate` command), `agent/engine/retriever.py`.

---

### Layer P: User-Defined Brain Providers & Extensibility (Mitigation #56)

#### Mitigation #56: Brain Provider Registry (`brains.json` — User-Defined APIs)
- **Problem**: The brain backend was effectively hardcoded to `mock|gemini|claude` via `--brain` and the `AI_BRAIN` env var. Users cannot plug in their own provider — e.g., OpenCodeZen DeepSeek v4, Google Gemini, Codex, a self-hosted vLLM, or an Ollama local model — without editing source code. Every user should be able to register whatever API they pay for.
- **Fix**: A hand-editable **`brains.json`** provider registry at the project root (next to `.env.example`), loaded at startup by `factory.py`:
  ```json
  {
    "active": "opencodezen",
    "providers": [
      {
        "name": "opencodezen",
        "kind": "openai_compatible",
        "base_url": "https://opencodezen.example.com/v1",
        "model": "deepseek-v4",
        "api_key_env": "OPENCODEZEN_API_KEY",
        "context_window": 128000,
        "max_tokens": 8192,
        "description": "OpenCodeZen DeepSeek v4"
      },
      {
        "name": "gemini",
        "kind": "google_genai",
        "model": "gemini-2.5-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "context_window": 1048576
      },
      {
        "name": "codex",
        "kind": "openai_compatible",
        "base_url": "https://api.opencode.example.com/v1",
        "model": "codex-opencode",
        "api_key_env": "OPENCODE_API_KEY"
      },
      {
        "name": "claude",
        "kind": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY"
      },
      {
        "name": "mock",
        "kind": "mock"
      }
    ]
  }
  ```
  1. **`kind` drives the adapter** (Subsystem 6): `openai_compatible` (any `/v1/chat/completions` endpoint — OpenCodeZen, DeepSeek, Codex, vLLM, Ollama, LM Studio), `google_genai` (GenAI SDK), `anthropic` (LiteLLM/direct), `mock`.
  2. **Keys stay out of the file**: each provider references its key via `api_key_env`, so secrets live in `.env` / environment variables, never in `brains.json`. A missing key triggers an automatic MockBrain fallback with a clear warning.
  3. **Selection precedence**: `--brain <provider>` flag > `AI_BRAIN` env var > `active` field in `brains.json` > `mock`. Unknown provider names are rejected with a list of registered names.
  4. **Health/versioning**: `factory.py` caches provider connectivity; a provider that fails 3 consecutive calls is downgraded for that session with the retry/backoff of Mitigation #9. All providers share the same JSON-extraction resilience (Mitigation #8) and token-cost reporting (Mitigation #10).
  5. **No source edits required**: adding a provider is a pure config change — `brains.json` + the referenced `api_key_env`. `brains.example.json` ships with the repo documenting every field.
- **Enforcement Location**: `brains.json` (user config), `agent/brains/factory.py`, `agent/brains/*`, `agent/cli.py` (`--brain`), `config.py`.

---

### Layer Q: Zero-Trust Execution Boundary & Sandbox Dependency Strategy (Mitigations #57–#60)

> These mitigations formalize the zero-trust compilation and sealed execution boundary redesign. They supersede the original AST-scanner-only model with a defense-in-depth architecture where the compiler, gVisor runtime, typed result relayer, and tiered sandbox images form independent, non-trusting layers. A single missed assumption in any one layer does not compromise the system.

#### Mitigation #57: Compiler → Sandbox Image Selection Pipeline (Import-Driven Image Resolution)
- **Problem**: The positive-match compiler (Mitigation #25) validates that all imports are in the Tiered Import Allowlist, and the gVisor sealed envelope (Mitigation #38) runs the skill inside a read-only squashfs rootfs. But these two systems are disconnected: the compiler may allow `import numpy` (a valid, useful module), but the gVisor rootfs may not contain `numpy`'s native `.so` libraries compiled for the gVisor kernel. The skill imports `numpy`, gVisor throws `ModuleNotFoundError`, and the typed relayer marks it `FAILED_EXECUTION` — wasting cycles, frustrating the agent's planning loop, and making the synthesizer's revision loop chase a phantom bug (the code is correct; the environment is wrong).
- **Fix**: Connect the compiler's import analysis directly to sandbox image selection. The compiler extracts all imports, maps each to a **sandbox image tier**, and selects the smallest image that satisfies all dependencies. If any import isn't available in ANY image tier, compilation fails immediately with a clear error — before any sandbox is spawned.

  **Sandbox Image Tiers** (pre-built at system install time as read-only squashfs/overlay layers):

  | Image | Contents | Approx. Size | Boot Time | Selected When |
  |-------|----------|-------------|-----------|---------------|
  | `sandbox-stdlib` (default) | Python 3.12 + full stdlib | ~80 MB | <500ms | All imports are Tier 1 stdlib modules |
  | `sandbox-scientific` | stdlib + `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` | ~650 MB | ~1.5s | Any import from `{numpy, scipy, pandas, sklearn, matplotlib}` |
  | `sandbox-web` | stdlib + `requests`, `httpx`, `beautifulsoup4`, `lxml` | ~150 MB | ~700ms | Any import from `{requests, httpx, bs4, lxml}` |
  | `sandbox-full` | stdlib + scientific + web + additional packages | ~1.2 GB | ~2.5s | Imports span multiple tiers, or operator override |

  **Resolution algorithm**:
  ```python
  def resolve_sandbox_image(compiled_imports: set[str]) -> str:
      """Select the smallest sandbox image that satisfies all imports.
      Called AFTER the positive-match compiler passes (Mitigation #25).
      Runs on the HOST, not inside the sandbox."""
      TIER_MAP = {
          "sandbox-stdlib": TIER_1_MODULES,  # json, re, math, pathlib, ...
          "sandbox-web": TIER_1_MODULES | {"requests", "httpx", "bs4", "lxml"},
          "sandbox-scientific": TIER_1_MODULES
          | {"numpy", "scipy", "pandas", "sklearn", "matplotlib"},
          "sandbox-full": TIER_1_MODULES
          | {
              "requests",
              "httpx",
              "bs4",
              "lxml",
              "numpy",
              "scipy",
              "pandas",
              "sklearn",
              "matplotlib",
          },
      }
      # Select smallest satisfying image (ordered by size)
      for image_name in [
          "sandbox-stdlib",
          "sandbox-web",
          "sandbox-scientific",
          "sandbox-full",
      ]:
          if compiled_imports.issubset(TIER_MAP[image_name]):
              return image_name
      # No image contains all imports → compilation failure
      missing = compiled_imports - TIER_MAP["sandbox-full"]
      raise CompilationError(
          f"Imports {missing} are not available in any sandbox image. "
          f"The skill cannot be executed. Available modules: {sorted(TIER_MAP['sandbox-full'])}"
      )
  ```

  **Key properties**:
  1. Image selection is **deterministic and host-side** — the skill cannot influence which image is chosen.
  2. Images are **pre-built and immutable** — no `pip install` inside the sandbox, ever. If a module isn't in a pre-built image, it doesn't exist in the sandbox's universe.
  3. The `sandbox-stdlib` image is the default and covers >90% of synthesized skills (CLI wrappers, JSON parsing, data transformation). The heavier images boot only when the compiler detects their specific imports.
  4. **Operator extension**: Custom images can be added to `config.py`'s `SANDBOX_IMAGES` registry by building a new Dockerfile and registering its module set. The compiler's resolution algorithm picks them up automatically.
  5. The synthesizer prompt (Subsystem 3) is informed of available modules: *"You may import from: {sorted(TIER_MAP['sandbox-full'])}. Any other import will be rejected at compile time."* This prevents the LLM from proposing skills with unavailable dependencies.
- **Enforcement Location**: `engine/validator.py` (image resolution), `config.py` (`SANDBOX_IMAGES` registry), `engine/synthesizer.py` (available-modules prompt injection).

#### Mitigation #58: Sandbox I/O Protocol (stdin/stdout Only, No Bind-Mounts)
- **Problem**: If the sealed envelope mounts a host directory (even `noexec`) into the gVisor namespace, a malicious skill can attempt `os.open("../../../etc/passwd")` — and because the gofer resolves paths on the host, it might escape the intended root if not strictly chrooted with `--rootless` and `--host-uds=off`. Any host bind-mount is a potential escape vector.
- **Fix**: Enforce a strict **stdin/stdout-only I/O protocol** between the host and the sandbox. No host filesystem is ever visible to the skill.
  1. **Input**: The host serializes all input data as a JSON payload and pipes it to the sandbox process's **stdin** (within the `RLIMIT_AS` memory cap from Mitigation #38). The skill reads from `sys.stdin`. No input files are mounted.
  2. **Output**: The skill writes its structured result to **stdout** as a JSON payload conforming to the `SkillResultSchema` (Mitigation #49). No output files are bind-mounted. The host captures stdout after the gVisor process exits.
  3. **Post-exit retrieval (fallback)**: If the skill must write intermediate files (e.g., a generated plot or binary artifact), it writes them to the gVisor's **internal** `/tmp` (the `noexec,nosuid,nodev` tmpfs from Mitigation #38, Layer 2). The host retrieves these files **after container exit** via `runsc exec cat /tmp/<filename>` — never via a bind-mount. The retrieved bytes are validated by the typed result relayer before being accepted.
  4. **Error channel**: stderr is captured separately by the host for diagnostic logging (fed to the synthesizer's revision loop per Mitigation #35) but is **never parsed as structured output** — only stdout is trusted for result payload.
  5. **No bidirectional communication during execution**: Once the sandbox starts, there is no host→sandbox or sandbox→host communication channel until the process exits. No RPC, no shared memory, no Unix domain sockets (blocked by `--host-uds=false`), no network (blocked by `--network=none`).
- **Enforcement Location**: `engine/validator.py` (sandbox I/O orchestration), `engine/state_machine.py` (result capture).

#### Mitigation #59: gVisor Rootfs Immutability & Container Build Pipeline
- **Problem**: If the gVisor rootfs is built ad-hoc or modified between runs, a previous skill execution could have poisoned the filesystem (e.g., by writing a malicious `.pth` file to `site-packages` that auto-executes on Python startup). The rootfs must be provably identical across runs.
- **Fix**:
  1. **Read-only squashfs**: Each sandbox image tier (Mitigation #57) is built as a **read-only squashfs layer** at system install time via `Dockerfile.sandbox-<tier>`. The squashfs is mounted read-only by gVisor; the skill cannot modify it.
  2. **Content-addressable verification**: Each built image is identified by a SHA-256 hash of its squashfs content, stored in `data/sandbox_manifest.json`. On startup, the governor (Mitigation #52) verifies image hashes. If a hash doesn't match (indicating tampering), the image is rejected and skill execution is blocked until the image is rebuilt.
  3. **No `pip install` inside sandbox**: The sandbox has no network (Mitigation #38, Layer 3) and a read-only rootfs. `pip install` fails at both the network and filesystem level. All dependencies must be in the pre-built image or they don't exist.
  4. **Ephemeral per-execution**: Each skill execution spawns a **fresh gVisor instance** from the immutable rootfs. No state persists between executions. The internal `/tmp` tmpfs is destroyed when the container exits.
- **Enforcement Location**: `engine/validator.py` (container lifecycle), `engine/governor.py` (image hash verification), `Dockerfile.sandbox-*` (build pipeline).

#### Mitigation #60: Synthesizer Dependency Awareness (Closing the Compiler-Sandbox Loop)
- **Problem**: Without informing the LLM of available modules, the synthesizer proposes skills with arbitrary `pip` dependencies. The compiler allows the import (it's not in Tier 3), but the sandbox doesn't have the module. The revision loop wastes 2 retry cycles chasing `ModuleNotFoundError` before giving up.
- **Fix**: Close the loop by injecting dependency constraints into the synthesis prompt:
  1. The synthesizer's skill-generation prompt includes: *"Available Python modules in the sandbox: [list from SANDBOX_IMAGES registry]. You MUST NOT import any module not in this list. If you need a module that is not available, state the dependency explicitly and the skill will be flagged as 'requires custom sandbox image'."*
  2. If the compiler (Mitigation #25) encounters an import for a module in `TIER_MAP['sandbox-full']` but not in the selected image, it upgrades the image selection (e.g., `sandbox-stdlib` → `sandbox-scientific`). If the module isn't in ANY image, compilation fails with an actionable error: *"Module 'tensorflow' is not available in any sandbox image. Add it to a custom image or choose an alternative."*
  3. Skills flagged as `requires_custom_image` are registered in `skills.db` with `is_verified = 0` and a clear label. They are never auto-executed.
- **Enforcement Location**: `engine/synthesizer.py` (prompt injection), `engine/validator.py` (image upgrade logic), `memory/procedural.py` (custom-image flag).

---

### Layer R: Reasoning Engine & Metacognitive Improvement (Mitigations #61–#70)

> These mitigations extend the architecture from **knowledge acquisition** ("I don't know X → research X → remember X") to **reasoning improvement** ("I attempted X, failed in a specific way, discovered why, and changed how I approach similar problems"). They implement a three-level improvement ladder: Level 1 (system improvement via memory/tools/critics), Level 2 (strategy improvement via retrievable reasoning patterns), and Level 3 (model improvement via verified self-generated training data, gated on model-weight control).

#### Mitigation #61: Tier 2.5 Reasoning Memory (`data/reasoning.db`)
- **Problem**: The episodic DB records *what happened* (tool calls, outputs, state mutations). The semantic DB records *what is true* (distilled facts). Neither records *how the agent reasoned* — the trajectory from initial hypothesis through failure to generalized correction. Without this, every failure is forgotten after 90 days, the Heartbeat has no material to build a personal curriculum from, and the domain delta computation (M#62) has no structured episodes to aggregate.
- **Fix**: Introduce Tier 2.5 Reasoning Memory — a new `reasoning.db` SQLite store with a `reasoning_episodes` table (schema defined in Section 3). Key properties:
  1. **Failure-first value**: successful episodes are stored, but high-novelty failures (unexpected outcome given high model confidence) are prioritized for curriculum replay. A successful task gives you "X worked." A failed task gives you the full SHyAOEDRGL tuple: **S**tate → **Hy**pothesis → **A**ction → **O**bservation → **E**rror → **D**iagnosis → **R**evised hypothesis → **G**eneralized **L**esson.
  2. **Linked to episodic**: every `reasoning_episode` carries a `trace_id` FK into `episodic.db`, so the full tool-invocation chain is recoverable.
  3. **Permanent retention**: no 90-day TTL. Reasoning episodes are the system's most valuable long-term asset.
  4. **Verification gate**: `verified = true` is set only by the Symbolic Verifier (M#65) or human approval. Unverified episodes are stored but excluded from the training-data pipeline (M#69).
- **Enforcement Location**: `memory/reasoning.py` (sole writer), `engine/reflection.py` (episode construction), `engine/heartbeat.py` (curriculum replay).

#### Mitigation #62: Cross-Cutting Reasoning Profile & Domain Delta System
- **Problem**: Tracking reasoning proficiency *inside* individual skill records would cause domain overfitting — the agent would over-specialize its reasoning style to the domain where it first encountered each skill (e.g., always using "decomposition" for git problems because that strategy succeeded on the first git skill). The reasoning profile must be **cross-cutting** (not owned by any skill) but **domain-queryable** (domain context must be retrievable at inference time).
- **Fix**: Add a `reasoning_profile` root key to `self_model.json` (schema in Section 3) with three sub-structures:
  1. **`global_scores`** (prior): baseline proficiency per reasoning category, computed across all domains. Initial values default to 0.5; updated by the reasoning benchmark suite (M#66).
  2. **`domain_deltas`** (posterior correction): per-domain adjustments on the global prior, computed weekly by a SQL aggregation job:
     ```sql
     SELECT reasoning_domain, strategy_label,
            AVG(CASE WHEN outcome_class = 'success' THEN 1.0 ELSE 0.0 END) as domain_success_rate
     FROM episodic_log
     WHERE strategy_label IS NOT NULL
     GROUP BY reasoning_domain, strategy_label
     ```
     `delta = domain_success_rate - global_score[strategy_label]`. Written to `domain_deltas[domain][strategy]`. This transforms the profile from a static report card into a **Bayesian prior/posterior system**: the agent knows it is "good at decomposition globally" but also knows that for web APIs its decomposition underperforms. When a novel git problem arrives, it loads `global_scores` (no domain overfit) rather than a domain-specific posterior.
  3. **`strategy_index`** (meta-policy): maps problem archetypes to preferred reasoning strategies. When the Planner classifies a new task, it reads `strategy_index[archetype]` and injects the corresponding system prompt template, writing `strategy_label` and `prompt_hash` to the task record (M#67) before execution begins.
- **Critical distinction**: `novelty_score` (task-relative to history — is this a new problem type?) is kept **orthogonal** from `complexity_score` (task-intrinsic — how deep is the decomposition tree?). Route MoA inference (M#70) on `complexity_score`; compute domain deltas on `novelty_score`. These are different signals and must not be conflated.
- **Enforcement Location**: `memory/self_model.py` (profile update, delta write), `engine/planner.py` (novelty_score computation, strategy injection), `engine/reflection.py` (weekly delta aggregation job).

#### Mitigation #63: Lateral Critic / Adversarial Verifier (Two-Solver Arbiter Pattern)
- **Problem**: A single LLM Solver→Critic loop fails if the Critic shares the same base model as the Solver. They share identical blind spots, latent biases, and hallucination patterns — the Critic will ratify ~95% of the Solver's errors because it rationalizes them identically. Using a larger/stronger model as Critic is expensive; using the same model is security theater.
- **Fix**: Replace the single Solver→Critic sequence with a **lateral arbiter pattern**:
  1. Run **two Solver instances** (different brain configs from `brains.json`, e.g., Solver = DeepSeek, Solver-B = Claude) on the same problem in parallel.
  2. If their outputs **agree** → skip the critic entirely (saves 66% inference cost on the Critic call). Record as `outcome_class = 'success'` with `hypothesis_count = 1`.
  3. If their outputs **diverge** → spawn a **third Arbiter instance** (preferably a different model family or a deterministic symbolic check) to evaluate the disagreement. The disagreement itself is far more informative than agreement. Record the divergence as a `reasoning_episode` with both hypotheses, the arbiter's resolution, and `hypothesis_count = 2`.
  4. The Critic (when invoked) is **forced to search** for: unsupported assumptions, contradictions, missing edge cases, alternative explanations, counterexamples, unnecessary complexity, incorrect causal reasoning. It may not return "Looks good" without flagging at least one potential weakness.
- **What the critic must not do**: serve as a superior judge. It is a lateral adversary. Its role is to increase the information content of disagreement, not to certify correctness.
- **Enforcement Location**: `engine/critic.py` (new module), `engine/orchestrator.py` (parallel solver dispatch, divergence detection), `brains/factory.py` (multi-brain selection).

#### Mitigation #64: Hypothesis Competition Engine & Counterfactual Training
- **Problem**: The agent commits to its first explanation before generating alternatives. For debugging ambiguous failures, early commitment biases all subsequent observations toward confirming the initial hypothesis. Counterfactual reasoning ("what change would make this solution fail?") is never applied, so solutions are stored as unconditionally correct rather than as conditionally correct under specific assumptions.
- **Fix**: Two interlocking mechanisms:
  1. **Hypothesis Competition**: When `novelty_score > 0.8` (novel problem) or when the Planner classifies the task as `debugging`, the Strategy Injector activates the `hypothesis_competition` prompt template, which forces the agent to:
     - Generate **3–5 competing hypotheses** before taking any action.
     - For each hypothesis, define **one discriminating test** (the cheapest observation that would rule it out).
     - Execute tests in order of `information_gain / execution_cost`.
     - Record `hypothesis_count` in the episodic row.
     The agent learns that certain diagnostic tests are more informative than others over time — this becomes procedural knowledge retrievable via `strategy_index`.
  2. **Counterfactual Training**: After any successful solution, the Heartbeat (during its reflection cycle) poses: *"What change would make this solution fail?"* Generates variants on: input scale, edge-case activation, state ordering, concurrent access, dependency removal. Runs variants in the gVisor sandbox (M#38). Records which assumptions the solution depends on. Stores as `reasoning_episode` with `generalized_rule = "Solution X works under conditions A/B/C, breaks under D."`
- **Enforcement Location**: `engine/heartbeat.py` (counterfactual reflection loop), `engine/planner.py` (hypothesis count injection), `memory/reasoning.py` (counterfactual episode storage).

#### Mitigation #65: Structured Reasoning Trace (SRT) + Symbolic Verifier
- **Problem**: Reasoning benchmarks (causal chains, counterfactuals, contradictory requirements) cannot be objectively verified by the sandbox. The sandbox verifies *what happened* (exit codes, output bytes), not *how the agent got there*. An LLM can output the correct final answer via flawed logic, pass the benchmark, and the training pipeline will label that trace as "successful reasoning" — poisoning the training set with post-hoc rationalization.
- **Fix**: For reasoning benchmarks, enforce a **Structured Reasoning Trace (SRT)** output format that the model must populate alongside its conclusion:
  ```json
  {
    "conclusion": "C",
    "premises": ["A", "A→B", "B→C"],
    "inference_rule": "transitive_implication",
    "rejected_hypotheses": [
      {"hypothesis": "A does not cause C", "ruled_out_by": "observation: B always follows A"}
    ],
    "confidence": 0.91
  }
  ```
  The **Symbolic Verifier** (a deterministic host-side process using Z3 or a minimal Prolog engine) then:
  1. Parses the SRT.
  2. Checks whether the listed premises and inference rules **logically entail** the stated conclusion.
  3. Returns `verified = true` only if the logical chain is sound.
  4. A model that guesses the correct answer but provides a logically invalid SRT receives `verified = false` — its trace is not admitted to the training-data pipeline.
  This means the sandbox is checking **how the agent got there**, not just **what it concluded**.
- **Enforcement Location**: `engine/verifier.py` (new symbolic verifier module), `engine/validator.py` (SRT schema enforcement), `memory/reasoning.py` (verified flag write).

#### Mitigation #66: Reasoning Benchmark Suite with ZPD Binary Search (Difficulty Calibration)
- **Problem**: A static reasoning benchmark demoralizes the agent if too hard (all failures, no learning signal) and wastes cycles if too easy (all successes, no improvement signal). Additionally, existing benchmark suites measure *knowledge* ("did you know the answer?") not *reasoning* ("did you apply the correct inferential strategy?"). A single `Reasoning Score = 82%` is less useful than knowing the **difficulty ceiling** the agent can reliably clear in each reasoning category.
- **Fix**:
  1. **Reasoning-specific benchmark categories** (separate from knowledge/skill/project benchmarks):
     - `decomposition`: break complex problems into subtasks
     - `hypothesis_testing`: design discriminating tests between competing explanations
     - `causal_reasoning`: distinguish causation from correlation; check transitivity
     - `counterexample_gen`: find inputs that break a proposed solution
     - `planning`: multi-step sequencing with resource constraints
     - `adversarial`: detect misleading evidence, conflicting requirements, red herrings
  2. **Parameterized difficulty**: each benchmark exposes deterministic difficulty knobs, for example:
     | Knob | Range | Example |
     |------|-------|---------|
     | `input_size` | 10 → 100,000 items | array sort correctness |
     | `edge_case_depth` | 0 (happy path) → 3 (adversarial) | exception handling |
     | `ambiguity_level` | 0 (clear spec) → 2 (contradictory requirements) | requirement conflict |
     | `chain_length` | 1 → 8 inference steps | transitive causal chain |
  3. **ZPD Binary Search**: The Heartbeat runs a binary search per reasoning category to locate the **Zone of Proximal Development** — the difficulty ceiling where the agent flips from PASS to FAIL:
     - Start at the agent's historical 50% success rate difficulty.
     - PASS → increase difficulty by 20%.
     - FAIL → decrease difficulty by 20%.
     - After 5 rounds, converge on the `zpd_ceiling` for that category.
     - Write `zpd_ceilings[category]` to `self_model.json`.
     The agent always practices at ~20–30% harder than its last reliable ceiling — never too hard to learn, never too easy to improve. "Decomposition score = 82%" now means: **"Can handle 82nd percentile of input complexity in decomposition tasks."** This is a granular, monotonic metric, not a static report card.
  4. **Reasoning benchmark vs. knowledge benchmark separation**: Reasoning benchmarks live in `tests/reasoning_suite/` (separate from `tests/benchmark_suite/`). The `competence_score` in `self_model.json` tracks knowledge/skill benchmarks; `zpd_ceilings` tracks reasoning benchmarks. They are deliberately separate: one measures *what the agent knows*; the other measures *how it thinks*.
- **Enforcement Location**: `engine/benchmark.py` (ZPD binary search loop), `tests/reasoning_suite/` (new directory, immutable like benchmark_suite), `memory/self_model.py` (zpd_ceilings update).

#### Mitigation #67: Episodic Telemetry Hotfix (Pre-Migration Dependency for M#62)
- **Problem**: The domain delta computation (M#62) requires knowing which reasoning strategy was active during each episodic trace. Without explicit ground-truth fields, deltas must be inferred from behavioral proxies ("used 4 tools before answering" ≈ "hypothesis_competition was active"), which is noisy and introduces systematic bias. The schema hotfix must precede any domain delta aggregation — retrofitting a running episodic DB mid-V2 is painful; adding fields before the first `INSERT` is free.
- **Fix**: Add six new explicitly-typed columns to `episodic.db` (schema updated in Section 3, Tier 1). **Build order**:
  1. **Before writing a single line of `episodic.py`**: lock the six-column schema in `models.py`'s `EpisodicLog` Pydantic model and `active_task.json`'s `TaskState` model simultaneously.
  2. **Planner extension (Week 1)**: the Planner already runs a vector search for RAG context (M#26). Extend it to compute `novelty_score = 1 - max_cosine_similarity_to_historical_task_embeddings`. Write to the task record at spawn time. Cost: one extra vector search — effectively zero.
  3. **Strategy Injector (Week 2)**: reads `novelty_score` from the task record and selects the system prompt template from `strategy_index`. Writes `strategy_label` and `prompt_hash` to the episodic row at injection time. Now `strategy_label` is **explicit ground truth**, not a behavioral inference.
  4. **Domain Delta Aggregation Job (Week 3–4)**: the weekly aggregation job becomes a clean 4-line SQL GROUP BY against `reasoning_domain`, `strategy_label`, `outcome_class`. Compare against global_scores → write deltas to `self_model.json`.
  **Without this hotfix, M#62's domain_deltas are statistically contaminated and must not be shipped.**
- **Enforcement Location**: `models.py` (EpisodicLog, TaskState schema), `engine/planner.py` (novelty_score), `engine/strategy_injector.py` (new module — strategy_label, prompt_hash write), `engine/reflection.py` (weekly delta aggregation).

#### Mitigation #68: Novelty & Entropy Filter on Experience Database
- **Problem**: The training-data pipeline (M#69) will accumulate thousands of reasoning episodes. Fine-tuning on this corpus without filtering causes **distribution collapse** — the gradient descent weights thousands of easy, common successes far more heavily than rare, insightful failures. The model regresses toward its most common solution patterns, becoming less creative and more conservative. 10,000 boring successes ruin a model; 500 high-novelty failures improve it.
- **Fix**: Before any episode reaches the Dataset Builder (M#69), it must pass a **Novelty & Entropy Filter**:
  1. **Novelty gate**: `episode.novelty_score > 0.7` OR the episode represents an unexpected failure (model confidence was high but `outcome_class = 'failure'`). Low-novelty successes are stored in `reasoning.db` but excluded from the training set.
  2. **Solution-path entropy gate**: the action sequence of the episode must be > 2 standard deviations away from the centroid of historical action sequences (measured by cosine similarity of the action embedding). Ensures the training set contains genuinely different solution paths, not variations on the same template.
  3. **Verified gate**: `episode.verified = true` (symbolic verifier cleared the SRT, M#65) OR human approval. Unverified episodes are never training data regardless of novelty.
  4. **Minimum corpus size**: the Dataset Builder refuses to run until the filtered corpus contains ≥ 500 verified, high-novelty episodes (analogous to the calibration minimum viability gate in M#53).
- **Enforcement Location**: `engine/dataset_builder.py` (novelty filter), `memory/reasoning.py` (novelty_score on episode write).

#### Mitigation #69: Experience → Training-Data Pipeline (V4, Model-Weight-Control Gated)
- **Problem**: Mitigations #61–#68 improve the system (Level 1) and improve strategies (Level 2), but the underlying model weights are unchanged. To achieve Level 3 (actual model improvement), verified reasoning episodes must become a fine-tuning dataset, and the fine-tuned model must be benchmarked before deployment.
- **Fix**: A three-stage pipeline, activated only when a self-hosted model is available (Llama / Mistral / Qwen via vLLM):
  1. **Dataset Builder**: pulls all episodes passing M#68's filter; structures them as preference pairs (verified correct trace vs. rejected incorrect trace) for DPO, or as supervised trajectories for SFT on pure reasoning chains.
  2. **Fine-tuning (DPO preferred over SFT)**: DPO compares pairs (good trace vs. bad trace) rather than absorbing absolute truth from a single trace — substantially more resilient to noisy data. LoRA adapters on a frozen base model allow V3 deployment without replacing the existing API dependency (see M#70).
  3. **Benchmark gate**: the fine-tuned model must pass the **full benchmark suite** (M#45) AND achieve a higher `zpd_ceiling` average (M#66) than the previous model on ≥ 3 of 6 reasoning categories before being promoted. If it fails, the old model weights are restored. This is the evolutionary loop: Model → solve → observe → collect failures → generate corrections → train → benchmark → deploy only if better.
  **Do not train on everything the system generates.** The filter chain is: Experience → Symbolic Verifier (M#65) → Novelty Filter (M#68) → human approval OR multiple independent validators → Dataset Builder → DPO → benchmark → deploy.
- **Enforcement Location**: `engine/dataset_builder.py`, `engine/trainer.py` (new, V4), `engine/benchmark.py` (promotion gate).

#### Mitigation #70: LoRA Co-Processor / Mixture-of-Agents Router (V3 Stepping Stone)
- **Problem**: Full model replacement (V4) requires controlling model weights, which is gated on self-hosting infrastructure. V3 needs a way to achieve partial Level 3 improvement — applying fine-tuning to specific reasoning tasks — without breaking the existing cloud API dependency.
- **Fix**: A **Mixture-of-Agents (MoA) router** that treats a fine-tuned LoRA adapter as a specialized reasoning co-processor alongside the generalist cloud model:
  - **Routing signal**: `complexity_score` (task-intrinsic decomposition depth, estimated by the Planner — **not** `novelty_score`, which is task-relative to history). These are orthogonal signals and must not be conflated.
  - **Routing logic**: `complexity_score < 0.5` → route to frozen cloud API (cheap, fast, routine tasks). `complexity_score ≥ 0.5` → route to fine-tuned LoRA adapter on a rented/local GPU (vLLM with Unsloth/Axolotl).
  - **Promotion criterion**: if the LoRA adapter outperforms the cloud API on reasoning benchmarks (M#66) in ≥ 80% of head-to-head comparisons over 30 days, swap the router priorities permanently.
  - **Safety**: the LoRA adapter is a *specialist*, not a replacement. It handles deep reasoning tasks; the cloud API handles everything else. The router is configurable in `brains.json` as a new provider kind (`"moa_router"`).
- **Enforcement Location**: `brains/moa_router.py` (new module), `brains/factory.py` (router provider kind), `engine/planner.py` (complexity_score computation), `config.py` (`MoA_ENABLED`, `LORA_ADAPTER_ENDPOINT`).

---

## 5. Subsystem Implementation Details

### Subsystem 1: Research Planner (`agent/engine/planner.py`)
- Input: High-level topic string (e.g., `"GitHub"`, `"Docker"`, `"Kubernetes"`).
- Decomposes topic into `CurriculumPlan`:
  - `conceptual_units`: Definitions, architecture, and core distinctions.
  - `practical_units`: CLI commands, syntax, parameters, flags, and configuration.
  - `skill_targets`: Specific tool names to synthesize for the active runtime (topic-prefixed; Python in V1, see Mitigation #36).
- **Scope Boundary**: Sets realistic boundaries for broad topics. Outputs: *"Learned GitHub CLI releases & PRs (4 units). Additional workflows can be studied on demand."*
- **Disambiguation**: Defaults to technical interpretation. Real brains state their interpretation explicitly.

### Subsystem 2: Autonomous Ingestor (`agent/engine/ingest.py`)
- **Source Selection (Mitigation #27)**:
  1. Check `TRUSTED_DOCS` registry for the topic → fetch official URLs directly (skip search).
  2. If no trusted docs entry, run DuckDuckGo search with **Domain Authority Scoring** (max 2 queries/unit, 1s delay).
  3. Score results by domain reputation, freshness, and authority. Take top 2.
- **Resilient Fetching Chain + Abort Guard** (per URL):
  1. Jina Reader (`https://r.jina.ai/<url>`) — JS-rendered markdown (8s timeout).
  2. Direct HTTP + Trafilatura — article extraction (8s timeout).
  3. Direct HTTP + BeautifulSoup — raw text (8s timeout).
  4. Optional Playwright/Chromium headless render (Mitigation #39, ≤30s/page, only if available).
  5. **Abort Guard**: If all providers fail OR extracted content is <100 chars (Mitigation #4), the curriculum unit is aborted with a coverage-gap report. **No synthetic fallback text is ever ingested.** (*Reconciled — the V1 build contains no "synthesize from curriculum description" path, and the abort is a guard, not a numbered fetch stage.*)
- **Semantic Chunker**: Breaks documents into ≤4000-char paragraph-aware chunks with overlap.
- **Extensibility**: `data/trusted_docs.json` allows users to add custom official doc URLs.

### Subsystem 3: Fact Distillation & Hierarchical Synthesizer (`agent/engine/synthesizer.py`)
- **Structured Knowledge Extraction (Mitigation #32)**: Distillation prompt extracts into exactly **3 core types** (`concept`, `syntax`, `troubleshooting`). `depth_level` is **not used in V1**.
- **Context Passage Extraction (Mitigation #33)**: Identifies coherent 200–500 word explanatory blocks and stores them in `context_passages` table. Max 10 per curriculum unit.
- **Fact Deduplication**: Rejects new facts with >0.95 similarity to existing active facts on the same topic.
- **Skill Context via RAG (Mitigation #26)**:
  - Queries `skills.db` vector store with current unit title + description.
  - Retrieves **top 3 relevant skills** (similarity > 0.50 only).
  - Injects **compressed signatures** (name + one-line description + params). Hard budget: **≤800 tokens**.
  - If zero skills match, the new skill is synthesized standalone.
- **Fact-Anchored Synthesis (Mitigation #12)**: Injects exact verified syntax from distilled facts into both code and test generation to prevent tautological mocks.
- Generates `unittest` test suites using `unittest.mock` with syntax-verifying assertions.

### Subsystem 4: Skill Compiler & Sealed Execution Engine (`agent/engine/validator.py`)
- **Zero-Trust Positive-Match Compiler (Mitigation #25)**: Every AST node is walked through a `SkillCompiler` where `generic_visit()` defaults to **rejection**. Only explicitly handled node types pass compilation. All dunder attribute access (`__bases__`, `__class__`, `__globals__`, etc.) is unconditionally rejected. The compiler also performs:
  - **Tiered Import Allowlist** resolution:
    - **Tier 1 (Always Allowed)**: `json`, `re`, `math`, `typing`, `dataclasses`, `datetime`, `collections`, `pathlib`, `shlex`, `argparse`, `textwrap`, `enum`, `uuid`, `hashlib`, `base64`, `copy`, `functools`, `itertools`.
    - **Tier 2 (Function-Scope Only, must be mocked)**: `subprocess`, `os`, `os.path`, `shutil`, `requests`, `httpx`. Must appear only inside `ast.FunctionDef` nodes. Test code must contain `@patch` for each.
    - **Extended Tier 2 (Sandbox-Image-Gated)**: `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`, `bs4`, `lxml`. Allowed only if the resolved sandbox image (Mitigation #57) contains the module.
    - **Tier 3 (Never Allowed)**: `socket`, `ctypes`, `importlib`, `__import__`, `compile`, `code`, `sys.exit`, `signal`, `multiprocessing`, `threading`, `webbrowser`, `http.server`, `smtplib`, `ftplib`, `pickle`, `marshal`, `shelve`, `builtins`.
  - **Import-driven sandbox image selection (Mitigation #57)**: After compilation passes, the extracted import set is mapped to the smallest satisfying sandbox image (`sandbox-stdlib`, `sandbox-web`, `sandbox-scientific`, or `sandbox-full`).
- **Dependency Check**: Extracts all imports from compilation, verifies availability in the selected sandbox image tier. Fails compilation immediately if any import is not in any tier — no sandbox is spawned for unavailable dependencies.
- **gVisor Sealed Execution (Mitigation #38)**: All real execution (mock tests and real dry-runs) happens inside the gVisor sealed envelope:
  - gVisor with read-only squashfs rootfs, `--network=none`, `--host-uds=false`, `--net-raw=false`, `--file-access=exclusive`.
  - Input via stdin (JSON payload), output via stdout (captured by host).
  - Resource limits: 2s CPU (gVisor primary) / 4s (host failsafe), 128MB memory (gVisor primary) / 256MB (host failsafe), 32 PIDs max, 64 open files max.
  - No host bind-mounts, no secrets in environment.
- **Typed Result Relayer (Mitigation #49)**: After gVisor exits, the host-side `parse_skill_output()` validates stdout against the immutable `SkillResultSchema`. Schema violation → `FAILED_VERIFICATION` regardless of exit code.
- **Real Dry-Run Executor (Mitigation #35/#38/#44/#51)**: Executes the skill against the real tool/engine inside the sealed envelope. Local CLIs/engines verify as `real_local` with no network; external-API skills verify as `real_external` only against an allowlisted host (Mitigation #51, via `socat`/`iptables` forwarding) or a deterministic local fixture server. Skills passing both mock tests and a real dry-run earn `real_local`/`real_external`; mock-only skills earn `mock`.
- **Pure-Deterministic Verification (Mitigation #35, optional)**: For side-effect-free skills (no Tier 2 imports), runs the skill twice with different PRNG seeds and compares output hashes. Mismatch → flagged for human review (not auto-quarantined).
- **Registration Gate**: Only skills with 100% test pass rate AND successful compilation are saved and registered. `verification_tier` (`compiled` / `static` / `mock` / `real_local` / `real_external`) and optional `determinism_verified` flag are recorded and displayed in `skills`.

### Subsystem 5: Two-Stage Confidence-Gated Retriever (`agent/engine/retriever.py`)
- **Cold Start**: Detects empty memory, returns dedicated welcome message without misleading scores.
- **Hybrid Search (Mitigation #30)**: Combines L2-normalized dense cosine similarity with SQLite FTS5 sparse keyword matching. CLI-token queries boost sparse weight. Final score: weighted combination of both.
- **Multi-Source Retrieval**: Searches `semantic_facts`, `context_passages`, and — for project-aware queries (file names, classes, paths, project concepts) — **`project_files` from Tier 4 Project Memory** (Mitigation #50). Project rows are matched by path/keyword FTS5 and semantic-summary embedding. (*The `concept_relationships` table and 1-hop traversal were removed in V1 — Mitigation #34.*)
- **Multi-Topic Retrieval**: Runs both unfiltered and topic-filtered queries, merges and deduplicates results.
- **Two-Stage Gate** (thresholds read from `data/calibration.json`, Mitigation #37; defaults 0.65 / 0.80 until calibrated):
  - Stage 1A ($< 0.65$): Immediate honest refusal.
  - Stage 1B ($\ge 0.80$): Grounded answer with **closed-world constraint** (Mitigation #29).
  - Stage 2 ($0.65 \le \text{score} < 0.80$): Fast brain discriminator check.
- **Closed-World Generation (Mitigation #29)**: ALL answer generation (including hard-pass zone) includes the strict constraint: *"Answer using ONLY the provided facts. If the specific detail is not stated, say 'my stored memory does not contain that specific detail.' NEVER extrapolate or guess."*

### Subsystem 6: Pluggable Decision Brains (`agent/brains/`)
- **Provider registry (`brains.json`, Mitigation #56)**: Users hand-edit `brains.json` to register any providers they want. `factory.py` loads it at startup and exposes each registered provider by name via `--brain <provider>` or `AI_BRAIN`.
- **`MockBrain`**: Always registered (`kind: "mock"`). 100% offline, deterministic, declares known topics, warns on unknown topics, enables CI/CD. Default fallback whenever no provider has a usable API key.
- **Built-in provider kinds** (schema in Mitigation #56):
  - `openai_compatible` (OpenCodeZen/DeepSeek, Codex, vLLM, Ollama, LM Studio, any `/v1/chat/completions` endpoint) — via LiteLLM.
  - `google_genai` (Gemini `gemini-2.5-flash` / `gemini-2.5-pro`) — via Google GenAI SDK.
  - `anthropic` (Claude) — via LiteLLM / direct SDK.
  - `mock` (offline).
- **`factory.py`**: Reads `brains.json`, resolves the selected provider (`--brain` flag > `AI_BRAIN` env var > `active` field in `brains.json` > `mock`), checks the provider's API key (from `api_key_env`), and falls back to MockBrain with a warning if the key is missing or the provider is unregistered. All real providers share the 3-retry/backoff and robust JSON extraction (Mitigations #8/#9).

### Subsystem 7: Foundational Knowledge Seeder (`agent/memory/seeder.py`)
- **Trigger**: Runs automatically on startup if `semantic.db` has zero active facts OR `skills.db` has zero verified skills. Also triggered by `--reseed` CLI flag.
- **Fact Loading**: Reads `seed_data/facts.json`, computes L2-normalized embeddings at runtime (not pre-computed), inserts into `semantic.db` with `source_type = "seed"`.
- **Skill Loading**: Reads each `.py` file from `seed_data/skills/`, passes it through `validator.py` (Tiered Import Allowlist + sandbox test), registers verified skills with `source_type = "seed"` in `skills.db`, copies files to `skills/`.
- **Idempotency**: Fact deduplication (>0.95 similarity) and `INSERT OR IGNORE` prevent double-seeding.
- **Versioning**: Compares `config.SEED_VERSION` against the version stored in episodic memory. If changed, re-seeds and supersedes old seed facts.
- **Dependency Order**: Runs after `embeddings.py` model init (Mitigation #2) and before `orchestrator.py` accepts user input.

### Subsystem 8: The Heartbeat Daemon (`agent/engine/heartbeat.py`)
- Runs as an asynchronous background task during idle turns.
- Inspects `semantic.db` staleness timestamps and `skills.db` unverified tools.
- Evaluates top-priority actionable leaves from `data/goals.db`.
- Initiates autonomous background learning, re-testing, or fact-refreshing within strict safety rate limits (max 3 actions/hour).

### Subsystem 9: Metacognitive Reflection Engine (`agent/engine/reflection.py`)
- Consolidates episodic logs, audits user corrections (`source_type = "user_corrected"`), and recalibrates topic competence scores.
- Periodically updates `data/self_model.json` with empirical skill pass ratios.
- Automatically flags conflicting or stale facts for targeted ingestion.

### Subsystem 10: Directed Acyclic Goal Graph Manager (`agent/memory/goals.py`)
- SQLite-backed DAG storage (`data/goals.db`) for tracking user-assigned or self-generated long-term goals and subgoals.
- Evaluates prerequisite dependency trees to ensure subgoals execute in valid topological order.
- Enforces the two-tier depth policy (Mitigation #47/#54): autonomous goals hard-capped at depth 2; user-supervised goals up to depth 4, each 2→3 and 3→4 expansion gated by explicit `[Y/n]` approval and flagged `supervised = true`.
- Parent-relevance discriminator (cosine ≥ 0.70 vs parent objective) applies at every depth level.

### Subsystem 11: Persistent Self-Model Manager (`agent/memory/self_model.py`)
- Maintains persistent JSON state (`data/self_model.json`) tracking identity, focal areas, known strengths, and knowledge gaps.
- Generates compressed meta-awareness prompts injected into reasoning Brains.
- **Write-protected**: sole writer of `self_model.json`. Accepts competence updates only from benchmark pass/fail, subprocess exit codes, user corrections, and user ratings (Mitigation #45/#52/#55); tamper detection rolls back unauthorized changes.

### Subsystem 12: Permission & HITL Governor (`agent/engine/governor.py`)
- Enforces Human-in-the-Loop (HITL) permission tiers:
  - **Tier 0 (Autonomous)**: Read docs, vector search, AST linting.
  - **Tier 1 (Guarded Autonomous)**: Sandbox/Docker test execution.
  - **Tier 2 (Explicit Approval Required)**: File writes, package installs, live host commands (prompts user with `[Y/n]` confirmation in terminal).
- Enforces daily token/call ceilings and step/failure circuit breakers.

### Subsystem 13: Deterministic Task State Machine (`agent/engine/state_machine.py`)
- Manages atomic step execution against `data/active_task.json`.
- Reads disk state, dispatches single action, checks circuit breakers, records step outcome, and transitions task states (`PENDING` $\to$ `RUNNING` $\to$ `VERIFYING` $\to$ `COMMITTED` $\to$ `FAILED`).

### Subsystem 14: Objective Benchmark Harness (`agent/engine/benchmark.py`)
- Executes immutable test suite in `tests/benchmark_suite/`.
- Computes empirical competence scores for `data/self_model.json`.
- Disallows agent code from modifying benchmark fixtures.

### Subsystem 15: Project Memory Manager (`agent/memory/project.py`)
- Sole writer of `data/projects.db` (Tier 4 Project Memory; Mitigation #50/#52).
- `project index <path>` performs an initial full-tree scan (with `.gitignore`-style exclusions); subsequent reads/writes update `project_files` incrementally via SHA-256 hash diffs (re-embed and re-summarize only changed files).
- `project decision <project> "<title>" "<decision>"` records architecture decisions with `related_files_json`.
- Exposes hybrid search over `project_files` (path/keyword FTS5 + semantic summaries) to the retriever (Subsystem 5) for cross-tier, code-aware answers.

### Subsystem 16: Reasoning & Metacognitive Engine (`agent/memory/reasoning.py`, `agent/engine/critic.py`, `agent/engine/verifier.py`, `agent/engine/strategy_injector.py`)
- **Reasoning Memory Manager (`agent/memory/reasoning.py`)**: Sole writer of `data/reasoning.db` (Tier 2.5; Mitigation #61). Stores structured SHyAOEDRGL reasoning episodes with verification flags, permanent retention, and novelty metrics.
- **Strategy Injector (`agent/engine/strategy_injector.py`)**: Injects archetype-specific prompt templates from `strategy_index` (Mitigation #62), stamps `strategy_label` and `prompt_hash` into episodic telemetry (Mitigation #67) before task execution.
- **Lateral Critic & Arbiter (`agent/engine/critic.py`)**: Dispatches two parallel solver brains; skips critic on consensus; invokes third-party arbiter on divergence (Mitigation #63).
- **Symbolic SRT Verifier (`agent/engine/verifier.py`)**: Deterministically parses Structured Reasoning Traces and verifies logical entailment using formal rules/provers before admitting traces to training datasets (Mitigation #65).
- **ZPD Benchmark Evaluator (`agent/engine/benchmark.py`)**: Runs 5-round binary search across parameterized difficulty knobs in `tests/reasoning_suite/` to compute granular `zpd_ceilings` for `self_model.json` (Mitigation #66).
- **Training Dataset Builder (`agent/engine/dataset_builder.py`)**: Filters reasoning episodes via novelty (>0.7) and solution-path entropy (>2σ) gates, outputting DPO/SFT dataset pairs for model fine-tuning (Mitigation #68, #69).

---

## 6. Complete Scenario Lifecycle (Walkthrough)

### Phase A: Before Learning (Seeded Knowledge)
- **User**: *"How do I check my git status?"*
- **Agent**: (Seed facts loaded at boot → retrieval score: 0.91 → direct pass)
  `"Run git status to see modified, staged, and untracked files in your repository. I also have an executable skill stdlib_git_status ready if you want me to run it for you."`
- **User**: *"How do I create a GitHub Release using the CLI?"*
- **Agent**: (Seed facts cover Git basics but not GitHub CLI releases → retrieval score: 0.58 → hard reject)
  `"I have foundational Git knowledge but haven't learned about GitHub CLI releases specifically. Want me to study it? Use learn GitHub."`

### Phase B: The Learning Directive
- **User**: `"learn GitHub"`
- **Agent**:
  - `[1/4]` Formulates curriculum (Git vs GitHub, Branching, `gh` CLI commands). Scope: "GitHub CLI releases & PRs (5 units)."
  - `[2/4]` Ingests documentation via Trusted Docs Registry → Jina Reader → fallback chain.
  - `[3/4]` Extracts structured knowledge: 18 facts (concept / syntax / troubleshooting), 4 context passages. Contradiction gating + deduplication applied.
  - `[4/4]` Synthesizes `github_create_release.py` and `github_create_pr.py`. Validates imports. Runs mocked sandbox tests (and a real dry-run verification when the runtime is available — Mitigation #35). Registers verified skills.
  `"✔ Finished studying GitHub. Extracted 18 facts (8 concepts, 6 syntax, 4 troubleshooting), 4 context passages, 2 verified skills."`

### Phase C: After Learning (Deep Understanding)
- **User**: *"How do I create a GitHub Release using the CLI?"*
- **Agent**: (Hybrid search: dense=0.88, sparse=0.92 → combined=0.90 → hard pass + closed-world grounding)
  `"To create a release: gh release create <tag> [--title <string>] [--notes <string>]. Example: gh release create v1.0.0 --title "Initial Release" --notes "First build". I also have skill github_create_release ready."`
- **User**: *"Should I rebase or merge my feature branch?"*
- **Agent**: (Retrieves `concept`-type facts on rebase vs merge + a context passage on branching strategies)
  `"Use rebase for clean linear history on private branches that haven't been shared. Use merge for shared branches to preserve history and avoid rewriting commits others depend on. Key tradeoff: rebase is cleaner but destructive; merge is safe but creates merge commits."`
- **User**: *"What's the max upload size for GitHub releases?"*
- **Agent**: (Hybrid score: 0.83 → hard pass, but closed-world check detects missing specific detail)
  `"I found related information about GitHub releases (creating, tagging, notes), but my stored memory does not contain the specific upload size limit. Want me to research this? Use learn GitHub-release-limits."`

### Phase D: Selective Forgetting
- **User**: `"forget GitHub"`
- **Agent**: `"Forgot all learned knowledge about 'GitHub': 18 facts, 4 passages, 2 skills removed. Seed facts preserved. Event recorded in audit log."`

### Phase E: In-Place Correction
- **User**: `"correct git 'rebase is safe on shared branches' -> 'rebase should NEVER be used on shared/public branches because it rewrites commit history that other developers have already based work on'"`
- **Agent**: `"Corrected: superseded old fact (confidence: 0.85) with your correction (confidence: 1.0, source: user_corrected). This correction has top authority and will not be overwritten by future learn commands."`

### Phase F: Autonomous Proactive Heartbeat Cycle (Proactive Mode)
- **Background State**: User is idle for 30 minutes. Heartbeat daemon triggers cycle.
- **Agent (Heartbeat Execution)**:
  - `[Perceive]`: Detects that `semantic.db` has 12 facts on Docker that have not been refreshed in 180 days; finds 1 unverified skill `docker_build_image` at `mock` tier.
  - `[Evaluate]`: Checks `data/goals.db` — active goal: *"Maintain Docker CLI competence"*.
  - `[Plan]`: Schedules background refresh of Docker CLI reference docs and headless execution smoke test.
  - `[Act]`: Ingests official docs via `TRUSTED_DOCS`, updates 3 stale facts, executes dry-run smoke test in headless sandbox, upgrading skill to `verification_tier = real_local` (local CLI, no network needed).
  - `[Reflect]`: Updates `data/self_model.json` (`docker` competence raised to 0.90, `skills_verified` incremented). Logs heartbeat event in `episodic.db`.
- **User returns & types**: `"self-model"`
- **Agent**: Displays updated competence matrix, recent autonomous actions, and active goals.

### Phase G: Project-Aware Collaboration (Tier 4 Project Memory + Supervised Goals)
- **User**: `"project index /path/to/your/project"` then *"Where is player movement handled in my Unity project?"*
- **Agent**: `[Perceive]` Cross-tier retrieval searches `project_files` (path/keyword FTS5 + semantic summary) alongside Unity-API facts in `semantic.db`.
  `"Player movement is handled in Assets/Scripts/PlayerController.cs (role: player controller MonoBehaviour, 214 lines). It reads Input.GetAxis('Horizontal'/'Vertical') and applies transform.Translate in Update(). I also have a fact on Unity's CharacterController.Move for collision-safe movement if you want to switch to it."`
- **User**: `"learn about Unity serialization for a save system"` (supervised task).
- **Agent**: Creates a supervised goal tree in `goals.db`. At depth 2→3 expansion it shows the proposed subtree and asks `[Y/n]`; on approval, marks deeper nodes `supervised = true`.
  `"Approved: Build save system → Design schema → Research Unity serialization (depth 3, supervised). I will not expand further without asking."`
- **User**: `"project decision MyUnityGame "Save format" "Use JSON via JsonUtility for deterministic, debuggable serialization."`
- **Agent**: Records the decision in `project_decisions` (through `memory/project.py`). Subsequent suggestions respect it.

---

## 7. CLI Command Reference

| Command | Description |
| :--- | :--- |
| `learn <topic>` | Decompose curriculum, crawl docs, distill structured knowledge & synthesize skills |
| `ask <question>` | Query memory with hybrid search + closed-world grounded generation |
| `correct <topic> "<old>" -> "<new>"` | Surgically correct a single fact with top authority |
| `demo` | Run the full 7-phase demonstration scenario |
| `skills` | List all synthesized & verified executable skills with verification tiers (`mock`/`real_local`/`real_external`) |
| `facts [topic]` | Inspect semantic facts by topic and knowledge type |
| `passages [topic]` | Inspect context passages in Tier 2 memory |
| `self-model` | Display persistent agent self-model, competence matrix, strengths, and knowledge gaps |
| `goals [add\|list\|complete]` | Inspect or update the Directed Acyclic Goal Graph (shows `supervised` flags; depth >2 requires approval) |
| `projects [list\|status]` | List indexed projects and Tier 4 Project Memory stats |
| `project index <path>` | Full-tree index of a workspace into `projects.db` (Mitigation #50) |
| `project decision <project> "<title>" "<decision>"` | Record an architecture decision into `project_decisions` |
| `rate <topic> good\|bad` | Submit a user rating that feeds the competence matrix (Mitigation #55) |
| `heartbeat [status\|run\|toggle]` | View heartbeat daemon status or trigger an immediate autonomous proactive cycle |
| `reflect` | Manually trigger metacognitive reflection and competence consolidation |
| `benchmark [run\|status]` | Run the immutable integration benchmark suite and update empirical competence |
| `task [status\|abort]` | Inspect or abort the active deterministic task state machine |
| `stats` | Display item counts across all memory tiers + competence breakdown |
| `run-skill <name> <k=v>` | Execute a learned skill |
| `forget <topic>` | Remove learned facts, passages, and skills (preserves seeds) |
| `forget <topic> --include-seeds` | Remove ALL knowledge including seed data |
| `refresh <topic>` | Re-run ingestion and update stale facts |
| `purge-superseded` | Delete all superseded facts from semantic.db |
| `calibrate-thresholds` | Run precision/recall sweep on labeled eval set; write model-keyed thresholds. Refuses if a category is below its minimum size (Mitigation #53) |
| `calibrate add <category> <query>` | Add a labeled example to `calibration/queries.json` (operator-only) |
| `set-threshold <val>` | Adjust confidence threshold dynamically (overrides calibrated values for this session) |
| `reset` | Clear all memory tiers, indexed projects, and reset self-model to initial state |
| `--brain <provider>` | Select brain backend by name from `brains.json` registry (e.g., `mock`, `gemini`, `claude`, `opencodezen`, `codex`; Mitigation #56). Default: `active` field, else MockBrain |
| `--stage 1\|2\|3` | Select rollout stage (1: Supervised, 2: Bounded Maintenance, 3: Constrained Goal) |
| `--permission-tier 0\|1\|2` | Set execution permission tier (Tier 2 requires `[Y/n]` prompts) |
| `--demo` | Execute 7-phase walkthrough and exit |
| `--dry-run` | Preview curriculum plan without API calls |
| `--reseed` | Force re-run foundational knowledge seeding |
| `--no-daemon` | Disable background autonomous heartbeat loop |

---

## 8. Implementation Safeguards Summary (All 70 Mitigations)

| # | Safeguard | Where Enforced |
| :--- | :--- | :--- |
| 1 | One-directional import graph (no circular imports) | All modules |
| 2 | First-run embedding download with Rich spinner | `memory/embeddings.py` |
| 3 | `pathlib.Path` everywhere, quoted paths for spaces | All path operations |
| 4 | Resilient fetch chain (Jina → Trafilatura → BeautifulSoup → optional Playwright) + abort guard, no synthetic fallback | `engine/ingest.py` |
| 5 | DuckDuckGo rate limit (2 queries/unit, 1s delay) | `engine/ingest.py` |
| 6 | Source URL + `ingested_at` + `staleness_days` on facts | `models.py`, `memory/semantic.py` |
| 7 | ≤4000 char chunks, >0.95 similarity deduplication | `engine/synthesizer.py` |
| 8 | 3-stage JSON extraction with safe fallback | `brains/gemini_brain.py`, `brains/claude_brain.py` |
| 9 | 3-retry exponential backoff on API calls | `brains/gemini_brain.py`, `brains/claude_brain.py` |
| 10 | `--dry-run` flag, API call count summary | `cli.py` |
| 11 | MockBrain known-topics declaration and warning | `brains/mock_brain.py` |
| 12 | Fact-anchored synthesis prompts, syntax-verifying assertions | `engine/synthesizer.py` |
| 13 | Import verification via sandbox image tier resolution (M57) | `engine/validator.py` |
| 14 | `shutil.which()` for CLI executables, `pathlib.Path` paths | Generated skill code |
| 15 | Topic-prefixed skill names, UNIQUE constraint | `memory/procedural.py` |
| 16 | Graceful partial-state recovery, `INSERT OR REPLACE` | `engine/orchestrator.py` |
| 17 | In-memory numpy matrix cache for <10K facts | `memory/semantic.py` |
| 18 | 90-day episodic retention policy, permanent learning events | `memory/episodic.py`, `config.py` |
| 19 | `purge-superseded` CLI command, `VACUUM` | `cli.py`, `memory/semantic.py` |
| 20 | SQLite WAL mode on all connections | `memory/episodic.py`, `memory/semantic.py`, `memory/procedural.py` |
| 21 | Cold start detection with dedicated welcome message | `engine/retriever.py` |
| 22 | Multi-topic merged retrieval with deduplication | `engine/retriever.py` |
| 23 | Technical-context default interpretation | `brains/base.py`, `engine/planner.py` |
| 24 | `forget <topic>` CLI command with audit logging | `cli.py`, `engine/orchestrator.py` |
| 25 | **Zero-trust positive-match compiler** (`generic_visit`=reject): every AST node explicitly handled; blacklist replaced by whitelist; all dunder access unconditionally rejected | `engine/validator.py` |
| 26 | RAG-based skill context injection (top 3, >0.50, ≤800 tok) | `engine/synthesizer.py` |
| 27 | Trusted Docs Registry + Domain Authority Scoring + Freshness Check | `engine/ingest.py`, `data/trusted_docs.json` |
| 28 | Foundational knowledge pre-seeding with seed versioning | `memory/seeder.py`, `seed_data/`, `config.py` |
| 29 | Closed-world generation guard on ALL answer zones | `engine/retriever.py` |
| 30 | Hybrid search: dense cosine + FTS5 sparse keyword matching | `memory/semantic.py`, `memory/embeddings.py`, `engine/retriever.py` |
| 31 | In-place `correct` command with `user_corrected` top authority | `cli.py`, `engine/orchestrator.py`, `memory/semantic.py` |
| 32 | 3-category knowledge type taxonomy (concept, syntax, troubleshooting); `depth_level` dropped for V1 | `engine/synthesizer.py`, `models.py` |
| 33 | Context passages (200–500 word long-form memory, max 10/unit) | `engine/synthesizer.py`, `memory/semantic.py` |
| 34 | `concept_relationships` table and graph traversal **removed**; hybrid FTS5/dense search only | `engine/synthesizer.py`, `memory/semantic.py`, `engine/retriever.py` |
| 35 | Verification tier ladder: compiled → static → mock → **real_local** → **real_external** → optional **pure_deterministic**; official schema anchors; revision loop on real failure; dual-run determinism check (flag, not quarantine) | `engine/validator.py`, `engine/synthesizer.py`, `engine/orchestrator.py` |
| 36 | Multi-runtime skill synthesis (Python V1; C#/GDScript/C++/shaders gated behind `--runtime`) with engine smoke-test feedback | `engine/synthesizer.py`, `engine/validator.py`, `memory/procedural.py`, `config.py` |
| 37 | Calibrated confidence thresholds via labeled eval set (`calibrate-thresholds`, model-keyed) | `cli.py`, `memory/semantic.py`, `engine/retriever.py` |
| 38 | **gVisor sealed execution envelope**: userspace kernel syscall interception; hardened gofer (`--host-uds=false`, `--net-raw=false`, `--file-access=exclusive`); read-only squashfs rootfs (no host bind-mounts); `--network=none`; resource limits (2s CPU/128MB RAM gVisor primary, 2× host failsafe); SIGKILL on violation; Windows Job Object fallback; Wasm deferred to V3 | `engine/validator.py`, `cli.py`, `config.py` |
| 39 | Resilient ingestion: response cache, provider health tracking, headless-JS escape hatch, `fetch_quality` provenance | `engine/ingest.py`, `models.py`, `memory/semantic.py` |
| 40 | Persistent Self-Model (`self_model.json`) & Empirical Competence Matrix | `memory/self_model.py`, `models.py` |
| 41 | Heartbeat Daemon (Autonomous background perceive-evaluate-plan-act loop with rate limits) | `engine/heartbeat.py`, `main.py` |
| 42 | Directed Acyclic Goal Graph (`goals.db`) & dependency-aware prerequisite resolution | `memory/goals.py`, `cli.py` |
| 43 | Metacognitive Reflection Engine (Consolidation sleep cycle & competence score recalibration) | `engine/reflection.py`, `cli.py` |
| 44 | Headless Runtime Harness & compiler error feedback loops for autonomous bug fixing | `engine/validator.py`, `engine/synthesizer.py` |
| 45 | External Metric Anchoring: competence updated strictly via immutable benchmark suites & exit codes | `engine/benchmark.py`, `memory/self_model.py` |
| 46 | Circuit Breakers: hard 5-step cap, 2-failure limit, daily call/token ceiling | `engine/governor.py`, `engine/heartbeat.py` |
| 47 | Two-tier Goal DAG depth: autonomous max 2; supervised max 4 with approval (M54) + parent relevance discriminator at all depths | `memory/goals.py` |
| 48 | Automated Full Regression Suite triggered on any skill mutation or synthesis | `engine/validator.py`, `engine/orchestrator.py` |
| 49 | **Deterministic Task FSM + Typed Result Relayer**: FSM `VERIFYING→COMMITTED` gated by host-side `parse_skill_output()` against immutable `SkillResultSchema`; skill self-reported exit code is NOT trusted; schema violation = `FAILED_VERIFICATION`; end-to-end `trace_id` telemetry | `engine/state_machine.py`, `engine/validator.py`, `memory/episodic.py` |
| 50 | Tier 4 Project Memory (`projects.db`): automatic file indexing, write-protected, cross-tier retrieval | `memory/project.py`, `engine/retriever.py`, `cli.py` |
| 51 | Real-Local vs Real-External verification: network allowlist via `socat`/`iptables` forwarding (not sandbox config) + deterministic local API fixture server | `engine/validator.py`, `config.py`, `cli.py` |
| 52 | Self-model & project-memory write protection: sole-writer modules, tamper detection, rollback | `memory/self_model.py`, `memory/project.py`, `engine/governor.py`, `main.py` |
| 53 | Calibration dataset minimum viability (50/30/20/30) with refusal-to-calibrate and periodic re-runs | `cli.py`, `engine/retriever.py`, `config.py` |
| 54 | User-supervised deep goal trees: depth 4 with `[Y/n]` approval per expansion beyond depth 2 | `memory/goals.py`, `cli.py` |
| 55 | Competence signals for non-skill domains: topic quizzes, user ratings, fact-verification events | `engine/reflection.py`, `memory/self_model.py`, `cli.py` |
| 56 | User-defined brain provider registry (`brains.json`): OpenAI-compatible/Google/Anthropic/mock, key via env var, selection precedence | `brains/factory.py`, `brains/*`, `cli.py`, `config.py` |
| 57 | **Compiler→sandbox image selection pipeline**: import-driven image resolution (`sandbox-stdlib`/`sandbox-web`/`sandbox-scientific`/`sandbox-full`); compilation fails immediately if module not in any image; no `pip install` inside sandbox | `engine/validator.py`, `config.py`, `engine/synthesizer.py` |
| 58 | **Sandbox I/O protocol**: stdin/stdout only, no host bind-mounts; input via stdin JSON, output via stdout JSON; post-exit retrieval via `runsc exec cat`; no bidirectional communication during execution | `engine/validator.py`, `engine/state_machine.py` |
| 59 | **gVisor rootfs immutability**: read-only squashfs layers, content-addressable SHA-256 verification, ephemeral per-execution, no `pip install`, tamper detection on startup | `engine/validator.py`, `engine/governor.py`, `Dockerfile.sandbox-*` |
| 60 | **Synthesizer dependency awareness**: available-modules prompt injection, compiler→image upgrade logic, `requires_custom_image` flag for unavailable dependencies | `engine/synthesizer.py`, `engine/validator.py`, `memory/procedural.py` |
| 61 | **Tier 2.5 Reasoning Memory** (`reasoning.db`): `reasoning_episodes` table with SHyAOEDRGL tuple, `verified` gate, permanent retention, failure-first value, linked to episodic via `trace_id` | `memory/reasoning.py`, `engine/reflection.py`, `engine/heartbeat.py` |
| 62 | **Cross-cutting reasoning profile** in `self_model.json`: `global_scores` (prior), `domain_deltas` (weekly SQL aggregation posterior), `strategy_index` (meta-policy); domain-queryable but not skill-attached; `novelty_score` orthogonal to `complexity_score` | `memory/self_model.py`, `engine/planner.py`, `engine/reflection.py` |
| 63 | **Lateral Critic / Adversarial Verifier**: two-solver parallel dispatch; skip critic if agreement (66% inference cost saving); spawn third-instance Arbiter only on divergence; Critic forced to search for unsupported assumptions/contradictions/counterexamples | `engine/critic.py`, `engine/orchestrator.py`, `brains/factory.py` |
| 64 | **Hypothesis Competition Engine + Counterfactual Training**: `novelty_score > 0.8` activates 3–5 competing hypotheses + discriminating tests; post-success Heartbeat generates counterfactuals (input scale, edge-cases, state ordering); stores conditional correctness as `reasoning_episode` | `engine/heartbeat.py`, `engine/planner.py`, `memory/reasoning.py` |
| 65 | **Structured Reasoning Trace (SRT) + Symbolic Verifier**: reasoning benchmarks require `{conclusion, premises, inference_rule, rejected_hypotheses}` SRT output; Z3/Prolog host-side verifier checks logical soundness; logically unsound traces rejected from training data regardless of correct final answer | `engine/verifier.py`, `engine/validator.py`, `memory/reasoning.py` |
| 66 | **Reasoning Benchmark Suite with ZPD Binary Search**: 6-category reasoning suite (decomposition/hypothesis_testing/causal/counterexample/planning/adversarial); parameterized difficulty knobs; 5-round binary search finds ZPD ceiling; `zpd_ceilings` written to `self_model.json`; separate from knowledge benchmarks | `engine/benchmark.py`, `tests/reasoning_suite/`, `memory/self_model.py` |
| 67 | **Episodic Telemetry Hotfix** (pre-migration dependency for M#62): 6 new episodic columns (`prompt_hash`, `strategy_label`, `novelty_score`, `reasoning_domain`, `outcome_class`, `hypothesis_count`); `active_task.json` gains `strategy_label` + `prompt_hash`; must be locked before first episodic INSERT | `models.py`, `engine/planner.py`, `engine/strategy_injector.py`, `engine/reflection.py` |
| 68 | **Novelty & Entropy Filter**: training corpus admission requires `novelty_score > 0.7` OR unexpected failure + solution-path entropy > 2σ from centroid + `verified = true`; minimum 500 episodes before Dataset Builder runs | `engine/dataset_builder.py`, `memory/reasoning.py` |
| 69 | **Experience → Training-Data Pipeline** (V4, model-weight-control gated): verified episodes → DPO preference pairs → LoRA fine-tune → benchmark gate (must improve ≥3/6 ZPD ceilings vs. previous model) → deploy only if better; SFT never on unverified data | `engine/dataset_builder.py`, `engine/trainer.py`, `engine/benchmark.py` |
| 70 | **LoRA Co-Processor / MoA Router** (V3 stepping stone): route on `complexity_score` (not `novelty_score`) to cloud API (routine) vs. LoRA adapter (complex reasoning); promote adapter if beats cloud API on 80% of reasoning benchmarks over 30 days; configurable as `moa_router` in `brains.json` | `brains/moa_router.py`, `brains/factory.py`, `engine/planner.py`, `config.py` |

---

## 9. Documented V1 Boundaries & Engineered Build Order

> [!NOTE]
> These are known architectural constraints, not bugs. They define the honest capability boundary of the system, its safety governors, and its phased implementation sequence.

### Engineered Build Order & Staged Verification

The engine is constructed in a strict **Memory Core $\to$ Project Grounding $\to$ Skill Execution $\to$ Planning \& FSM $\to$ Metacognitive Autonomy $\to$ Domain Validation $\to$ Evolution** dependency ladder. Each phase must pass its concrete proof milestone before proceeding to the next:

1. **Phase 0: Memory Core & Offline Loop (Zero Setup, MockBrain)**
   - **Scope**: Tier 1 Episodic (`episodic.db`), Tier 2 Semantic (`semantic.db`), Tier 3 Procedural metadata (`skills.db`), FastEmbed ONNX local embeddings, closed-world generation guard (Mitigation #29), and zero-dependency offline loop with `MockBrain` (Mitigation #11).
   - **Proof Milestone**: Seed initial facts $\to$ Query git status (passes with $\ge 0.80$ retrieval) $\to$ Query missing knowledge (honestly refuses) $\to$ Skill learned in run 1 is reused from disk on run 2.

2. **Phase 1: Real Brain & Project Codebase Memory (Grounded Retrieval)**
   - **Scope**: Direct API brain integration (Gemini, Claude, OpenAI, Ollama), Tier 4 Project Codebase Memory (`projects.db`, Mitigation #50), SHA-256 hash diff incremental indexing, and cross-tier grounded retrieval.
   - **Proof Milestone**: Index workspace (`project index <path>`) and ask: *"Where is player movement handled in my project?"* $\to$ Verifies the agent retrieves exact files and facts with hallucination-resistant grounded retrieval achieving >= 98% factual precision on the closed-world benchmark test suite with verified refusal on out-of-distribution queries.

3. **Phase 2: Real Skill Execution & Safety (Subprocess & Revision Loop)**
   - **Scope**: AST tiered allowlist (Mitigation #25), isolated subprocess execution with 5s timeout, typed output validation (`SkillResultSchema`, Mitigation #49), real dry-run execution (`real_local`), and 2-retry synthesizer revision loop (Mitigation #35).
   - **Proof Milestone**: Synthesizer generates a tool, sandbox catches real compiler/runtime stderr on simulated error, revision loop automatically repairs code, and validator registers verified tool in `skills.db`.

4. **Phase 3: Supervised Planning, Task FSM & Chat/Stream Interaction**
   - **Scope**: Directed Acyclic Goal Graph (`goals.db`, Mitigation #42), deterministic Task FSM (`data/active_task.json`, Mitigation #49), step limits, 2-failure circuit breaker, HITL Tier-2 approval gates (`[Y/n]` prompts, Mitigation #46), and live chat adapter with viewer episodic memory.
   - **Proof Milestone**: Execute multi-step task, persist state on disk across turns, simulate crash/interruption (`Ctrl+C`), resume cleanly from disk without repeating prior steps, and prompt before destructive file writes.

5. **Phase 4: Autonomous Maintenance & Metacognitive Reasoning (V2)**
   - **Scope**: Heartbeat background daemon (`agent/engine/heartbeat.py`, Mitigation #41), Metacognitive reflection cycles (`agent/engine/reflection.py`, Mitigation #43), persistent Self-Model (`data/self_model.json`, Mitigation #40), anti-tamper protection (Mitigation #52), Tier 2.5 Reasoning Memory (`reasoning.db`, Mitigation #61), Lateral Critic (Mitigation #63), and ZPD difficulty calibration (Mitigation #66).
   - **Proof Milestone**: Heartbeat detects stale facts during idle and refreshes them within 3 actions/hour rate limit; tamper watchdog rolls back manual JSON edits; Lateral Critic arbitrates divergent multi-solver reasoning.

6. **Phase 5: Domain Validation & Engine Integration (Unity / Blender MCP)**
   - **Scope**: Real-world external domain stress-test. Unity MCP (C# scripts, scene/prefab inspection, headless UTF test runner; Mitigation #36), Blender MCP (headless Python 3D mesh generation/export), Roslyn compile error revision loop (`CS0246`, `CS1061`), and Git feature branch isolation (`ai-feat/*`).
   - **Proof Milestone**: Dispatch game development task $\to$ Agent authors C# scripts, generates 3D FBX assets via Blender, runs headless Unity Test Runner, automatically fixes compiler errors, and verifies all tests pass.

7. **Phase 6: Evolutionary Loop & Model Fine-Tuning (MoA / DPO Pipeline)**
   - **Scope**: Mixture-of-Agents router (Mitigation #70) routing on `complexity_score`, Novelty & Entropy filter (Mitigation #68), automated DPO preference dataset builder (Mitigation #69), and benchmark promotion gate for model weights.
   - **Proof Milestone**: Ingest 1,000 reasoning episodes $\to$ filter keeps $\ge 500$ verified, high-novelty DPO pairs $\to$ MoA router cleanly dispatches between base and fine-tuned adapters.

### Known Architectural Boundaries
- **Knowledge Depth**: The agent retrieves structured knowledge (3 types: concept, syntax, troubleshooting; context passages) and generates closed-world grounded answers. Complex multi-step reasoning chains remain bounded by the LLM brain's inherent reasoning capabilities — the agent provides the facts, the brain reasons over them.
- **Knowledge Source Boundary**: Broad general knowledge comes from the configured Brain (Gemini/Claude/OpenAI pretraining), not from memory. The memory tiers *augment* that with persistent, private, sourced facts, verified skills, and project state. MockBrain or a small local model provides only what was explicitly ingested — no latent broad knowledge (Section 1).
- **Sandboxed Testing**: Tool validation uses `unittest.mock` to verify command construction, flags, and parameter logic safely without executing real destructive system calls. Skills proven only this way are labeled `verification_tier = mock` and are **not** claimed to be real-world verified (Mitigation #35). Skills that pass a real dry-run inside the gVisor sealed envelope (Mitigation #38) are labeled `real_local` (offline CLI/engine) or `real_external` (allowlisted API or schema-faithful fixture; Mitigation #51). All sandbox output is validated by the host-side typed result relayer (Mitigation #49) — self-reported exit codes are not trusted.
- **Sandbox Isolation — Platform Boundary**: The full gVisor sealed execution envelope (Mitigation #38) with userspace kernel, read-only squashfs rootfs, and hardened gofer provides production-grade isolation on **Linux only**. On Windows, the sandbox falls back to Job Objects with restricted tokens, which cannot enforce filesystem mount options (`noexec`, `nosuid`) or network namespaces; full isolation on Windows requires Docker Desktop or WSL2. This is a platform-level constraint, not an architectural gap.
- **WebAssembly Deferral (V3)**: Wasm (Wasmtime/WAMR) provides hardware-level linear memory isolation and native control-flow integrity, structurally superior to gVisor's syscall interception. However, no mature Python→Wasm compilation toolchain exists, and running CPython inside Wasm still exposes the Python VM as the attack surface. Wasm is deferred to V3 when the skill IR is mature enough to emit non-Python output (Mitigation #38).
- **Sandbox Module Universe**: Skills can only use Python modules that are pre-installed in one of the four sandbox images (`sandbox-stdlib`, `sandbox-web`, `sandbox-scientific`, `sandbox-full`; Mitigation #57). Modules outside the `sandbox-full` image (e.g., `tensorflow`, `torch`, `opencv`) are rejected at compile time. Operators can extend the universe by building custom images and registering them in `config.py`, but the sandbox never runs `pip install` at runtime.
- **Threshold Calibration**: The default gates (0.65 / 0.80) are embedding-model-dependent starting points, not universal constants. `calibrate-thresholds` refuses to run until `calibration/queries.json` meets minimum sizes (50 true_hit / 30 related_unknown / 20 opposing / 30 unrelated; Mitigation #53), then writes model-keyed gates.
- **Web Coverage**: Ingestion is best-effort. Sites that change HTML, block all fetch providers, or require uninstalled headless browsers will abort their unit (Mitigation #4/#39). The system will report the coverage gap honestly rather than fill it with guesses, but coverage will be uneven for hostile or JS-only docs.
- **Runtime Coverage**: V1 skill synthesis is Python-only. C#, GDScript, C++, and shader runtimes require the target engine toolchain and are gated behind `--runtime` as experimental (Mitigation #36).
- **Project Memory Coverage**: Tier 4 indexes only what the agent has seen or been told to index. Unvisited code, private submodules, and files excluded by `.gitignore`-style rules are unknown until `project index` or an agent read/write covers them (Mitigation #50).
- **Autonomous Proactivity Bounds**: The Heartbeat Daemon operates under strict bounds (max 3 actions/hour, pause on active user input). It does not execute unbounded live system operations without sandbox isolation (Mitigation #38/#41/#46). Autonomous goals stay at depth 2; deeper trees require explicit user supervision and per-expansion approval (Mitigation #54).
- **Embedding Model Limitation**: Dense embeddings are trained on natural language, not code syntax. Hybrid search (Mitigation #30) compensates with exact keyword matching, but highly technical queries may still need the `correct` command for fine-tuning.

---

## 9.1 Autonomous Deep Research & Scientific Knowledge Ingestion Specification

### The Problem of Unbounded Learning Prompts
A prompt like *"Learn everything about AI on the web"* is mathematically unbounded. When dispatched to an autonomous agent without programmatic boundaries, it fails for three structural reasons:
1. **Infinite Crawl Graph & Budget Exhaustion**: The web contains petabytes of interconnected material. Without stopping criteria, autonomous recursion crawls indefinitely until disk space, memory, or LLM token budgets are depleted.
2. **Vector Space Pollution (Retrieval Drift)**: Ingesting thousands of generic, broad web articles into Tier 2 Semantic Memory (`semantic.db`) floods the dense vector index with low-entropy noise. This degrades cosine similarity precision for specific technical queries.
3. **Synthesis & Context Compression Bottlenecks**: Compressing massive multi-document web crawls forces aggressive summarization, collapsing deep mathematical models or precise API signatures into superficial trivia.

### Prompt Scoping & Bounded Execution Matrix

| Scoping Tier | Prompt Example | Planner Behavior | System Outcome |
| :--- | :--- | :--- | :--- |
| **Unbounded / Too Broad** | *"Learn everything about AI on the web."* | Creates vague DAG nodes (*"History of AI"*, *"Overview of Vision"*); scrapes generic blog summaries. | Shallow summaries, cluttered vector store, low engineering utility. |
| **Bounded Topic** | *"Research current speculative decoding algorithms and save a comparative breakdown to `speculative_decoding.md`."* | Decomposes into 3–4 focused DAG queries, targets official documentation / papers, distills atomic facts. | High-signal semantic memory and a verified, actionable markdown reference. |
| **Targeted Deep Dive** | *"Research Unity 6 Input System migration patterns and write a sample C# player controller."* | Gathers exact API syntax, creates functional code, verifies output, and indexes ASTs. | Executable code files and immediately queryable project facts. |

### Academic & Scientific Repositories vs. Open Web Scraping
For deep technical and mathematical learning, the engine prioritizes **academic and scientific repositories** (arXiv, Semantic Scholar, CrossRef, PubMed, Google Scholar) over general web crawling:
- **High Signal-to-Noise Ratio**: Academic papers eliminate search-engine optimization (SEO) spam, marketing copy, and unverified blog posts, providing verified benchmarks, formal proofs, and exact mathematical formulations.
- **Structured Layout Ingestion**: Papers adhere to standardized sections (*Abstract*, *Methodology*, *Results*, *Discussion*). The planner can ingest only the abstract first (Tier 0) to evaluate relevance before committing to full-text PDF parsing.
- **Structured JSON APIs & DOIs**: Direct integration with arXiv API, Semantic Scholar Graph API, and CrossRef removes fragile HTML DOM scraping and bypasses anti-bot barriers.

### Architectural Research Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ User Directive:                                                                         │
│ task "Research 2024-2026 arXiv papers on Mamba vs Transformers and save to mamba.md"    │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. TaskPlanner (DAG Decomposition with Hard Depth & Breadth Caps)                       │
│    - Cap Breadth: max 4 DAG sub-goals                                                   │
│    - Cap Sources: top 3 papers per query                                                │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Academic Ingestion (Tier 0 / Tier 1 MCP or Tool)                                    │
│    - arXiv / Semantic Scholar API query -> Title, Abstract, PDF URL, Citations          │
│    - Lightweight PDF Extraction (`pypdf` / `pymupdf`) -> Extract sections              │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Cognitive Distillation & Synthesis (Tier 0 Reasoning)                                │
│    - Extract atomic facts -> Tier 2 Semantic Memory (dedup cosine >0.95)                │
│    - Synthesize comparative technical markdown report                                   │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Artifact Persistence & Project Memory Indexing (Tier 2 Action)                       │
│    - HITL Governor approval: [GOVERNOR] Approve writing `mamba_research.md`? [Y/n]      │
│    - Atomic file write to workspace root                                                │
│    - Immediate `project_memory.upsert_file()` -> file is instantly queryable by `ask`   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9.2 Engineering Execution Estimates & Core Pitfalls

### Remaining Implementation Effort
- **Total Estimated Remaining Effort**: **16 to 24 active coding hours** (roughly 1 to 2 weeks at a steady pace) across Phases 4 through 6.

### Core Architectural Pitfalls to Watch Out For
1. **Token Bloat in Web/PDF Scraping**:
   - *Risk*: Ingesting entire 20-page arXiv PDFs or deep documentation trees directly into LLM prompts quickly blows through context windows and degrades throughput on local models.
   - *Architecture Invariant*: PDF and web parsers must chunk documents into structured sections (*Abstract*, *Key Findings*, *Methodology*), generate embeddings for vector retrieval, and pass only dense, high-signal passages to the LLM brain.
2. **Sandbox Timeout & Security Constraints**:
   - *Risk*: Dynamic execution of synthesized Python code (Tier 1) can hang on infinite loops or attempt unauthorized host access.
   - *Architecture Invariant*: Subprocess runners must strictly enforce execution timeouts (e.g., max 5–10 seconds) and restrict subprocess filesystem and network permissions via AST allowlists and sandboxed envelopes.
3. **Prompt Drift in Smaller / Local Models**:
   - *Risk*: Local 7B/8B models (e.g., Llama 3.2, Qwen 2.5) tend to lose prompt adherence or hallucinate tool schemas during extended multi-turn tool chains.
   - *Architecture Invariant*: Tool definitions, Pydantic schemas, and system prompts must be kept compact, modular, and concise with deterministic fallback overrides (e.g., `_enforce_file_tiers`).


## 10. Directory & File Structure

```
e:\AI double/
├── ARCHITECTURE.md           # This persistent system architecture specification
├── README.md                 # Project overview and quick start guide
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project packaging & test configuration
├── .env.example              # Environment variables template (one API key var per provider, e.g., OPENCODEZEN_API_KEY, GOOGLE_API_KEY)
├── brains.json               # User-defined brain provider registry (Mitigation #56) — hand-edit to add your API provider
├── brains.example.json       # Documented example registry (openai_compatible / google_genai / anthropic / mock)
├── Dockerfile.sandbox-stdlib      # Build: Python 3.12 + stdlib only (~80MB read-only squashfs)
├── Dockerfile.sandbox-web         # Build: stdlib + requests/httpx/bs4/lxml (~150MB)
├── Dockerfile.sandbox-scientific  # Build: stdlib + numpy/scipy/pandas/sklearn (~650MB)
├── Dockerfile.sandbox-full        # Build: stdlib + web + scientific (~1.2GB)
├── calibration/              # Labeled eval set for threshold calibration
│   └── queries.json          # true_hit / related_unknown / opposing / unrelated items
├── seed_data/                # Foundational knowledge (ships with project)
│   ├── facts.json            # 50-100 curated structured facts (3 types)
│   └── skills/               # 8 pre-written, tested Python tools
│       ├── stdlib_read_file.py
│       ├── stdlib_write_file.py
│       ├── stdlib_list_directory.py
│       ├── stdlib_system_info.py
│       ├── stdlib_git_status.py
│       ├── stdlib_http_get.py
│       ├── stdlib_json_parse.py
│       └── stdlib_run_command.py
├── data/                     # Auto-created on startup
│   ├── episodic.db           # Tier 1: Episodic memory with trace_id & strategy telemetry (WAL mode; M67)
│   ├── semantic.db           # Tier 2: Semantic memory (2 tables: facts, passages)
│   ├── reasoning.db          # Tier 2.5: Reasoning Memory (reasoning_episodes, SHyAOEDRGL tuples; M61)
│   ├── skills.db             # Tier 3: Skill registry (WAL mode)
│   ├── projects.db           # Tier 4: Project Memory (projects, project_files, project_decisions) (M50)
│   ├── goals.db              # Directed Acyclic Goal Graph (Mitigation #42, #47, #54)
│   ├── active_task.json      # Deterministic Task FSM state with strategy_label & prompt_hash (M49, M67)
│   ├── self_model.json       # Persistent Self-Model with reasoning_profile (global_scores, domain_deltas, strategy_index, zpd_ceilings; M40, #62, #66)
│   ├── state_manifest.json   # Startup tamper-detection hashes for self_model.json & projects.db (M52)
│   ├── sandbox_manifest.json # SHA-256 hashes of sandbox image squashfs layers (M59)
│   ├── trusted_docs.json     # User-extensible official docs registry (optional)
│   ├── calibration.json      # Calibrated thresholds per embedding model (Mitigation #37, #53)
│   ├── ingest_cache/         # HTTP response cache for resilient ingestion (Mitigation #39)
│   └── models/               # Cached embedding models (FastEmbed ONNX)
├── agent/
│   ├── __init__.py
│   ├── config.py             # Settings, paths, thresholds, SEED_VERSION, NETWORK_ALLOWLIST, brain provider resolution (brains.json), MoA settings, dir auto-creation
│   ├── models.py             # Pydantic schemas (Fact, Passage, Skill, Project, ProjectFile, ProjectDecision, Goal, TaskState, SelfModel, EpisodicLog, ReasoningEpisode, SRT, Plan)
│   ├── cli.py                # Rich REPL (learn, ask, correct, forget, projects, project, rate, self-model, goals, reflect, heartbeat, task)
│   ├── main.py               # Entrypoint (--brain, --demo, --reseed, --stage, --permission-tier, --no-daemon)
│   ├── brains/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract BaseBrain interface
│   │   ├── mock_brain.py     # Deterministic offline brain (known-topics declaration)
│   │   ├── gemini_brain.py   # Google GenAI brain (3-retry, JSON extraction)
│   │   ├── claude_brain.py   # Claude/OpenAI LiteLLM brain (3-retry, JSON extraction)
│   │   ├── moa_router.py     # Mixture-of-Agents Router: complexity_score routing to cloud API vs LoRA adapter (M70)
│   │   └── factory.py        # Brain loader: reads brains.json registry, resolves provider, MockBrain fallback (M56)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── embeddings.py     # L2-normalized vectors + FTS5 sparse keyword index
│   │   ├── episodic.py       # Tier 1: Interaction logs, audit trails, trace_id, strategy telemetry, 90-day retention (M67)
│   │   ├── semantic.py       # Tier 2: Facts + passages, hybrid search, corrections
│   │   ├── reasoning.py      # Tier 2.5: Reasoning Memory manager — sole writer of reasoning.db (M61)
│   │   ├── procedural.py     # Tier 3: Skill registry, topic-prefixed, UNIQUE constraint
│   │   ├── project.py        # Tier 4: Project Memory manager — sole writer of projects.db (M50, #52)
│   │   ├── goals.py          # Directed Acyclic Goal Graph manager (Mitigation #42, #47, #54)
│   │   ├── self_model.py     # Persistent Self-Model & competence tracker with reasoning_profile (write-protected; M40, #62)
│   │   └── seeder.py         # Foundational boot seeder (computes embeddings at runtime)
│   └── engine/
│       ├── __init__.py
│       ├── planner.py        # Curriculum planner (scope boundary, disambiguation, novelty_score & complexity_score computation; M62, M67, M70)
│       ├── strategy_injector.py # Injects prompt templates from strategy_index, writes strategy_label & prompt_hash (M62, M67)
│       ├── ingest.py         # Trusted Docs Registry → DDG Domain Scoring → fetch chain + abort guard
│       ├── synthesizer.py    # Structured distillation (3 types), passages, skills
│       ├── validator.py      # Skill compiler (positive-match), gVisor sealed executor, typed result relayer, image resolver, regression suite
│       ├── critic.py         # Lateral Critic / Adversarial Verifier (two-solver parallel dispatch + divergence arbiter; M63)
│       ├── verifier.py       # Symbolic Verifier: parses Structured Reasoning Traces (SRTs), validates logical entailment with Z3/Prolog (M65)
│       ├── retriever.py      # Hybrid search (semantic + project) + closed-world grounding
│       ├── heartbeat.py      # Autonomous background daemon: counterfactual reflection, curriculum replay (M41, #46, #64)
│       ├── reflection.py     # Metacognitive reflection, consolidation, weekly domain_delta SQL aggregation job (M43, #55, #62, #67)
│       ├── governor.py       # HITL permission tiers, circuit breakers & tamper watchdog (Mitigation #46, #52)
│       ├── state_machine.py  # Deterministic Task FSM & trace telemetry with strategy_label (Mitigation #49, M67)
│       ├── benchmark.py      # Objective benchmark test runner & ZPD binary search difficulty evaluator (M45, M66)
│       ├── dataset_builder.py # Novelty (>0.7) & entropy (>2σ) filter, DPO/SFT dataset generator for model fine-tuning (M68, M69)
│       └── orchestrator.py   # Master coordinator (learn, correct, forget, refresh, reflect, task)
├── skills/                   # Verified executable Python tools (topic-prefixed .py files)
│   └── __init__.py
└── tests/
    ├── benchmark_suite/      # Immutable developer-written integration benchmark tests (Mitigation #45)
    │   ├── test_bench_git.py
    │   ├── test_bench_python.py
    │   └── test_bench_cli.py
    ├── reasoning_suite/      # Immutable reasoning benchmark suite with parameterized difficulty knobs (M66)
    │   ├── test_reason_decomposition.py
    │   ├── test_reason_hypothesis.py
    │   ├── test_reason_causal.py
    │   ├── test_reason_counterexample.py
    │   ├── test_reason_planning.py
    │   └── test_reason_adversarial.py
    ├── test_memory.py        # Tests for 4-tier memory, FTS5, corrections, user_corrected authority
    ├── test_reasoning.py     # Tests for Tier 2.5 reasoning.db, SRT symbolic verifier, lateral critic, domain_deltas (M61-M65)
    ├── test_brains.py        # Tests for MockBrain known-topics, BrainFactory brains.json resolution, MoA router
    ├── test_ingest.py        # Tests for Trusted Docs, domain scoring, fetch chain + abort guard
    ├── test_validator.py     # Tests for positive-match compiler, tiered import resolution, sandbox image selection, tier assignment
    ├── test_sandbox.py       # Tests for gVisor sealed envelope, resource limits, I/O protocol, rootfs immutability (M38, #57-#60)
    ├── test_retriever.py     # Tests for hybrid search (semantic + project), closed-world grounding, thresholds
    ├── test_seeder.py        # Tests for seed loading, versioning, idempotency, source_type
    ├── test_project.py       # Tests for Tier 4 indexing, hash-diff updates, decisions, write-protection (M50, #52)
    ├── test_self_model.py    # Tests for self-model persistence, reasoning_profile, competence recalculation, tamper rollback
    ├── test_goals.py         # Tests for DAG prerequisite trees, supervised depth approvals (M47, #54)
    ├── test_heartbeat.py     # Tests for perceive-evaluate-plan-act idle cycle, counterfactual generation, no-op cadence
    ├── test_reflection.py    # Tests for consolidation, correction audit, weekly domain_delta aggregation (M55, M62, M67)
    ├── test_governor.py      # Tests for permission tiers, circuit breakers, tamper watchdog
    ├── test_state_machine.py # Tests for FSM transitions, strategy_label logging, and trace_id logging
    ├── test_benchmark.py     # Tests for immutable benchmark harness and ZPD binary search calibration
    └── test_end_to_end.py    # Full 7-phase: Seed → Query → Learn → Correct → Forget → Heartbeat → Project
```
