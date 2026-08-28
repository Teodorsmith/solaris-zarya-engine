# Solaris Zarya Engine

> **Autonomous AI Double with 4-Tier Persistent SQLite Memory, AST-Sandboxed Skill Synthesis, Crash-Resumable Task FSM, and Deterministic Permission Governance.**

---

## Overview

Solaris Zarya Engine is an autonomous developer agent built from first principles with a deterministic engineering spine. Rather than relying on unstructured conversational loops or unverified tool execution, the engine enforces strict memory hierarchy, confidence-gated retrieval, host-side AST security validation, crash-resilience, and human-in-the-loop (HITL) permission tiers.

```
                                  ┌───────────────────────────────┐
                                  │      User / CLI Interface     │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Brain Manager & Providers   │
                                  │ (Gemini / Groq / OpenAI /     │
                                  │  Local Ollama / MockBrain)    │
                                  └───────┬───────────────┬───────┘
                                          │               │
                     ┌────────────────────┴───┐       ┌───┴───────────────────┐
                     ▼                        ▼       ▼                       ▼
      ┌─────────────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐
      │  Confidence-Gated Retriever │  │  Goal DAG Planner   │  │   Skill Synthesizer         │
      │  - Semantic Facts (0.65/0.8)│  │  - Depth-Capped DAG │  │   - AST Allowlist Sandbox   │
      │  - Project Files (Projects) │  │  - File Tier Override│  │   - 2-Retry Stderr Repair  │
      │  - Closed-World Refusal     │  │  - Prereq Resolution│  │   - Host-Side Typed Output  │
      └──────────────┬──────────────┘  └──────────┬──────────┘  └──────────────┬──────────────┘
                     │                            │                            │
                     ▼                            ▼                            ▼
      ┌───────────────────────────────────────────────────────────────────────────────────────┐
      │                                Permission Governor                                    │
      │              Enforces Tier 0 / 1 / 2, Depth Caps, and Mutation Logging                 │
      └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                                  │
                                                  ▼
      ┌───────────────────────────────────────────────────────────────────────────────────────┐
      │                      Crash-Resilient Task FSM (active_task.json)                      │
      │           Atomic State Transitions: PENDING -> RUNNING -> COMMITTED -> COMPLETED       │
      └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                                  │
                                                  ▼
      ┌───────────────────────────────────────────────────────────────────────────────────────┐
      │                             4-Tier Persistent SQLite Memory                           │
      │  [Tier 1: episodic.db]  [Tier 2: semantic.db]  [Tier 3: procedural.db]  [Tier 4: projects.db]│
      └───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities (Phases 0–5 Implemented & Phase 6 Core)

### 1. 4-Tier Persistent SQLite Memory (WAL Mode)
- **Tier 1 — Episodic Memory (`data/episodic.db`)**: 90-day retention log recording questions, answers, refusals, task executions, and governor approvals/denials with audit traces.
- **Tier 2 — Semantic Memory (`data/semantic.db`)**: Atomic fact storage with deduplication (>0.95 similarity gate) and contradiction checks.
- **Tier 3 — Procedural Memory (`data/procedural.db`)**: Registry of synthesized Python skills synced with real, human-readable `.py` files in `skills/`.
- **Tier 4 — Project Memory (`data/projects.db`)**: Incremental workspace indexer scanning files, computing SHA-256 hashes, generating role summaries, and embedding files for cross-tier retrieval.
- **Goal Memory (`data/goals.db`)**: Persistent store for hierarchical Goal Directed Acyclic Graphs (DAGs) and prerequisite tracking.

### 2. FastEmbed ONNX & Calibrated Confidence Gating
- Local **BAAI/bge-small-en-v1.5** 384-dimensional ONNX embeddings (no GPU or PyTorch required), with deterministic offline fallback for CI/offline testing.
- **Two-Threshold Confidence Gate**:
  - $\ge 0.80$: **Confident** — answers with high certainty.
  - $0.65 \le \text{score} < 0.80$: **Tentative** — passes context to the LLM with caution warnings.
  - $< 0.65$: **Refused** — honest refusal (*"I haven't learned about that yet..."*), eliminating hallucinated answers on missing knowledge.

### 3. AST Allowlist & Sandboxed Skill Synthesis
- Synthesizes pure Python skills with dedicated unit test suites.
- Host-side AST security validator blocks dangerous primitives (`os.system`, `subprocess`, `eval`, `exec`, `__builtins__`, `socket`, `ctypes`).
- Subprocess execution harness with 5-second hard timeouts and captured stderr feedback.
- Automated 2-retry revision loop to repair syntax and runtime errors before registration.

### 4. Crash-Resilient Task FSM & Goal DAG Planner
- Decomposes complex user instructions into a Directed Acyclic Graph (DAG) of sub-goals with prerequisite resolution and depth caps.
- Atomic state persistence via `data/active_task.json` and `data/state_manifest.json`.
- Automatic resume recovery: if interrupted or killed mid-task, restarts from disk state without repeating completed steps.
- Deterministic tier override: file-write operations are automatically upgraded to Tier 2 regardless of LLM classification.

### 5. Domain Validation & Game Engine Synthesis (Phase 5)
- **Official Unity CLI (`com.unity.pipeline`)**: Structured JSON output for headless testing (`run_tests`), live C# evaluation (`eval_csharp`), recompilation, and status queries. Includes legacy `-batchmode` fallback for older Unity versions.
- **Blender MCP Client**: Headless 3D procedural script execution (`bpy`) with AST + runtime path sandboxing restricting all file writes strictly to `data/exports/`.
- **Git Staging Worktree Isolation**: Synthesizes and tests game assets in isolated branches (`data/sandboxes/worktrees/`) before prompting for Tier 2 Governor approval to merge.

### 6. Metacognitive Reasoning, DPO & MoA Router (Phases 4 & 6)
- **Heartbeat Daemon & Self-Model**: Runs autonomous background loops to verify memory, aggregate telemetry, and clean up stale data (max 3 actions/hour).
- **Mixture-of-Agents Router (`moa_router`)**: Dynamically routes tasks between cloud base brains and local LoRA reasoning adapters based on intrinsic complexity.
- **DPO Dataset Builder & QLoRA Fine-Tuning**: Harvests high-novelty, high-entropy reasoning episodes from `episodic.db` and coordinates 4-bit NF4 QLoRA training.

---

## Quickstart

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Teodorsmith/solaris-zarya-engine.git
   cd solaris-zarya-engine
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your API keys:
   ```bash
   cp .env.example .env
   ```
   Supported keys:
   - `GEMINI_API_KEY` (Google Gemini)
   - `GROQ_API_KEY` (Groq)
   - `OPENAI_API_KEY` (OpenAI)
   - `LOCAL_LLM_API_KEY` / `OPENAI_LIKE_API_KEY` (Ollama, LM Studio, vLLM)
   - `AI_BRAIN=mock` (Offline deterministic mode)

---

## CLI & REPL Commands

Start the interactive REPL:
```bash
python -m agent.main
```

### Available Commands

| Command | Usage | Description |
| :--- | :--- | :--- |
| `ask <question>` | `ask Where is configuration handled?` | Query memory with confidence-gated refusal. |
| `learn <topic>` | `learn "Docker networking"` | Autonomous curriculum research & fact extraction. |
| `learn resume` | `learn resume` | Resume interrupted learning checkpoint. |
| `skill <topic>` | `skill calculate fibonacci number` | Synthesize, sandboxed-validate, and persist a Python tool. |
| `skills` | `skills` | List all registered procedural skills. |
| `run-skill <name> [args]` | `run-skill calculate_fibonacci_number '{"n": 10}'` | Execute a registered skill in a secure subprocess. |
| `task plan <goal>` | `task plan Create summary.md with git tips` | Decompose objective into Goal DAG with prerequisites. |
| `task run` | `task run` | Execute active planned task DAG via Task FSM. |
| `project index [path]` | `project index .` | Scan, hash, summarize, and embed workspace files into Tier 4. |
| `project list` | `project list` | List indexed project files and roles. |
| `unity-synth <topic>` | `unity-synth "PlayerHealth component"` | Synthesize & test C# script inside isolated Unity worktree. |
| `blender-synth <topic>` | `blender-synth "Low poly crate"` | Synthesize Python script & export 3D assets in Blender. |
| `ingest-paper <id|url>` | `ingest-paper 2305.16291` | Distill academic paper into semantic memory. |
| `facts` | `facts` | Display stored semantic facts. |
| `stats` | `stats` | Display record counts across all memory tiers. |
| `self-model` | `self-model` | Show competence matrix, boot count, and ZPD profile. |
| `benchmark reasoning` | `benchmark reasoning` | Run ZPD reasoning calibration suite. |
| `dataset stats\|build` | `dataset build --limit 50` | Build DPO pairs from verified reasoning episodes. |
| `train dpo\|promote` | `train dpo --epochs 3` | QLoRA DPO model fine-tuning & checkpoint promotion. |
| `brain switch <provider>` | `brain switch moa_router` | Hot-swap active brain at runtime. |
| `brain list` | `brain list` | List available providers and show active brain. |
| `/clear` | `/clear` | Clear conversational context. |

---

## Running Tests & Benchmarks

Run the complete pytest suite:
```bash
pytest
```

Run retrieval probe benchmarks:
```bash
python probe_bench.py
```

Run the 20-item threshold sweep calibration:
```bash
python threshold_sweep.py
```

---

## Architecture & Roadmap

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep technical specification covering memory schemas, retrieval mechanics, AST allowlists, FSM invariants, and future architectural horizons.
- **[ROADMAP.md](ROADMAP.md)**: Tactical implementation roadmap across Phases 0–6, tracking completed milestones and multi-month engineering horizons.

---

## ⚖️ Licensing & Dual-Licensing

This project is made available under a **Dual-Licensing Model**:

* **Open Source (AGPLv3):** Free for personal, academic, and open-source community use under the terms of the **GNU Affero General Public License v3 (AGPLv3)**.
  * *Note on Section 13:* If you run a modified version of this software on a server and provide access over a computer network (SaaS, API, or web services), you **must** make the corresponding source code of your modifications freely available to all remote network users.
* **Commercial License / AGPL Exemption:** If you intend to incorporate this engine into closed-source commercial applications, proprietary SaaS platforms, or internal proprietary infrastructure without open-sourcing your code under AGPLv3, you must purchase a **Commercial License**.

### 💼 Commercial Licensing & Inquiries
To purchase a commercial license, request custom exemptions, or inquire about enterprise support:
* **Contact:** `teosmith.studios@gmail.com`

---

### 🏛️ Jurisdiction & Governing Law
Any legal claim, dispute, or copyright enforcement proceeding arising out of or in connection with this software or its unauthorized use shall be governed by and construed in accordance with the laws of the **European Union** and the local courts of the copyright holder's jurisdiction.
