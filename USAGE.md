
Status: Pre-alpha
Core validation: partial
Experimental features: not production-ready

# Usage Guide

> Practical guide for first-time users of Solaris Zarya Engine.

---

## Getting Started

After installation, start the interactive REPL:

```bash
python -m agent.main
```

On first boot, the engine automatically seeds foundational facts into memory. You'll see:

```
First boot: seeded N facts.
Agent REPL — Phase 3 (Brain: GeminiBrain)
```

You can also pass flags:

| Flag | Effect |
|:---|:---|
| `--demo` | Runs 3 scripted questions and exits (good for testing) |
| `--reseed` | Force-reloads seed facts from `seed_data/facts.json` |
| `--no-daemon` | Disables the Heartbeat background daemon |

---

## How the System Works

### Memory Tiers

Solaris Zarya stores everything persistently in SQLite databases under `data/`:

| Tier | Database | What it stores |
|:---|:---|:---|
| 1 - Episodic | `episodic.db` | Every interaction log, audit trail, task execution trace (90-day TTL) |
| 2 - Semantic | `semantic.db` | Atomic facts, passages, and embeddings |
| 2.5 - Reasoning | `reasoning.db` | Structured reasoning episodes (permanent) |
| 3 - Procedural | `skills.db` + `skills/` | Synthesized Python tools and their metadata |
| 4 - Projects | `projects.db` | Indexed files from your workspace (hashes, summaries, embeddings) |
| Goals | `goals.db` | Hierarchical goal DAGs and prerequisites |

### Confidence Gating

When you ask a question, the retriever scores how relevant stored knowledge is to your query:

- **Score >= 0.80** (Confident): Returns a grounded answer from memory.
- **Score 0.65 - 0.79** (Tentative): Passes context to the LLM with a caution warning.
- **Score < 0.65** (Refused): Honestly says *"I haven't learned about that yet."*

This prevents hallucinated answers on topics the engine hasn't been taught.

### Permission Tiers

The engine classifies all actions into safety tiers:

| Tier | What it covers | Approval |
|:---|:---|:---|
| Tier 0 | Read, search, reason | Auto-approved |
| Tier 1 | Sandboxed skill execution | Auto-approved (requires Docker) |
| Tier 2 | File writes, skill synthesis, system changes | Requires `[Y/n]` approval |

**Tier 1 caveat:** Sandboxed execution requires Docker to be installed and running. If Docker is unavailable, the engine refuses execution by default. An explicit `--unsafe-host` flag can override this for local (unisolated) execution, but this bypasses the security sandbox.

**Tier 2 has two approval points for multi-step tasks:**

1. **Plan-level approval** -- When you run `task <description>`, the engine shows the full Goal DAG and asks `"Approve plan and begin execution? [Y/n]:"`. This approves the plan, not individual actions.
2. **Action-level approval** -- Before each Tier 2 action (e.g., writing a file), you will see another `[y/N]` prompt describing the specific action. Plan approval alone does not cover subsequent file writes.

---

## Commands Reference

### Asking Questions

```
ask How do I check my git status?
```

Searches semantic memory and project files. Returns an answer grounded in stored facts, or refuses honestly if nothing relevant is found.

### Learning New Topics

```
learn docker networking
```

Autonomously researches the topic: breaks it into curriculum units via the LLM, searches the web (DuckDuckGo), fetches and extracts content from results (Jina Reader + BeautifulSoup), distills atomic facts and passages, and stores them in semantic memory. Also exports a markdown note to `data/knowledge/`. Costs API calls (not free with real brains).

```
learn resume
```

Resumes a previously interrupted learning session from its checkpoint (`data/active_curriculum.json`).

### Synthesizing Skills

```
skill calculate fibonacci number
```

Generates a Python tool, validates it through AST security checks (blocks `os.system`, `eval`, `exec`, `subprocess`, `socket`, `ctypes`), runs unit tests in a sandbox, and registers it in procedural memory. The skill file is saved to `skills/`. The newly created skill file is automatically indexed into Project Memory.

### Listing and Running Skills

```
skills
```

Shows all registered skills with their verification tier and file path.

```
skills docker
```

Filters skills by keyword.

```
run-skill calculate_fibonacci_number '{"n": 10}'
```

Executes a registered skill in a secure subprocess. Pass arguments as a JSON string.

### Correcting Facts

```
correct 5 The default port is 8080, not 8000
```

Overwrites a specific fact by ID. Corrected facts get top authority (confidence: 1.0) and won't be overwritten by automatic re-ingestion.

```
correct Unity "old fact text" -> "new fact text"
```

Finds and replaces a fact by topic and text match.

### Viewing Stored Knowledge

```
facts
```

Lists all facts in semantic memory.

```
facts docker
```

Filters facts by topic or text keyword.

```
stats
```

Shows record counts across all memory tiers.

### Project Indexing

```
project index .
```

Scans the current directory, computes SHA-256 hashes, generates summaries, and embeds files into Project Memory. Only changed files are re-processed on subsequent runs.

```
project index /path/to/your/project
```

Index a specific path.

```
project list
```

Shows all indexed project files.

**Auto-indexing:** Files created by the agent (via `task` or `skill`) are automatically indexed into Project Memory immediately after being written. You do not need to run `project index .` again to search them.

### Multi-Step Tasks

```
task Research Docker CLI, extract facts, and synthesize a container inspect tool
```

Plans a Goal DAG, shows you the plan for approval, then executes through the crash-resilient Task FSM. Each step is executed one atomic action at a time, with state persisted to disk after every transition.

### Brain Management

```
brain list
```

Shows available providers and which one is currently active.

```
brain switch groq llama-3.3-70b-versatile
```

Hot-swaps the active brain at runtime. No restart needed.

```
brain switch mock
```

Switches to MockBrain (free, deterministic, limited knowledge).

```
brain switch moa_router
```

Switches to the Mixture-of-Agents Router. Routes tasks dynamically based on complexity: routine tasks go to the base brain, high-complexity reasoning tasks are routed to a LoRA reasoning adapter when one is available. Hot-reloads trained LoRA checkpoints from `data/checkpoints/`.

**Warning:** If the requested provider name is unrecognized or construction fails, the engine silently falls back to MockBrain. After switching, always verify with `brain list` to confirm the expected provider is active. Invalid or unavailable model names may also cause errors at query time without an obvious fallback message.

### Reading Files

```
read agent/config.py
```

Displays a file with syntax highlighting. Markdown files render with formatting.

### Chat

```
chat Tell me about yourself
```

Routes input to the conversational Chat Engine (maintains 10-turn context).

```
/clear
```

Clears the conversational context.

Any unrecognized text is automatically routed to the Chat Engine as conversation.

### DPO Dataset Pipeline

```
dataset stats
```

Shows DPO dataset statistics (pair count, file size).

```
dataset build --dry-run
```

Previews candidates that would be harvested from episodic memory without writing anything.

```
dataset build --limit 50
```

Harvests up to 50 DPO pairs from verified reasoning episodes. The pipeline automatically extracts failure-to-self-repair trajectories from episodic memory and filters for novelty and solution-path entropy.

```
dataset clear
```

Deletes the DPO dataset file after confirmation.

### Game Engine & Asset Tools

The engine integrates with Unity (via the official Unity CLI / `com.unity.pipeline` with legacy batchmode fallback) and Blender for game development workflows.

```
unity-synth "Create a PlayerHealth component with TakeDamage and Heal methods"
```

Autonomously synthesizes a Unity C# script inside an isolated Git staging worktree (`data/sandboxes/worktrees/`). Validates it against security rules (blocking banned namespaces and auto-executing attributes), runs tests headlessly via the Unity CLI (or batchmode), iteratively fixes compile/test errors, and pauses at the Governor Tier 2 gate (`[Y/n]`) before merging into your main workspace.

```
blender-synth "Create a low-poly tree model and export as fbx"
```

Synthesizes a Blender Python (`bpy`) automation script with AST and path-sandboxed file writes. Executes headlessly to generate 3D assets into `data/exports/` and requests Tier 2 approval before staging into Unity `Assets/`.

### Multi-Step Tasks

```
task plan Research Docker CLI, extract facts, and synthesize a container inspect tool
```

Decomposes a complex objective into a Directed Acyclic Goal Graph (DAG) with prerequisites and safety tier annotations.

```
task run
```

Executes the active planned task DAG step by step through the crash-resilient Task FSM. Each transition persists state to disk for atomic rollback or crash resumption.

### Fine-Tuning & Model Promotion

```
train list
```

Shows available LoRA checkpoints with status and source dataset.

```
train dpo --epochs 3 --batch-size 4
```

Runs real 4-bit NF4 QLoRA DPO training on CUDA. Automatically falls back to fp16 on Windows if bfloat16 is unsupported.

```
train promote <checkpoint_dir_name>
```

Marks an evaluated, benchmark-passing LoRA checkpoint as promoted, allowing `MoABrain` (`moa_router`) to immediately hot-swap it for complex reasoning tasks.

```
train dpo --dry-run
```

Previews the training run without downloading models or using GPU.

### Metacognitive Tools

```
self-model
```

Shows the empirical competence matrix, reasoning profile scores, known strengths, and knowledge gaps.

```
benchmark reasoning
```

Runs ZPD binary search calibration across 6 reasoning categories to discover difficulty ceilings. Real calibration requires a real brain; MockBrain results are deterministic but may not reflect true capability.

### Experimental Commands

```
ingest-paper 2301.07041
```

Ingests an arXiv paper by ID, URL, or search query. Extracts facts and stores them in semantic memory. Requires optional dependencies (PyMuPDF). May not work on all platforms.

---

## Tips

### 1. Start with MockBrain

If you don't have API keys yet, set `AI_BRAIN=mock` in your `.env`. MockBrain is free and deterministic -- good for exploring the system without spending tokens.

### 2. Seed Before Asking

Run `learn <topic>` before asking questions about it. The engine only knows what it has explicitly ingested (plus whatever the underlying LLM brain knows broadly).

### 3. Index Your Project Early

Run `project index .` so the engine can ground answers in your actual codebase. Without this, `ask` only searches semantic facts.

### 4. Use `facts` to Audit Memory

Run `facts` periodically to see what the engine actually knows. This is the ground truth for retrieval.

### 5. Correct Wrong Facts Immediately

If you spot an incorrect answer, use `correct` right away. User-corrected facts have top authority and won't be overwritten.

### 6. Watch for Confidence Warnings

When an answer shows a "Tentative" confidence warning, the engine is telling you it's not fully certain. Verify independently before relying on it.

### 7. Approve Tier 2 Actions Carefully

The `[Y/n]` prompt exists for a reason. Review what the engine wants to write before approving. Remember: plan approval does not cover individual file writes -- you will be prompted again before each one.

### 8. Crash Recovery Requires Your Input

If the process is killed mid-task, restart and the engine will detect the unfinished task. If an action may or may not have completed, you will see:

```
[FSM] Ambiguous resume detected.
Action may or may not have completed.

Options:
  r  re-run this action
  s  skip it and continue
  f  fail this step and stop
  a  abort task
```

Choose `r` to re-run, `s` to skip, `f` to fail the step, or `a` to abort. The engine will **not** automatically re-run or skip -- it stops and asks you.

### 9. Reseed When Schema Changes

If you update the engine and the memory schema changes, run `python -m agent.main --reseed` to reload foundational facts.

### 10. Verify Brain After Switching

After `brain switch`, always run `brain list` to confirm the expected provider is active. Unknown providers silently fall back to MockBrain.

---

## Troubleshooting

| Problem | Solution |
|:---|:---|
| `I haven't learned about that yet` | Run `learn <topic>` first, or `project index .` for codebase questions |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your virtualenv |
| API rate limit errors | The engine retries 3x with backoff. If persistent, switch to a different brain or use MockBrain |
| Task crash loop at boot | The engine will offer to resume. Use the `[Y/n]` prompt to resume or clear. If resume fails repeatedly, inspect `data/active_task.json` manually before deleting |
| Embedding dimension mismatch | Run `python -m agent.main --reseed` or delete `data/semantic.db` and reseed |
| Skills directory is empty | Synthesize skills with `skill <topic>` -- they are generated on demand |
| Skill execution refused | Docker may not be running. Tier 1 sandboxed execution requires Docker. Check `docker info` |
| Brain switch fell back to MockBrain | The provider name may be unrecognized. Run `brain list` and use an exact provider name |
