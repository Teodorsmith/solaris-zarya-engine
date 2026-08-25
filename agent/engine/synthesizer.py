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

"""Skill and Knowledge Synthesizers.

Generates standalone validated Python skills (with AST security rules)
and extracts structured facts and passages for semantic memory.
Persists synthesized skill episodes into reasoning memory (Mitigation #68).
"""

from __future__ import annotations

import ast
import datetime
from datetime import timezone
import logging
import os
import re
import time
from typing import Any, TYPE_CHECKING
import uuid
from pathlib import Path

from agent.brains.base import BaseBrain
from agent.brains.mock_brain import MockBrain
from agent.engine.governor import PermissionGovernor
from agent.engine.retriever import Retriever
from agent.engine.validator import SecurityError, SkillValidator
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.memory.reasoning import ReasoningMemory
from agent.models import EpisodicLog, Fact, Passage, ReasoningEpisode, Skill

if TYPE_CHECKING:
    from agent.memory.semantic import SemanticMemory
    from agent.engine.dataset_builder import DatasetBuilder
    from agent.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class SynthesizerError(Exception):
    pass


class SkillSynthesizer:
    # Regex patterns for light secret redaction before storing in dataset.
    # Covers: generic key=value secrets, sk-/ghp- tokens, AWS keys,
    # PEM private keys, bearer tokens, and home-dir file paths.
    _SECRET_PATTERNS = [
        # Generic key=value: api_key = "...", SECRET = '...', TOKEN: '...'
        re.compile(
            r'(?i)(api[_-]?key|secret[_-]?key?|token|password|passwd|auth'
            r'|bearer|credential|private[_-]?key)\s*[=:]\s*["\']?[\w\-\.\/\+]{8,}["\']?'
        ),
        # OpenAI-style sk- tokens
        re.compile(r'sk-[A-Za-z0-9]{20,}'),
        # GitHub personal access tokens
        re.compile(r'ghp_[A-Za-z0-9]{36}'),
        # AWS secret/access keys (AKIA... or 40-char base64-ish secret)
        re.compile(r'AKIA[0-9A-Z]{16}'),
        re.compile(r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*[\w\/\+]{40}'),
        # PEM-encoded private key blocks
        re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.DOTALL),
        # Bearer token in HTTP headers
        re.compile(r'(?i)authorization\s*[=:]\s*bearer\s+[\w\-\.]{8,}'),
        # Home-directory paths that expose usernames  (e.g. C:\Users\alice\ or /home/alice/)
        re.compile(r'(?:[Cc]:\\\\Users\\\\|/home/)[A-Za-z0-9_\-\.]+[/\\\\]'),
    ]

    def __init__(
        self,
        brain: BaseBrain,
        retriever: Retriever,
        procedural: ProceduralMemory,
        validator: SkillValidator,
        governor: PermissionGovernor,
        project: ProjectMemory | None = None,
        reasoning_memory: ReasoningMemory | None = None,
        episodic_memory: "EpisodicMemory | None" = None,
        dataset_builder: "DatasetBuilder | None" = None,
    ):
        self.brain = brain
        self.retriever = retriever
        self.procedural = procedural
        self.validator = validator
        self.governor = governor
        self.project = project  # may be None — hook degrades gracefully
        self.episodic_memory = episodic_memory
        self.dataset_builder = dataset_builder
        if reasoning_memory is None:
            from agent.config import REASONING_DB

            try:
                self.reasoning_mem = ReasoningMemory(REASONING_DB)
            except Exception:
                self.reasoning_mem = None
        else:
            self.reasoning_mem = reasoning_memory
        self.max_retries = 6

    @classmethod
    def _redact_secrets(cls, text: str) -> str:
        """Lightly redact obvious secret patterns before storing in dataset."""
        for pattern in cls._SECRET_PATTERNS:
            text = pattern.sub("<REDACTED>", text)
        return text

    def _generate_skill_prompt(
        self, topic: str, context: str, error_feedback: str | None = None
    ) -> str:
        prompt = f"""
You are an expert Python tool developer.
Write a standalone, self-contained Python skill.
Topic: {topic}
Context:
{context}

CRITICAL SECURITY & AST RULES:
1. Built-in `open()` is STRICTLY FORBIDDEN.
   - For reading: `pathlib.Path(path).read_bytes()` or `.read_text(...)`.
   - For writing: `pathlib.Path(path).write_bytes(...)` or `.write_text(...)`.
2. Allowed imports: `pathlib`, `json`, `re`, `math`, `hashlib`, `typing`,
   `dataclasses`, `datetime`, `enum`, `collections`, `functools`, `unittest`.
3. FORBIDDEN patterns (AST validator will REJECT):
   - NO `__class__`, `__dict__`, `__subclasses__`, `__bases__`, `__builtins__`
   - NO `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `eval`
   - NO `sys` module, NO `os` module
4. Use direct attribute access: `obj.attr` not `getattr(obj, "attr")`.
5. Avoid `@classmethod`/`cls` patterns — prefer plain methods and functions.
6. The skill MUST define an `execute(**kwargs)` function as its entrypoint.
   - `execute(**kwargs)` must accept generic keyword arguments and return a
     dictionary envelope `dict[str, Any]` (containing `"result"`).
   - If supporting multi-operation tasks (e.g. set, clear, combine, check),
     accept common aliases (e.g. `operation="set"` and `operation="set_flag"`).
7. PREFERRED STYLE — plain module-level functions over plain ints/strings.
   NO operator overloading (__and__, __or__, __xor__, __invert__), NO
   custom __new__/__init__ tricks, NO Enum subclasses needing them.
   Example of the expected shape:

       def has_flag(flags: int, bit: int) -> bool:
           return (flags & (1 << bit)) != 0

       def execute(flags: int = 0, bit: int = 0, **kwargs) -> dict:
           return {{"result": has_flag(flags, bit)}}

8. OUTPUT FORMAT — return EITHER:
   A) A single valid JSON object with keys 'skill_name', 'description',
      'code', 'test_code' (code values as properly escaped JSON strings), OR
   B) Two markdown blocks: first ```python block = the skill code,
      second ```python block = unittest test cases.
"""
        if error_feedback:
            prompt += (
                f"\nYour previous attempt failed execution testing:\n"
                f"Error:\n{error_feedback}\n\n"
                "Please fix the imports, implementation, and test cases. "
                "Output valid, self-contained Python code.\n"
            )

        return prompt

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove stray markdown fences the model may embed inside JSON."""
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*[ \t]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return cleaned

    @staticmethod
    def _extract_fenced_blocks(text: str) -> list[str]:
        """Extract all ```python / ``` fenced code blocks from raw text."""
        return re.findall(
            r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL
        )

    @staticmethod
    def _slugify_topic(topic: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic).strip("_").lower()
        return slug or "generated_skill"

    @staticmethod
    def _unwrap_double_serialized(code: str) -> str:
        """If the model JSON-encoded the code twice, unwrap one layer."""
        candidate = code.strip()
        if candidate[:1] in {'"', "{", "["} and "\n" not in candidate[:2]:
            try:
                import json as _json

                unwrapped = _json.loads(candidate)
                if isinstance(unwrapped, str):
                    return unwrapped
            except ValueError:
                pass
        return code

    @staticmethod
    def _ast_ok(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _repair_code(self, code: str) -> str:
        """Best-effort repair of known small-model defects."""
        if self._ast_ok(code):
            return code
        cleaned = self._strip_code_fences(code)
        if self._ast_ok(cleaned):
            return cleaned
        stripped = cleaned.strip()
        # Defect: stray '{' prepended before the module docstring/code body.
        while stripped.startswith("{"):
            candidate = stripped[1:].lstrip()
            if not self._ast_ok(candidate):
                break
            stripped = candidate
        return stripped

    def _candidate_payloads(self, response: str, topic: str):
        """Yield (skill_name, description, code, test_code) candidates."""
        parsed = self.brain.extract_json(response)
        if isinstance(parsed, dict):
            code = parsed.get("code")
            test_code = parsed.get("test_code")
            if (
                isinstance(code, str)
                and isinstance(test_code, str)
                and code.strip()
                and test_code.strip()
            ):
                skill_name = (
                    parsed.get("skill_name") or self._slugify_topic(topic)
                )
                description = parsed.get("description") or topic
                yield (
                    str(skill_name).replace(" ", "_"),
                    str(description),
                    self._repair_code(self._unwrap_double_serialized(code)),
                    self._strip_code_fences(
                        self._unwrap_double_serialized(test_code)
                    ),
                )

        blocks = self._extract_fenced_blocks(response)
        if len(blocks) >= 2:
            yield (
                self._slugify_topic(topic),
                topic,
                self._repair_code(blocks[0]),
                blocks[1].strip(),
            )

    def learn_skill(self, topic: str) -> Skill:
        # Retrieve context
        result = self.retriever.retrieve(topic)
        context = ""
        for f in result.facts:
            context += f"- {f.text}\n"

        error_feedback = None
        history: list[dict[str, Any]] = []
        # Track the most recent failed code candidate for DPO harvesting
        latest_rejected_code: str | None = None
        latest_error: str | None = None

        for attempt in range(self.max_retries + 1):
            if error_feedback:
                logger.info(
                    "Pausing for 2.0s before repair attempt..."
                )
                time.sleep(2.0)

            prompt = self._generate_skill_prompt(
                topic, context, error_feedback
            )
            try:
                response = self.brain.generate(
                    prompt, json_mode=True, temperature=0.2
                )
            except TypeError:
                # Brain without json_mode/temperature kwargs support
                response = self.brain.generate(prompt)

            debug_path = Path("data") / "last_synthesis_debug.log"
            try:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(
                    f"=== ATTEMPT {attempt} (json_mode=True) ===\n{prompt}\n\n"
                    f"=== RAW RESPONSE ===\n{response}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass

            skill_name = None
            description = None
            code = None
            raw_test_code = None
            syntax_error_text = "No usable payload found in response."
            for s_name, s_desc, s_code, s_test in self._candidate_payloads(
                response, topic
            ):
                if self._ast_ok(s_code):
                    skill_name, description, code, raw_test_code = (
                        s_name,
                        s_desc,
                        s_code,
                        s_test,
                    )
                    break
                try:
                    ast.parse(s_code)
                except SyntaxError as se:
                    syntax_error_text = (
                        f"SyntaxError: {se.msg} (line {se.lineno})"
                    )

            if code is None:
                error_feedback = (
                    f"{syntax_error_text} Your reply was not usable. Return "
                    "EITHER a valid JSON object with 'skill_name', "
                    "'description', 'code', 'test_code' string keys, OR "
                    "two markdown blocks: ```python (skill code with an "
                    "execute() entrypoint) and ```python (unittest tests). "
                    "Ensure all quotes/brackets are balanced."
                )
                history.append(
                    {
                        "attempt": attempt,
                        "code": "",
                        "error": error_feedback,
                    }
                )
                logger.warning(
                    f"Attempt {attempt} failed: Invalid JSON structure."
                )
                continue

            def sanitize_test_script(test_code: str, s_name: str) -> str:
                header = [
                    "import unittest",
                    "import sys",
                    "import typing",
                    "from typing import Any, Dict, List, Optional, Union",
                    f"from {s_name} import *",
                ]
                clean_lines = test_code.splitlines()
                has_unittest = any("unittest" in line for line in clean_lines)

                prefix = (
                    "\n".join(header) + "\n\n"
                    if not has_unittest
                    else "import typing\n"
                )
                return prefix + test_code

            test_code = sanitize_test_script(raw_test_code, skill_name)

            try:
                # 1. Validate AST and Run tests + harness
                self.validator.validate_and_run(
                    skill_name, code, test_code
                )

                # 2. Resolve target file path
                skills_dir = "skills"
                os.makedirs(skills_dir, exist_ok=True)
                file_path = os.path.join(skills_dir, f"{skill_name}.py")

                # 3. Governor check before writing to disk
                if self.governor is None:
                    raise SynthesizerError(
                        "Skill writing requires a PermissionGovernor."
                    )
                if not self.governor.request_skill_write_permission(
                    skill_name, file_path, code
                ):
                    raise SynthesizerError(
                        f"Skill write for '{skill_name}' was denied."
                    )

                # 4. Persist to disk
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

                # 5. Auto-index into Project Memory
                if self.project is not None:
                    abs_path = Path(file_path).resolve()
                    project_root = (
                        self.project.active_root or abs_path.parent
                    )
                    self.project.upsert_file(
                        abs_path, brain=self.brain, project_root=project_root
                    )
                    logger.info("Auto-indexed skill file: %s", file_path)

                # 6. Save to Procedural Memory
                skill = Skill(
                    name=skill_name,
                    description=description,
                    file_path=file_path,
                    verification_tier="mock",
                    success_count=1,
                    fail_count=0,
                )
                self.procedural.register(skill)
                logger.info(
                    "Successfully synthesized and registered skill '%s'",
                    skill_name,
                )

                # 7. Harvest DPO pair if there was a prior failed attempt
                if (
                    latest_rejected_code is not None
                    and self.dataset_builder is not None
                    and not isinstance(self.brain, MockBrain)
                ):
                    clean_chosen = self._redact_secrets(code)
                    clean_rejected = self._redact_secrets(latest_rejected_code)
                    try:
                        self.dataset_builder.harvest_dpo_pair(
                            prompt=topic,
                            chosen=clean_chosen,
                            rejected=clean_rejected,
                            metadata={
                                "source": "skill_synthesizer",
                                "error": latest_error,
                                "success_source": "mock_only",  # tests are LLM-generated
                                "chosen_exit_code": 0,
                                "rejected_exit_code": 1,
                            },
                        )
                    except Exception as exc:
                        logger.warning("Failed to harvest DPO pair from skill repair: %s", exc)

                # 8. Log skill_repair_resolved to episodic memory
                if (
                    latest_rejected_code is not None
                    and self.episodic_memory is not None
                ):
                    try:
                        import json as _json
                        self.episodic_memory.log_event(
                            EpisodicLog(
                                kind="skill_repair_resolved",
                                content=_json.dumps({
                                    "prompt": topic,
                                    "chosen_code": self._redact_secrets(code),
                                    "rejected_code": self._redact_secrets(latest_rejected_code),
                                    "error": latest_error,
                                    "skill_name": skill_name,
                                    "exit_code": 0,
                                }),
                                outcome="success",
                                strategy_label="skill_synthesis",
                                reasoning_domain="code_synthesis",
                                outcome_class="success",
                            )
                        )
                    except Exception as exc:
                        logger.warning("Failed to log skill_repair_resolved: %s", exc)

                # 9. Record verified ReasoningEpisode in reasoning.db
                history.append(
                    {
                        "attempt": attempt,
                        "code": code,
                        "error": None,
                    }
                )
                if self.reasoning_mem is not None:
                    first_err = (
                        history[0].get("error") if len(history) > 1 else None
                    )
                    diag = (
                        "Initial attempt failed validation; "
                        "self-corrected via error feedback."
                        if len(history) > 1
                        else None
                    )
                    novelty = 0.85 if len(history) > 1 else 0.7
                    episode = ReasoningEpisode(
                        task_id=(
                            f"synth_{skill_name}_{int(time.time())}_"
                            f"{uuid.uuid4().hex[:6]}"
                        ),
                        state=(
                            f"Synthesize procedural skill for topic '{topic}'"
                        ),
                        hypothesis=(
                            f"Implement valid Python tool for '{topic}' "
                            "adhering to AST rules and passing unit tests"
                        ),
                        action=f"Synthesized skill '{skill_name}':\n{code}",
                        observation=(
                            "Passed AST security validation and unit tests."
                        ),
                        error=first_err,
                        diagnosis=diag,
                        revised_hypo=code if len(history) > 1 else None,
                        generalized_rule=(
                            f"Valid procedural skill for {topic}"
                        ),
                        strategy_label="skill_synthesis",
                        reasoning_domain="code_synthesis",
                        outcome_class="success",
                        hypothesis_count=len(history),
                        verified=True,
                        confidence=1.0,
                        novelty_score=novelty,
                        entropy_score=0.7,
                    )
                    try:
                        self.reasoning_mem.log_episode(episode)
                    except Exception as exc:
                        logger.warning(
                            "Failed to log synthesis episode: %s", exc
                        )

                return skill

            except SecurityError as e:
                error_feedback = str(e)
                if "open" in error_feedback:
                    error_feedback += (
                        " (HINT: Replace open(...) with "
                        "pathlib.Path(path).read_bytes() or .read_text())"
                    )
                if (
                    "__class__" in error_feedback
                    or "getattr" in error_feedback
                ):
                    error_feedback += (
                        " (HINT: NEVER write self.__class__(...) or "
                        "getattr(...). Do not override __and__/__or__ and "
                        "instead write plain module-level functions.)"
                    )
                if (
                    "Unit tests failed" in error_feedback
                    or "AssertionError" in error_feedback
                ):
                    error_feedback += (
                        " (HINT: Manually evaluate each assertion with real "
                        "values BEFORE answering. If a test expectation is "
                        "mathematically wrong, fix the TEST; if code is "
                        "wrong, fix the CODE.)"
                    )
                # Track the latest full code candidate + error for DPO harvesting
                latest_rejected_code = code or ""
                latest_error = error_feedback
                history.append(
                    {
                        "attempt": attempt,
                        "code": code or "",
                        "error": error_feedback,
                    }
                )
                logger.warning(
                    f"Attempt {attempt} failed validation: {error_feedback}"
                )
                continue

        # If all retries fail, log the failure episode to reasoning.db
        if self.reasoning_mem is not None:
            slug = self._slugify_topic(topic)
            task_id = (
                f"synth_{slug}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            )
            episode = ReasoningEpisode(
                task_id=task_id,
                state=f"Synthesize procedural skill for topic '{topic}'",
                hypothesis=f"Implement valid Python tool for '{topic}'",
                action=history[-1].get("code", "") if history else "",
                observation=(
                    f"Failed validation after {len(history)} attempts: "
                    f"{error_feedback}"
                ),
                error=str(error_feedback),
                diagnosis="Exhausted retry budget without passing validation.",
                strategy_label="skill_synthesis",
                reasoning_domain="code_synthesis",
                outcome_class="failure",
                hypothesis_count=len(history),
                verified=False,
                confidence=0.0,
                novelty_score=0.9,
                entropy_score=0.6,
            )
            try:
                self.reasoning_mem.log_episode(episode)
            except Exception as exc:
                logger.warning(
                    "Failed to log failed synthesis episode: %s", exc
                )

        raise SynthesizerError(
            f"Failed to synthesize skill '{topic}' after {self.max_retries} "
            f"retries. Last error: {error_feedback}"
        )


class KnowledgeSynthesizer:
    def __init__(self, brain: BaseBrain, semantic: "SemanticMemory"):
        self.brain = brain
        self.semantic = semantic

    def distill_to_semantic_db(
        self, raw_text: str, topic: str
    ) -> tuple[list[Fact], list[Passage]]:
        """Extracts atomic concept facts and context passages."""
        # Avoid blowing up context window, restrict to ~20K chars max.
        text_chunk = raw_text[:20000]

        # 1. Extract Facts
        prompt_facts = f"""
You are an expert knowledge extractor.
Extract concise, falsifiable, atomic facts from the text about "{topic}".
Focus on entities, dates, causality, statistics, and definitions.
Output ONLY a JSON array of strings.

CRITICAL: Respond ONLY with a raw JSON array.

Text:
{text_chunk}
"""
        response_facts = self.brain.generate(prompt_facts)
        facts = self.brain.extract_json(response_facts)

        added_facts: list[Fact] = []
        if isinstance(facts, list):
            for f_text in facts:
                if isinstance(f_text, str) and len(f_text) > 10:
                    fact = Fact(
                        id=0,
                        text=f_text,
                        confidence=0.8,
                        source_type="web_ingestion",
                        topic=topic,
                        created_at=(
                            datetime.datetime.now(timezone.utc).isoformat()
                        ),
                    )
                    created, returned_fact = self.semantic.add_fact(fact)
                    if created:
                        added_facts.append(returned_fact)

        # 2. Extract Passages
        prompt_passages = f"""
You are an expert historian/scientist.
Extract 1 to 3 critical narrative passages from the text about "{topic}".
Preserve complex causal sequences, methodology, or nuance.
Output ONLY a JSON array of strings.

CRITICAL: Respond ONLY with a raw JSON array.

Text:
{text_chunk}
"""
        response_passages = self.brain.generate(prompt_passages)
        passages = self.brain.extract_json(response_passages)

        added_passages: list[Passage] = []
        if isinstance(passages, list):
            for p_text in passages:
                if isinstance(p_text, str) and len(p_text) >= 150:
                    passage = Passage(
                        id=0,
                        text=p_text,
                        topic=topic,
                        source_type="web_ingestion",
                        created_at=(
                            datetime.datetime.now(timezone.utc).isoformat()
                        ),
                    )
                    self.semantic.add_passage(passage)
                    added_passages.append(passage)

        return added_facts, added_passages


class CSharpSynthesizer:
    def __init__(
        self,
        brain: BaseBrain,
        unity_client: Any,
        project: ProjectMemory,
        reasoning_memory: ReasoningMemory | None = None,
    ):
        self.brain = brain
        self.unity_client = unity_client
        self.project = project
        if reasoning_memory is None:
            from agent.config import REASONING_DB

            try:
                self.reasoning_mem = ReasoningMemory(REASONING_DB)
            except Exception:
                self.reasoning_mem = None
        else:
            self.reasoning_mem = reasoning_memory
        self.max_retries = 2

    def _pre_validate_csharp(self, code: str) -> str | None:
        """Check for balanced braces and basic class structure."""
        if code.count("{") != code.count("}"):
            return "Syntax Error: Unbalanced braces '{}' in C# code."
        if "class " not in code:
            return "Syntax Error: Missing 'class' definition."
        return None

    def _generate_csharp_prompt(
        self, topic: str, script_type: str, error_feedback: str | None = None
    ) -> str:
        prompt = "You are an expert Unity C# Developer.\n"
        prompt += f"Write a Unity C# {script_type} for: {topic}\n"
        if script_type == "MonoBehaviour":
            prompt += (
                "Ensure the class inherits from MonoBehaviour and includes "
                "standard Unity lifecycle methods (Start, Update) if needed.\n"
            )
        elif script_type == "ScriptableObject":
            prompt += (
                "Ensure the class inherits from ScriptableObject and has a "
                "[CreateAssetMenu] attribute.\n"
            )
        elif script_type == "EditorWindow":
            prompt += (
                "Ensure the class inherits from UnityEditor.EditorWindow and "
                "includes a [MenuItem] for instantiation.\n"
            )

        prompt += (
            "\nOutput ONLY a valid JSON object with keys:\n"
            "- 'class_name' (string)\n"
            "- 'code' (string, properly escaped)\n"
        )
        if error_feedback:
            prompt += (
                "\nYour previous attempt failed compilation with the following "
                f"errors:\n{error_feedback}\n\n"
                "Please fix the errors and output the revised JSON.\n"
            )
        return prompt

    def learn_csharp_script(
        self, topic: str, script_type: str = "MonoBehaviour"
    ) -> dict[str, Any]:
        """Synthesize C# script and run Roslyn self-repair loop."""
        error_feedback = None
        history: list[dict[str, Any]] = []

        for attempt in range(self.max_retries + 1):
            if error_feedback:
                time.sleep(2.0)

            prompt = self._generate_csharp_prompt(
                topic, script_type, error_feedback
            )
            try:
                response = self.brain.generate(
                    prompt, json_mode=True, temperature=0.2
                )
            except TypeError:
                response = self.brain.generate(prompt)

            parsed = self.brain.extract_json(response)
            if (
                not isinstance(parsed, dict)
                or not parsed.get("code")
                or not parsed.get("class_name")
            ):
                error_feedback = (
                    "Invalid JSON format. Require 'class_name' and 'code'."
                )
                history.append(
                    {"attempt": attempt, "code": "", "error": error_feedback}
                )
                continue

            class_name = parsed["class_name"]
            code = parsed["code"]

            pre_val_err = self._pre_validate_csharp(code)
            if pre_val_err:
                error_feedback = pre_val_err
                history.append(
                    {"attempt": attempt, "code": code, "error": error_feedback}
                )
                continue

            # Write to Assets/Scripts/ (create if not exists)
            scripts_dir = self.unity_client.project_path / "Assets" / "Scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            file_path = scripts_dir / f"{class_name}.cs"
            file_path.write_text(code, encoding="utf-8")

            # Run headless test to trigger compilation
            test_result = self.unity_client.run_tests()

            compiler_errors = test_result.get("compiler_errors", [])
            # Filter errors for our specific file
            my_errors = [
                e for e in compiler_errors
                if f"{class_name}.cs" in e.get("file", "")
            ]

            if my_errors:
                err_strs = [
                    f"Line {e['line']}: error {e['code']}: {e['message']}"
                    for e in my_errors
                ]
                error_feedback = "\n".join(err_strs)
                history.append(
                    {"attempt": attempt, "code": code, "error": error_feedback}
                )
                continue

            # Success
            if self.project:
                self.project.upsert_file(file_path.resolve(), brain=self.brain)

            if self.reasoning_mem is not None:
                episode = ReasoningEpisode(
                    task_id=(
                        f"csharp_synth_{int(time.time())}_"
                        f"{uuid.uuid4().hex[:6]}"
                    ),
                    state=f"Synthesize C# {script_type}: {topic}",
                    hypothesis="Implement valid Unity C# script",
                    action=f"Created {class_name}.cs",
                    observation="Passed Unity Roslyn compiler.",
                    outcome_class="success",
                    hypothesis_count=len(history) + 1,
                    verified=True,
                    confidence=1.0,
                    novelty_score=0.85 if attempt > 0 else 0.7,
                    entropy_score=0.7,
                )
                try:
                    self.reasoning_mem.log_episode(episode)
                except Exception:
                    pass

            return {
                "status": "success",
                "file_path": str(file_path),
                "class_name": class_name,
                "attempts": attempt + 1
            }

        # Failure
        if self.reasoning_mem is not None:
            episode = ReasoningEpisode(
                task_id=(
                    f"csharp_synth_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                ),
                state=f"Synthesize C# {script_type}: {topic}",
                hypothesis="Implement valid Unity C# script",
                action="Attempted synthesis",
                observation=(
                    f"Failed validation after {self.max_retries} "
                    f"retries: {error_feedback}"
                ),
                error=error_feedback,
                diagnosis="Exhausted retry budget for C# compilation.",
                outcome_class="failure",
                hypothesis_count=len(history),
                verified=False,
                confidence=0.0,
                novelty_score=0.9,
                entropy_score=0.6,
            )
            try:
                self.reasoning_mem.log_episode(episode)
            except Exception:
                pass

        raise SynthesizerError(
            f"Failed to synthesize C# script after {self.max_retries} "
            f"retries. Errors: {error_feedback}"
        )
