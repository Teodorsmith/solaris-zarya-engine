# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import ast
import logging
import os
import re
from pathlib import Path

from agent.brains.base import BaseBrain
from agent.engine.governor import PermissionGovernor
from agent.engine.retriever import Retriever
from agent.engine.validator import SecurityError, SkillValidator
from agent.memory.procedural import ProceduralMemory
from agent.memory.project import ProjectMemory
from agent.models import Skill

logger = logging.getLogger(__name__)


class SynthesizerError(Exception):
    pass


class SkillSynthesizer:
    def __init__(
        self,
        brain: BaseBrain,
        retriever: Retriever,
        procedural: ProceduralMemory,
        validator: SkillValidator,
        governor: PermissionGovernor,
        project: ProjectMemory | None = None,
    ):
        self.brain = brain
        self.retriever = retriever
        self.procedural = procedural
        self.validator = validator
        self.governor = governor
        self.project = project  # may be None — hook degrades gracefully
        self.max_retries = 6

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
   - For reading files, ALWAYS use: `pathlib.Path(file_path).read_bytes()` or `pathlib.Path(file_path).read_text(encoding="utf-8")`.
   - For writing files, ALWAYS use: `pathlib.Path(file_path).write_bytes(...)` or `pathlib.Path(file_path).write_text(...)`.
2. Allowed imports: `pathlib`, `json`, `re`, `math`, `hashlib`, `typing`, `dataclasses`, `datetime`, `enum`, `collections`, `functools`, `itertools`, `unittest`.
3. FORBIDDEN patterns (AST validator will REJECT):
   - NO `__class__`, `__dict__`, `__subclasses__`, `__bases__`, `__mro__`, `__globals__`, `__builtins__`
   - NO `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `eval`, `exec`, `compile`, `__import__`
   - NO `sys` module, NO `os` module
4. Use simple direct attribute access: `obj.attr` not `getattr(obj, "attr")`.
5. Avoid `@classmethod`/`cls` patterns — prefer plain methods and module-level functions.
6. The skill MUST define an `execute(**kwargs)` function as its entrypoint.
7. PREFERRED STYLE — plain module-level functions over plain ints/strings.
   NO operator overloading (__and__, __or__, __xor__, __invert__), NO
   custom __new__/__init__ tricks, NO Enum subclasses needing them.
   Example of the expected shape:

       def has_flag(flags: int, bit: int) -> bool:
           return (flags & (1 << bit)) != 0

       def execute(flags: int = 0, bit: int = 0) -> dict:
           return {{"result": has_flag(flags, bit)}}

8. OUTPUT FORMAT — return EITHER:
   A) A single valid JSON object with keys 'skill_name', 'description', 'code', 'test_code' (code values as properly escaped JSON strings), OR
   B) Two markdown blocks: first ```python block = the skill code, second ```python block = unittest test cases.
"""
        if error_feedback:
            prompt += f"\nYour previous attempt failed execution testing:\nError:\n{error_feedback}\n\nPlease fix the imports, implementation, and test cases. Output valid, self-contained Python code.\n"

        return prompt

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove stray markdown fences the model may embed inside JSON values."""
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*[ \t]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return cleaned

    @staticmethod
    def _extract_fenced_blocks(text: str) -> list[str]:
        """Extract all ```python / ``` fenced code blocks from raw text."""
        return re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)

    @staticmethod
    def _slugify_topic(topic: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic).strip("_").lower()
        return slug or "generated_skill"

    @staticmethod
    def _unwrap_double_serialized(code: str) -> str:
        """If the model JSON-encoded the code value twice, unwrap one layer."""
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
        """Yield (skill_name, description, code, test_code) candidates.

        Strategy 1: structured JSON (preferred — precise metadata).
        Strategy 2: markdown fences (small models emit escaped JSON strings
        unreliably, but plain fenced code far more reliably).
        """
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
                skill_name = parsed.get("skill_name") or self._slugify_topic(topic)
                description = parsed.get("description") or topic
                yield (
                    str(skill_name).replace(" ", "_"),
                    str(description),
                    self._repair_code(self._unwrap_double_serialized(code)),
                    self._strip_code_fences(self._unwrap_double_serialized(test_code)),
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

        for attempt in range(self.max_retries + 1):
            if error_feedback:
                import time

                logger.info(
                    "Pausing for 2.0s before repair attempt to avoid rate limits..."
                )
                time.sleep(2.0)

            prompt = self._generate_skill_prompt(topic, context, error_feedback)
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
                    syntax_error_text = f"SyntaxError: {se.msg} (line {se.lineno})"

            if code is None:
                error_feedback = (
                    f"{syntax_error_text} Your reply was not usable. Return EITHER "
                    "a valid JSON object with 'skill_name', 'description', 'code', "
                    "'test_code' string keys, OR two markdown blocks: ```python "
                    "(skill code with an execute() entrypoint) and ```python "
                    "(unittest tests). Ensure all quotes/brackets are balanced."
                )
                logger.warning(f"Attempt {attempt} failed: Invalid JSON structure.")
                continue

            def sanitize_test_script(test_code: str, s_name: str) -> str:
                header = [
                    "import unittest",
                    "import sys",
                    "import typing",
                    "from typing import Any, Dict, List, Optional, Union, Tuple, Set",
                    f"from {s_name} import *",
                ]
                clean_lines = test_code.splitlines()
                has_unittest = any("unittest" in line for line in clean_lines)

                prefix = "\n".join(header) + "\n\n" if not has_unittest else "import typing\n"
                return prefix + test_code

            test_code = sanitize_test_script(raw_test_code, skill_name)

            try:
                # 1. Validate AST and Run tests + harness
                result_schema = self.validator.validate_and_run(
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
                        f"Skill file write for '{skill_name}' was denied by Governor."
                    )

                # 4. Persist to disk
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

                # 5. Auto-index into Project Memory so the file is immediately
                #    searchable without needing a manual `project index .`
                if self.project is not None:
                    abs_path = Path(file_path).resolve()
                    # Use the canonical root from the DB — avoids fragile
                    # parent.parent path arithmetic that breaks for arbitrary paths.
                    project_root = self.project.active_root or abs_path.parent
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
                    f"Successfully synthesized, validated, and registered skill '{skill_name}'"
                )
                return skill

            except SecurityError as e:
                error_feedback = str(e)
                if "open" in error_feedback:
                    error_feedback += " (HINT: Replace open(...) with pathlib.Path(path).read_bytes() or .read_text())"
                if "__class__" in error_feedback or "getattr" in error_feedback:
                    error_feedback += (
                        " (HINT: NEVER write self.__class__(...) or getattr(...). "
                        "Do not override __and__/__or__/__xor__/__invert__ and do "
                        "not subclass Enum with custom __new__. Instead write plain "
                        "module-level functions operating on ints, e.g. def "
                        "has_flag(flags: int, bit: int) -> bool: return (flags & "
                        "(1 << bit)) != 0, plus an execute(**kwargs) entrypoint.)"
                    )
                if "Unit tests failed" in error_feedback or "AssertionError" in error_feedback:
                    error_feedback += (
                        " (HINT: Manually evaluate each assertion with real values "
                        "BEFORE answering. Bit numbering starts at 0 from the least "
                        "significant bit: 0b1010 has bits 1 and 3 set, so bit 3 IS "
                        "set. If a test expectation is mathematically wrong, fix the "
                        "TEST; if the implementation is wrong, fix the CODE. Keep "
                        "implementation and tests consistent.)"
                    )
                logger.warning(f"Attempt {attempt} failed validation: {error_feedback}")
                continue

        raise SynthesizerError(
            f"Failed to synthesize skill '{topic}' after {self.max_retries} retries. Last error: {error_feedback}"
        )


class KnowledgeSynthesizer:
    def __init__(self, brain: BaseBrain, semantic: "SemanticMemory"):
        self.brain = brain
        self.semantic = semantic

    def distill_to_semantic_db(
        self, raw_text: str, topic: str
    ) -> tuple[list["Fact"], list["Passage"]]:
        """
        Extracts atomic concept facts and 200-500 word context_passages from raw text.
        Persists them to semantic.db. Deduplication is handled by semantic.add_fact().
        Returns a tuple of (added_facts, added_passages).
        """
        import datetime
        from datetime import timezone

        from agent.models import Fact, Passage

        # Avoid blowing up context window with raw HTML dumps, restrict to ~20K chars max.
        text_chunk = raw_text[:20000]

        # 1. Extract Facts
        prompt_facts = f"""
You are an expert knowledge extractor.
Extract concise, falsifiable, atomic facts from the following text about "{topic}".
Focus on entities, dates, causality, statistics, and definitions.
Output ONLY a JSON array of strings.

CRITICAL: Respond ONLY with a raw JSON array. Do NOT include markdown code blocks, do NOT write introductory or concluding text.

Text:
{text_chunk}
"""
        response_facts = self.brain.generate(prompt_facts)
        facts = self.brain.extract_json(response_facts)

        added_facts = []
        if isinstance(facts, list):
            for f_text in facts:
                if isinstance(f_text, str) and len(f_text) > 10:
                    fact = Fact(
                        id=0,
                        text=f_text,
                        confidence=0.8,
                        source_type="web_ingestion",
                        topic=topic,
                        created_at=datetime.datetime.now(timezone.utc).isoformat(),
                    )
                    created, returned_fact = self.semantic.add_fact(fact)
                    if created:
                        added_facts.append(returned_fact)

        # 2. Extract Passages
        prompt_passages = f"""
You are an expert historian/scientist.
Extract 1 to 3 critical narrative passages (200-500 words each) from the following text about "{topic}".
Preserve complex causal sequences, methodology, or nuance that would be lost in atomic facts.
Output ONLY a JSON array of strings.

CRITICAL: Respond ONLY with a raw JSON array. Do NOT include markdown code blocks, do NOT write introductory or concluding text.

Text:
{text_chunk}
"""
        response_passages = self.brain.generate(prompt_passages)
        passages = self.brain.extract_json(response_passages)

        added_passages = []
        if isinstance(passages, list):
            for p_text in passages:
                # We enforce >150 chars as proxy for a reasonable paragraph length
                if isinstance(p_text, str) and len(p_text) >= 150:
                    passage = Passage(
                        id=0,
                        text=p_text,
                        topic=topic,
                        source_type="web_ingestion",
                        created_at=datetime.datetime.now(timezone.utc).isoformat(),
                    )
                    self.semantic.add_passage(passage)
                    added_passages.append(passage)

        return added_facts, added_passages
