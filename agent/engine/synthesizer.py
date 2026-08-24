# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from agent.brains.base import BaseBrain
from agent.engine.governor import PermissionGovernor
from agent.engine.retriever import Retriever
from agent.engine.validator import SkillValidator, SecurityError
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
        project: Optional[ProjectMemory] = None,
    ):
        self.brain = brain
        self.retriever = retriever
        self.procedural = procedural
        self.validator = validator
        self.governor = governor
        self.project = project  # may be None — hook degrades gracefully
        self.max_retries = 2

    def _generate_skill_prompt(self, topic: str, context: str, error_feedback: Optional[str] = None) -> str:
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
2. Allowed imports: `pathlib`, `json`, `re`, `math`, `hashlib`, `typing`, `dataclasses`, `datetime`.
3. Do NOT use dunder attributes (e.g., `__class__`, `__subclasses__`).
4. Output valid JSON containing 'skill_name', 'description', 'code', and 'test_code'.
"""
        if error_feedback:
            prompt += f"\nYour previous attempt failed with this error:\n{error_feedback}\nPlease fix the error.\n"
            
        return prompt

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
                logger.info("Pausing for 2.0s before repair attempt to avoid rate limits...")
                time.sleep(2.0)
                
            prompt = self._generate_skill_prompt(topic, context, error_feedback)
            response = self.brain.generate(prompt)
            
            parsed = self.brain.extract_json(response)
            if not parsed or "code" not in parsed or "test_code" not in parsed:
                error_feedback = "You did not return a valid JSON object with 'code' and 'test_code' keys."
                logger.warning(f"Attempt {attempt} failed: Invalid JSON structure.")
                continue

            skill_name = parsed.get("skill_name", "generated_skill").replace(" ", "_")
            description = parsed.get("description", topic)
            code = parsed["code"]
            test_code = parsed["test_code"]

            try:
                # 1. Validate AST and Run tests + harness
                result_schema = self.validator.validate_and_run(skill_name, code, test_code)

                # 2. Resolve target file path
                skills_dir = "skills"
                os.makedirs(skills_dir, exist_ok=True)
                file_path = os.path.join(skills_dir, f"{skill_name}.py")

                # 3. Governor check before writing to disk
                if self.governor is None:
                    raise SynthesizerError("Skill writing requires a PermissionGovernor.")
                if not self.governor.request_skill_write_permission(skill_name, file_path, code):
                    raise SynthesizerError(f"Skill file write for '{skill_name}' was denied by Governor.")
                
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
                    self.project.upsert_file(abs_path, brain=self.brain, project_root=project_root)
                    logger.info("Auto-indexed skill file: %s", file_path)

                # 6. Save to Procedural Memory
                skill = Skill(
                    name=skill_name,
                    description=description,
                    file_path=file_path,
                    verification_tier="mock",
                    success_count=1,
                    fail_count=0
                )
                self.procedural.register(skill)
                logger.info(f"Successfully synthesized, validated, and registered skill '{skill_name}'")
                return skill

            except SecurityError as e:
                error_feedback = str(e)
                if "open" in error_feedback:
                    error_feedback += " (HINT: Replace open(...) with pathlib.Path(path).read_bytes() or .read_text())"
                logger.warning(f"Attempt {attempt} failed validation: {error_feedback}")
                continue

        raise SynthesizerError(f"Failed to synthesize skill '{topic}' after {self.max_retries} retries. Last error: {error_feedback}")

class KnowledgeSynthesizer:
    def __init__(self, brain: BaseBrain, semantic: 'SemanticMemory'):
        self.brain = brain
        self.semantic = semantic

    def distill_to_semantic_db(self, raw_text: str, topic: str) -> tuple[list['Fact'], list['Passage']]:
        """
        Extracts atomic concept facts and 200-500 word context_passages from raw text.
        Persists them to semantic.db. Deduplication is handled by semantic.add_fact().
        Returns a tuple of (added_facts, added_passages).
        """
        from agent.models import Fact, Passage
        import datetime
        from datetime import timezone

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
                        created_at=datetime.datetime.now(timezone.utc).isoformat()
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
                        created_at=datetime.datetime.now(timezone.utc).isoformat()
                    )
                    self.semantic.add_passage(passage)
                    added_passages.append(passage)
                    
        return added_facts, added_passages
