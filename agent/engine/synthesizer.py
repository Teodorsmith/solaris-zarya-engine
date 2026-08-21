import json
import logging
import os
import re
from typing import Optional

from agent.brains.base import BaseBrain
from agent.engine.retriever import Retriever
from agent.engine.validator import SkillValidator, SecurityError
from agent.memory.procedural import ProceduralMemory
from agent.models import Skill

logger = logging.getLogger(__name__)

class SynthesizerError(Exception):
    pass

class SkillSynthesizer:
    def __init__(self, brain: BaseBrain, retriever: Retriever, procedural: ProceduralMemory, validator: SkillValidator):
        self.brain = brain
        self.retriever = retriever
        self.procedural = procedural
        self.validator = validator
        self.max_retries = 2

    def _generate_skill_prompt(self, topic: str, context: str, error_feedback: Optional[str] = None) -> str:
        prompt = f"""
You are an expert Python developer writing a safe, pure-logic skill.
Topic: {topic}
Context:
{context}

You must write two Python scripts:
1. The skill code. It must contain a function called `execute` that returns a JSON-serializable value.
2. The unit test suite using `unittest`.

You are strictly restricted to the following modules:
json, re, math, typing, dataclasses, datetime, collections, pathlib, textwrap, enum, uuid, hashlib, base64, copy, functools, itertools

DO NOT USE os.system, subprocess, eval, exec, __builtins__, or any reflection/dunder methods.
"""
        if error_feedback:
            prompt += f"\nYour previous attempt failed with this error:\n{error_feedback}\nPlease fix the error.\n"
        
        prompt += """
Output ONLY a valid JSON object matching exactly this schema, with no markdown formatting outside of it:
{
  "skill_name": "name_of_skill_using_underscores",
  "description": "Short description",
  "code": "def execute():\\n    ...",
  "test_code": "import unittest\\n..."
}
"""
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
                
                # 2. Persist to disk
                skills_dir = "skills"
                os.makedirs(skills_dir, exist_ok=True)
                file_path = os.path.join(skills_dir, f"{skill_name}.py")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # 3. Save to Procedural Memory
                skill = Skill(
                    name=skill_name,
                    description=description,
                    file_path=file_path,
                    verification_tier="mock",
                    success_count=1,
                    fail_count=0
                )
                self.procedural.register(skill)
                logger.info(f"Successfully synthesized and validated skill '{skill_name}'")
                return skill

            except SecurityError as e:
                error_feedback = str(e)
                logger.warning(f"Attempt {attempt} failed validation: {error_feedback}")
                continue

        raise SynthesizerError(f"Failed to synthesize skill '{topic}' after {self.max_retries} retries. Last error: {error_feedback}")
