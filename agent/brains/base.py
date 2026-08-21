"""Abstract interface every brain (MockBrain, and Phase 1's real brains) implements."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class BaseBrain(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Free-form text generation given a prompt."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embedding for this text. May delegate to a shared EmbeddingEngine,
        or, for a real provider in Phase 1, use that provider's own
        embedding endpoint instead."""

    @staticmethod
    def extract_json(text: str) -> dict | list | None:
        """Best-effort JSON extraction from an LLM response: strips a
        wrapping markdown code fence if present, then finds the first
        top-level {...} or [...] block. Returns None rather than raising
        if nothing parses — callers decide how to handle a brain that
        didn't return valid JSON."""
        stripped = text.strip()
        
        # Strip <think>...</think> blocks from models like DeepSeek/Qwen
        stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()
        
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
        match = re.search(r"[\{\[].*[\}\]]", stripped, flags=re.DOTALL)
        candidate = match.group(0) if match else stripped
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
