# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Abstract interface every brain (MockBrain, and Phase 1's real brains) implements."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class QuotaExceededError(Exception):
    """Raised when a provider hits an unrecoverable quota or rate limit (e.g. 429)."""


class BaseBrain(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
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
            stripped = re.sub(r"^```[a-zA-Z]*[ \t]*\n?", "", stripped)
            stripped = re.sub(r"\n?```\s*$", "", stripped)

        candidates = [stripped]
        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start != -1 and end > start:
                candidates.append(stripped[start:end + 1])

        for candidate in candidates:
            for repaired in BaseBrain._json_repair_variants(candidate):
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _escape_control_chars_in_strings(raw: str) -> str:
        """Escape literal newlines/tabs inside JSON string values only.

        Small models frequently emit code values containing real newlines,
        which is invalid JSON ("Invalid control character"). Escaping them
        preserves semantics exactly (a literal \\n in a string == escaped).
        """
        out: list[str] = []
        in_string = False
        escaped = False
        for ch in raw:
            if in_string:
                if escaped:
                    out.append(ch)
                    escaped = False
                elif ch == "\\":
                    out.append(ch)
                    escaped = True
                elif ch == '"':
                    in_string = False
                    out.append(ch)
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(ch)
            else:
                if ch == '"':
                    in_string = True
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _json_repair_variants(raw: str):
        """Yield progressively more aggressive repair attempts."""
        yield raw
        yield BaseBrain._escape_control_chars_in_strings(raw)
        no_trailing = re.sub(r",\s*([\]}])", r"\1", raw)
        yield no_trailing
        yield BaseBrain._escape_control_chars_in_strings(no_trailing)
        # Smart quotes -> ASCII quotes (outside-string heuristic is skipped;
        # smart quotes are illegal inside JSON strings anyway)
        dequoted = re.sub(r"[\u201c\u201d]", '"', raw)
        dequoted = re.sub(r"[\u2018\u2019]", "'", dequoted)
        yield dequoted
        yield BaseBrain._escape_control_chars_in_strings(
            re.sub(r",\s*([\]}])", r"\1", dequoted)
        )
