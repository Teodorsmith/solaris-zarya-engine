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

"""
Deterministic, offline brain. No API key, no network beyond the local
embedding model. Exists so Phase 0 runs and is testable with nothing
external required.

MockBrain does not reason — `generate()` is a stub that's never actually
called by Phase 0's main flow. `ask` is answered directly by the
retriever from retrieved facts (engine/retriever.py), and `learn` only
calls the seeder (see cli.py) — neither needs free-form generation. The
method still has to exist and work, though, so BaseBrain's contract is
fully implemented and Phase 1 can swap in a real brain without changing
any caller.
"""

from __future__ import annotations

from agent.brains.base import BaseBrain
from agent.memory.embeddings import EmbeddingEngine


class MockBrain(BaseBrain):
    def __init__(self, embedder: EmbeddingEngine | None = None):
        self._embedder = embedder

    def generate(self, prompt: str, **kwargs) -> str:
        # A simple deterministic "generation" for tests
        if "skill code" in prompt or "Topic: " in prompt:
            import re
            match = re.search(r"Topic:\s*([^\n]+)", prompt)
            skill_name = "email_normalizer"
            if match:
                skill_name = re.sub(r"[^a-zA-Z0-9]+", "_", match.group(1)).strip("_").lower()
            return f"""
{{
  "skill_name": "{skill_name}",
  "description": "Validates and normalizes strings",
  "code": "import re\\n\\ndef execute():\\n    pass\\n",
  "test_code": "import unittest\\n\\nclass TestEmail(unittest.TestCase):\\n    def test_pass(self):\\n        self.assertTrue(True)\\n"
}}
"""
        return f"[Mock generation for prompt: {prompt[:30]}...]"

    def embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError(
                "MockBrain has no EmbeddingEngine configured. "
                "Pass one in: MockBrain(embedder=EmbeddingEngine())."
            )
        return self._embedder.embed(text)
