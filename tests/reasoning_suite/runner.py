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

"""ZPD Binary Search Runner -- Mitigation #66.

Runs the reasoning benchmark suite and computes the ZPD ceiling per category.
The ceiling is the highest difficulty level the agent can reliably clear.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.config import (
    REASONING_SUITE_DIR,
    ZPD_CATEGORIES,
    ZPD_DIFFICULTY_MAX,
    ZPD_DIFFICULTY_MIN,
    ZPD_MAX_ROUNDS,
)
from agent.engine.verifier import SRTVerifier
from agent.models import SRTTrace

logger = logging.getLogger(__name__)


class ZPDRunner:
    """Runs reasoning benchmarks to find the Zone of Proximal Development."""

    def __init__(self, brain_manager) -> None:
        self.brain_manager = brain_manager
        self.verifier = SRTVerifier()
        self.categories_dir = REASONING_SUITE_DIR / "categories"
        self._load_fixtures()

    def _load_fixtures(self) -> None:
        self.fixtures: dict[str, dict[int, dict[str, Any]]] = {}
        if not self.categories_dir.exists():
            return

        for cat in ZPD_CATEGORIES:
            path = self.categories_dir / f"{cat}.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.fixtures[cat] = {}
            for prob in data.get("problems", []):
                diff = prob["difficulty"]
                self.fixtures[cat][diff] = prob

    def run_all(self, dry_run: bool = False) -> dict[str, int]:
        """Run the binary search over all categories.

        Returns:
            dict mapping category name to the ZPD ceiling difficulty level.
        """
        ceilings: dict[str, int] = {}
        for cat in ZPD_CATEGORIES:
            if cat not in self.fixtures:
                logger.warning("ZPDRunner: no fixtures for %s", cat)
                ceilings[cat] = 0
                continue

            logger.info("ZPDRunner: testing %s", cat)
            ceilings[cat] = self._search_category(cat)

        return ceilings

    def _search_category(self, category: str) -> int:
        """Run up to ZPD_MAX_ROUNDS of binary search to find the ceiling."""
        lo = ZPD_DIFFICULTY_MIN
        hi = ZPD_DIFFICULTY_MAX
        last_pass = 0

        for _ in range(ZPD_MAX_ROUNDS):
            if lo > hi:
                break

            mid = (lo + hi) // 2
            prob = self.fixtures[category].get(mid)
            if not prob:
                break

            logger.debug("ZPDRunner: %s difficulty %d...", category, mid)
            passed = self._run_problem(prob)

            if passed:
                last_pass = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return last_pass

    def _run_problem(self, prob: dict[str, Any]) -> bool:
        """Run a single problem and evaluate PASS/FAIL.

        A PASS requires BOTH the correct answer and a valid SRT.
        """
        prompt = prob["prompt"]
        expected_ans = prob["answer_key"]
        srt_required = prob.get("srt_required", True)

        # We expect the brain to return JSON matching SRTTrace.
        # In a real environment we'd use a strict schema parser.
        from rich.console import Console

        from agent.brains.base import QuotaExceededError

        console = Console()

        while True:
            try:
                raw_response = self.brain_manager.brain.generate(
                    f"{prompt}\nReturn JSON matching SRTTrace schema. "
                    f"Expected inference rule is {prob.get('expected_inference_rule')}."
                )
                break
            except QuotaExceededError as qe:
                console.print(f"[yellow]Brain quota exceeded: {qe}[/yellow]")
                new_brain = self.brain_manager.fallback()
                if new_brain:
                    console.print(
                        f"[bold green]Switched to fallback brain: {new_brain.__class__.__name__}[/bold green]"
                    )
                else:
                    console.print(
                        "[bold red]All brain quotas exhausted during benchmark.[/bold red]"
                    )
                    return False

        try:
            # Simple JSON extract for the mock brain
            if "{" in raw_response and "}" in raw_response:
                start = raw_response.find("{")
                end = raw_response.rfind("}") + 1
                json_str = raw_response[start:end]
            else:
                json_str = raw_response

            data = json.loads(json_str)
            srt = SRTTrace(**data)
        except Exception as exc:
            logger.debug("ZPDRunner: Failed to parse SRTTrace: %s", exc)
            return False

        # Check answer correctness (substring match for now)
        if expected_ans.lower() not in srt.conclusion.lower():
            logger.debug(
                "ZPDRunner: Wrong answer (got %s, expected %s)",
                srt.conclusion,
                expected_ans,
            )
            return False

        if not srt_required:
            return True

        # SRT required: run verifier
        v_result = self.verifier.verify(srt)
        if not v_result.verified:
            logger.debug("ZPDRunner: SRT rejected (%s)", v_result.reason)
            return False

        return True
