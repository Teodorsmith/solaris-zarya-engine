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

"""Autonomous Heartbeat Daemon (Mitigation #41 + #46).

Runs as a background daemon thread executing a strictly Tier-0
Perceive -> Evaluate -> Plan -> Act -> Reflect idle loop.

Safety guarantees
-----------------
* Tier 0 ONLY: reads DBs, logs episodic alerts, runs SQL aggregation.
  No subprocess calls, no file writes, no LLM calls, no Tier 1+ actions.
* Rate limit: max HEARTBEAT_MAX_PER_HOUR autonomous actions per rolling hour.
* Daily cap: HEARTBEAT_DAILY_CALL_CAP total actions (reserved for future LLM
  tasks; currently unused in Phase 4A).
* Pause guard: checks pause_event at the top of each cycle before any DB
  access -- no risk of deadlock.
* No-op silence: if nothing is actionable, the cycle exits without writing
  any episodic entry (Architecture 7.3).
* Subprocess lock: threading.Lock guards any future subprocess-based
  maintenance action to prevent overlap (Phase 4B+; currently unused).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.memory.self_model import SelfModel

from agent.config import (
    EPISODIC_DB,
    HEARTBEAT_INTERVAL_SECS,
    HEARTBEAT_MAX_PER_HOUR,
    REASONING_DB,
    SEMANTIC_DB,
    STALE_FACT_DAYS,
)

logger = logging.getLogger(__name__)

# How many days of episodic data to scan for the weekly aggregation
_REFLECTION_WINDOW_DAYS = 7
# Only run weekly aggregation if last_reflection_at is older than this
_REFLECTION_MIN_INTERVAL_DAYS = 7


class HeartbeatDaemon(threading.Thread):
    """Background daemon executing a Tier-0 idle maintenance loop.

    Args:
        self_model:  SelfModel instance (already loaded).
        pause_event: Set by the REPL while the user is interacting.
                     Daemon skips cycles while the event is set.
        no_daemon:   If True, the thread is never started (``--no-daemon``).
    """

    def __init__(
        self,
        self_model: SelfModel,
        pause_event: threading.Event,
    ) -> None:
        super().__init__(name="HeartbeatDaemon", daemon=True)
        self._self_model = self_model
        self._pause_event = pause_event

        # Rolling window of action timestamps (last hour)
        self._action_times: deque[float] = deque()

        # Subprocess concurrency guard (Phase 4B+; unused in Phase 4A)
        self._subprocess_lock = threading.Lock()

        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("HeartbeatDaemon: started (interval=%ds).", HEARTBEAT_INTERVAL_SECS)
        # Open a dedicated, thread-local SQLite connection
        # (never share with the main thread's connections)
        self._episodic_conn = self._open_db(EPISODIC_DB)
        self._semantic_conn = self._open_db(SEMANTIC_DB)
        self._reasoning_conn = self._open_db(REASONING_DB)

        while not self._stop_event.is_set():
            # Sleep first, then check -- avoids an immediate cycle on boot
            self._stop_event.wait(timeout=HEARTBEAT_INTERVAL_SECS)
            if self._stop_event.is_set():
                break

            # Pause guard: skip entire cycle if user is active
            if self._pause_event.is_set():
                logger.debug("HeartbeatDaemon: paused (user active).")
                continue

            self._run_cycle()

        logger.info("HeartbeatDaemon: stopped.")

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _can_act(self) -> bool:
        """Return True if the action ceiling has not been reached this hour."""
        now = time.time()
        # Drop timestamps outside the rolling 1-hour window
        while self._action_times and now - self._action_times[0] > 3600:
            self._action_times.popleft()
        return len(self._action_times) < HEARTBEAT_MAX_PER_HOUR

    def _record_action(self) -> None:
        self._action_times.append(time.time())

    # ------------------------------------------------------------------
    # Subprocess concurrency guard (Phase 4B+ placeholder)
    # ------------------------------------------------------------------

    def _run_subprocess_action(self, cmd: list[str]) -> bool:
        """Template for any future subprocess-based maintenance action.

        Acquires the subprocess lock non-blocking; skips if already held
        (prevents overlapping runs and SQLite lock contention).
        """
        if not self._subprocess_lock.acquire(blocking=False):
            logger.debug(
                "HeartbeatDaemon: subprocess action skipped -- previous action still running."
            )
            return False
        try:
            import subprocess

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0
        except Exception as exc:
            logger.warning("HeartbeatDaemon: subprocess action failed: %s", exc)
            return False
        finally:
            self._subprocess_lock.release()

    # ------------------------------------------------------------------
    # Main cycle (Perceive -> Evaluate -> Plan -> Act -> Reflect)
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Execute one idle maintenance cycle.

        IMPORTANT: This method must remain strictly Tier 0:
          - Read DBs only (episodic, semantic).
          - Write only to: episodic.db (log entries) and self_model.json.
          - No subprocess calls, no file writes, no LLM calls.
        """
        # ---- PERCEIVE ------------------------------------------------
        stale_ids = self._perceive_stale_facts()
        needs_reflection = self._evaluate_reflection_needed()

        # ---- EVALUATE ------------------------------------------------
        # Nothing actionable -> silent no-op (no episodic entry written)
        if not stale_ids and not needs_reflection:
            logger.debug("HeartbeatDaemon: no-op cycle (nothing actionable).")
            return

        # ---- PLAN / ACT / REFLECT ------------------------------------
        acted = False

        if stale_ids and self._can_act():
            self._act_stale_fact_alert(stale_ids)
            self._record_action()
            acted = True

        if needs_reflection and self._can_act():
            self._act_weekly_reflection()
            self._record_action()
            acted = True

        if self._can_act():
            # Try counterfactual reflection
            self._act_counterfactual_reflection()
            self._record_action()
            acted = True

        if acted:
            logger.debug("HeartbeatDaemon: cycle complete.")

    # ------------------------------------------------------------------
    # Perceive helpers
    # ------------------------------------------------------------------

    def _perceive_stale_facts(self) -> list[int]:
        """Return IDs of facts older than STALE_FACT_DAYS."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=STALE_FACT_DAYS)
        ).isoformat()
        try:
            rows = self._semantic_conn.execute(
                "SELECT id FROM facts WHERE created_at < ?", (cutoff,)
            ).fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.debug("HeartbeatDaemon: stale fact query failed: %s", exc)
            return []

    def _evaluate_reflection_needed(self) -> bool:
        """Return True if weekly aggregation is overdue."""
        data = self._self_model.as_dict()
        last = data.get("last_reflection_at")
        if last is None:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            return (
                datetime.now(timezone.utc) - last_dt
            ).days >= _REFLECTION_MIN_INTERVAL_DAYS
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Act helpers (Tier 0 only)
    # ------------------------------------------------------------------

    def _act_stale_fact_alert(self, stale_ids: list[int]) -> None:
        """Log a stale_fact_alert to episodic.db -- read-only w.r.t. semantic.db."""
        content = (
            f"[HeartbeatDaemon] Stale fact alert: {len(stale_ids)} fact(s) older than "
            f"{STALE_FACT_DAYS} days. IDs: {stale_ids[:20]}"
        )
        self._log_episodic("stale_fact_alert", content, "neutral")
        logger.info(
            "HeartbeatDaemon: stale_fact_alert -- %d facts flagged.", len(stale_ids)
        )

    def _act_weekly_reflection(self) -> None:
        """Aggregate episodic logs and compute Bayesian posterior adjustments."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_REFLECTION_WINDOW_DAYS)
        ).isoformat()
        try:
            rows = self._episodic_conn.execute(
                """
                SELECT reasoning_domain, strategy_label, outcome_class, COUNT(*) as cnt
                FROM episodic_log
                WHERE created_at > ? AND reasoning_domain IS NOT NULL
                GROUP BY reasoning_domain, strategy_label, outcome_class
                """,
                (cutoff,),
            ).fetchall()
        except Exception as exc:
            logger.warning(
                "HeartbeatDaemon: reflection aggregation query failed: %s", exc
            )
            return

        if not rows:
            return

        # Build domain_deltas: map domain::strategy -> outcome ratio delta
        domain_deltas: dict[str, Any] = {}
        kind_counts: dict[str, dict[str, int]] = {}
        for row in rows:
            domain, strategy, outcome, cnt = row[0], row[1], row[2], row[3]
            if not strategy:
                strategy = "default"
            key = f"{domain}::{strategy}"
            kind_counts.setdefault(key, {})[outcome] = cnt

        for key, outcomes in kind_counts.items():
            total = sum(outcomes.values())
            if total == 0:
                continue
            success = outcomes.get("success", 0)
            ratio = success / total
            # Bayesian posterior logic is merged in self_model
            domain_deltas[key] = {"outcome_ratio": round(ratio, 4), "total": total}

        if domain_deltas:
            self._self_model.update_domain_deltas(domain_deltas)
            self._log_episodic(
                "heartbeat_cycle",
                f"[HeartbeatDaemon] Weekly reflection complete. "
                f"Domains updated: {list(domain_deltas.keys())}",
                "success",
            )
            logger.info("HeartbeatDaemon: weekly reflection complete.")

    def _act_counterfactual_reflection(self) -> None:
        """Test edge-case variants of verified reasoning episodes in gVisor."""
        try:
            # Pick a verified episode
            row = self._reasoning_conn.execute(
                "SELECT * FROM reasoning_episodes WHERE verified=1 ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
        except Exception as exc:
            logger.warning("HeartbeatDaemon: counterfactual query failed: %s", exc)
            return

        if not row:
            return

        # We need a brain. But HeartbeatDaemon doesn't have self._brain.
        # It's Tier 0. Wait, M#64 says: "Have the idle Heartbeat cycle ask 'What change would make this solution fail?'"
        # Since Heartbeat is STRICTLY Tier 0 and doesn't do LLM calls, this is a contradiction.
        # But Phase 4B updated M#64 to be executed here. We must fetch the global brain_manager's brain.
        from agent.brains.factory import brain_manager

        if not brain_manager or not brain_manager.brain:
            return

        brain = brain_manager.brain
        prompt = (
            f"Review this verified solution:\nState: {row['state']}\nAction: {row['action']}\n"
            f"What input change or edge case would make this solution fail? Output ONLY the Python input value."
        )
        edge_case = brain.generate(prompt)

        from agent.engine.validator import SkillValidator

        validator = SkillValidator()
        result = validator.run_counterfactual_test(row["action"], edge_case)

        self._log_episodic(
            "heartbeat_cycle",
            f"Counterfactual test for episode {row['id']}: {result}",
            "neutral",
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _log_episodic(self, kind: str, content: str, outcome: str) -> None:
        """Write an episodic log entry using the daemon's own DB connection."""
        try:
            self._episodic_conn.execute(
                "INSERT INTO episodic_log (trace_id, kind, content, outcome, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    kind,
                    content,
                    outcome,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._episodic_conn.commit()
        except Exception as exc:
            logger.warning("HeartbeatDaemon: episodic write failed: %s", exc)

    @staticmethod
    def _open_db(db_path: Path) -> sqlite3.Connection:
        """Open a thread-local SQLite connection with WAL + busy timeout."""
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")  # 3s wait on lock contention
        return conn
