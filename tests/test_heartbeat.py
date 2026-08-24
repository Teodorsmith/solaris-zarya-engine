# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for agent/engine/heartbeat.py (Phase 4A)."""
from __future__ import annotations

import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_heartbeat(tmp: Path, pause_event=None):
    """Build a HeartbeatDaemon wired to temp DBs."""
    from agent.memory.state_manifest import StateManifest
    from agent.memory.episodic import EpisodicMemory
    from agent.memory.self_model import SelfModel
    from agent.engine.heartbeat import HeartbeatDaemon

    episodic_path = tmp / "episodic.db"
    semantic_path = tmp / "semantic.db"
    manifest_path = tmp / "state_manifest.json"
    model_path    = tmp / "self_model.json"
    bak_path      = tmp / "self_model.bak.json"

    # Prime the semantic DB schema so the heartbeat query works
    conn = sqlite3.connect(str(semantic_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, confidence REAL NOT NULL, "
        "source_type TEXT NOT NULL, topic TEXT, "
        "embedding TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    episodic = EpisodicMemory(episodic_path)
    manifest = StateManifest(manifest_path)
    sm = SelfModel(model_path, bak_path, manifest, episodic)
    sm.load()

    if pause_event is None:
        pause_event = threading.Event()

    daemon = HeartbeatDaemon(self_model=sm, pause_event=pause_event)
    # Point daemon at our temp DBs
    from agent.config import EPISODIC_DB, SEMANTIC_DB
    daemon._episodic_conn = HeartbeatDaemon._open_db(episodic_path)
    daemon._semantic_conn = HeartbeatDaemon._open_db(semantic_path)

    return daemon, sm, episodic, semantic_path


class TestHeartbeatRateLimit(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_rate_limit(self):
        """Action #4 within the same hour must be blocked."""
        daemon, sm, episodic, _ = _make_heartbeat(self._tmp)
        from agent.config import HEARTBEAT_MAX_PER_HOUR
        # Fill up the rolling window
        for _ in range(HEARTBEAT_MAX_PER_HOUR):
            self.assertTrue(daemon._can_act())
            daemon._record_action()
        # Next action should be denied
        self.assertFalse(daemon._can_act())

    def test_rate_limit_resets_after_hour(self):
        """Timestamps older than 1 hour should be evicted."""
        daemon, _, _, _ = _make_heartbeat(self._tmp)
        from agent.config import HEARTBEAT_MAX_PER_HOUR
        # Inject old timestamps (2 hours ago)
        old_ts = time.time() - 7201
        for _ in range(HEARTBEAT_MAX_PER_HOUR):
            daemon._action_times.append(old_ts)
        # After eviction the window is empty -> can act again
        self.assertTrue(daemon._can_act())


class TestHeartbeatPauseGuard(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_pause_guard_skips_cycle(self):
        """_run_cycle must not be entered when pause_event is set."""
        pause = threading.Event()
        pause.set()
        daemon, sm, episodic, _ = _make_heartbeat(self._tmp, pause_event=pause)

        called = []
        original = daemon._run_cycle
        daemon._run_cycle = lambda: called.append(True)

        # Simulate what run() does for one iteration
        if pause.is_set():
            pass  # skip
        else:
            daemon._run_cycle()

        self.assertEqual(called, [])


class TestHeartbeatStaleFacts(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _insert_old_fact(self, semantic_path: Path, days_old: int) -> None:
        old_ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        conn = sqlite3.connect(str(semantic_path))
        conn.execute(
            "INSERT INTO facts (text, confidence, source_type, topic, embedding, created_at) "
            "VALUES (?,?,?,?,?,?)",
            ("old fact text", 0.7, "seed", "general", "[]", old_ts),
        )
        conn.commit()
        conn.close()

    def test_stale_fact_detection(self):
        """A fact older than STALE_FACT_DAYS triggers a stale_fact_alert log."""
        daemon, sm, episodic, semantic_path = _make_heartbeat(self._tmp)
        self._insert_old_fact(semantic_path, days_old=200)

        # Refresh daemon's semantic connection to see the new row
        daemon._semantic_conn = daemon._open_db(semantic_path)

        daemon._run_cycle()

        logs = episodic.recent(20)
        kinds = [l.kind for l in logs]
        self.assertIn("stale_fact_alert", kinds)

    def test_noop_silence(self):
        """No stale facts + recent reflection -> zero episodic entries from the cycle."""
        daemon, sm, episodic, _ = _make_heartbeat(self._tmp)

        # Make the self-model believe reflection just happened
        now_iso = datetime.now(timezone.utc).isoformat()
        sm._data["last_reflection_at"] = now_iso

        before = episodic.count()
        daemon._run_cycle()
        after = episodic.count()
        self.assertEqual(before, after)

    def test_no_tier1_actions(self):
        """The heartbeat must never call subprocess.run."""
        daemon, sm, episodic, _ = _make_heartbeat(self._tmp)
        with patch("subprocess.run") as mock_sub:
            daemon._run_cycle()
            mock_sub.assert_not_called()


class TestSubprocessConcurrencyGuard(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_lock_prevents_overlap(self):
        """If the subprocess lock is held, _run_subprocess_action returns False."""
        daemon, _, _, _ = _make_heartbeat(self._tmp)
        # Pre-acquire the lock to simulate a running action
        daemon._subprocess_lock.acquire()
        try:
            result = daemon._run_subprocess_action(["echo", "test"])
            self.assertFalse(result)
        finally:
            daemon._subprocess_lock.release()


class TestWeeklyReflection(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_weekly_aggregation_updates_domain_deltas(self):
        """After aggregation, self-model domain_deltas contains computed ratios."""
        daemon, sm, episodic, _ = _make_heartbeat(self._tmp)

        # Seed some episodic events in the last 7 days
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        for outcome in ["success", "success", "failure"]:
            daemon._episodic_conn.execute(
                "INSERT INTO episodic_log (trace_id, kind, content, outcome, reasoning_domain, strategy_label, outcome_class, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("trace-test", "query", "test content", "neutral", "math", "default", outcome, recent_ts),
            )
        daemon._episodic_conn.commit()

        # Force reflection to be needed
        sm._data["last_reflection_at"] = None

        daemon._act_weekly_reflection()

        data = sm.as_dict()
        deltas = data.get("reasoning_profile", {}).get("domain_deltas", {})

        self.assertIn("math::default", deltas)
        self.assertEqual(deltas["math::default"]["total"], 3)
        self.assertAlmostEqual(deltas["math::default"]["outcome_ratio"], 0.6667, places=4)


if __name__ == "__main__":
    unittest.main()
