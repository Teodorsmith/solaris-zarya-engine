# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.memory.episodic import EpisodicMemory
from agent.models import EpisodicLog


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.db_path = self._tmp / "episodic.db"
        self.memory = EpisodicMemory(self.db_path)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_schema_migration(self):
        # Insert raw without new columns
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO episodic_log (trace_id, kind, content, outcome, created_at) VALUES ('123', 'system', 'test', 'success', '2026')"
            )

        # Read back using API
        logs = self.memory.get_trace("123")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].prompt_hash, None)
        self.assertEqual(logs[0].strategy_label, None)
        self.assertEqual(logs[0].hypothesis_count, 1)

    def test_log_new_telemetry(self):
        log = EpisodicLog(
            kind="query",
            content="test",
            prompt_hash="abc",
            strategy_label="tree_of_thought",
            novelty_score=0.85,
            reasoning_domain="math",
            outcome_class="success",
            hypothesis_count=3,
        )
        self.memory.log_event(log)

        retrieved = self.memory.recent(1)[0]
        self.assertEqual(retrieved.prompt_hash, "abc")
        self.assertEqual(retrieved.strategy_label, "tree_of_thought")
        self.assertEqual(retrieved.novelty_score, 0.85)
        self.assertEqual(retrieved.reasoning_domain, "math")
        self.assertEqual(retrieved.outcome_class, "success")
        self.assertEqual(retrieved.hypothesis_count, 3)


if __name__ == "__main__":
    unittest.main()
