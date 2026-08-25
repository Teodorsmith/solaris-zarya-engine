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
import tempfile
import unittest
from pathlib import Path

from agent.memory.reasoning import ReasoningMemory
from agent.models import ReasoningEpisode


class TestReasoningMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.db_path = self._tmp / "reasoning.db"
        self.memory = ReasoningMemory(self.db_path)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_episode(self, outcome="failure", domain="math", verified=False):
        return ReasoningEpisode(
            state="2+2",
            hypothesis="5",
            action="submit",
            observation="wrong",
            outcome_class=outcome,
            reasoning_domain=domain,
            verified=verified,
        )

    def test_log_and_retrieve_failures(self):
        self.memory.log_episode(self._make_episode(outcome="success"))
        self.memory.log_episode(self._make_episode(outcome="failure", domain="math"))
        self.memory.log_episode(self._make_episode(outcome="failure", domain="git"))

        fails = self.memory.get_failures()
        self.assertEqual(len(fails), 2)

        math_fails = self.memory.get_failures(domain="math")
        self.assertEqual(len(math_fails), 1)
        self.assertEqual(math_fails[0].reasoning_domain, "math")

    def test_mark_verified(self):
        ep_id = self.memory.log_episode(self._make_episode())
        self.memory.mark_verified(ep_id, '{"test": 123}')

        unverified = self.memory.get_unverified()
        self.assertEqual(len(unverified), 0)

        # Verify directly in DB
        row = self.memory.conn.execute(
            "SELECT verified, srt_json FROM reasoning_episodes WHERE id=?", (ep_id,)
        ).fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], '{"test": 123}')


if __name__ == "__main__":
    unittest.main()
