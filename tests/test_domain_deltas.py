# Copyright (C) 2026 Teodor Smith
import shutil
import tempfile
import unittest
from pathlib import Path

from agent.memory.self_model import SelfModel


class TestDomainDeltas(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.model_path = self._tmp / "self_model.json"
        self.bak_path = self._tmp / "self_model.bak.json"
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.state_manifest import StateManifest

        self.manifest = StateManifest(self._tmp / "manifest.json")
        self.episodic = EpisodicMemory(self._tmp / "episodic.db")
        self.self_model = SelfModel(
            self.model_path, self.bak_path, self.manifest, self.episodic
        )

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_bayesian_merge_new_domain(self):
        deltas = {"math::tree_of_thought": {"outcome_ratio": 0.8, "total": 10}}
        self.self_model.update_domain_deltas(deltas)

        data = self.self_model.as_dict()
        profile = data["reasoning_profile"]["domain_deltas"]
        self.assertIn("math::tree_of_thought", profile)
        self.assertEqual(profile["math::tree_of_thought"]["outcome_ratio"], 0.8)
        self.assertEqual(profile["math::tree_of_thought"]["total"], 10)

    def test_bayesian_merge_existing_domain(self):
        # Init with 10 total, 0.5 ratio
        self.self_model.update_domain_deltas(
            {"code::default": {"outcome_ratio": 0.5, "total": 10}}
        )

        # New delta: 10 total, 1.0 ratio
        # Merged ratio should be ((0.5*10) + (1.0*10)) / 20 = 15/20 = 0.75
        self.self_model.update_domain_deltas(
            {"code::default": {"outcome_ratio": 1.0, "total": 10}}
        )

        data = self.self_model.as_dict()
        profile = data["reasoning_profile"]["domain_deltas"]
        self.assertAlmostEqual(profile["code::default"]["outcome_ratio"], 0.75)
        self.assertEqual(profile["code::default"]["total"], 20)


if __name__ == "__main__":
    unittest.main()
