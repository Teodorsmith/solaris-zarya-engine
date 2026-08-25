# Copyright (C) 2026 Teodor Smith
import unittest
from unittest.mock import MagicMock

from agent.brains.mock_brain import MockBrain
from agent.engine.critic import CriticSession


class DummyEmbedder:
    def __init__(self, sim_value=1.0):
        self.sim_value = sim_value

    def embed(self, text):
        return [1.0]  # dummy

    def _cosine_similarity(self, a, b):
        return self.sim_value


class TestCriticSession(unittest.TestCase):
    def setUp(self):
        self.brain_a = MockBrain()
        self.brain_b = MockBrain()
        self.memory = MagicMock()

    def test_consensus_path(self):
        # Override cosine_similarity via monkeypatch on the session instance
        embedder = DummyEmbedder(sim_value=0.9)
        session = CriticSession(self.brain_a, self.brain_b, embedder, self.memory)
        session._cosine_similarity = lambda a, b: 0.9

        res = session.solve("What is 2+2?")
        self.assertEqual(res.verdict, "consensus")
        self.assertIsNone(res.episode)
        self.memory.log_episode.assert_not_called()

    def test_divergent_path(self):
        embedder = DummyEmbedder(sim_value=0.5)
        session = CriticSession(self.brain_a, self.brain_b, embedder, self.memory)
        session._cosine_similarity = lambda a, b: 0.5

        # We expect a divergence, so the arbiter will be called.
        res = session.solve("What is 2+2?")
        self.assertEqual(res.verdict, "divergent")
        self.assertIsNotNone(res.episode)
        self.assertEqual(res.episode.hypothesis_count, 2)
        self.assertTrue(self.memory.log_episode.called)

    def test_single_provider_fallback(self):
        embedder = DummyEmbedder(sim_value=0.5)
        # brain_b is None
        session = CriticSession(self.brain_a, None, embedder, self.memory)
        self.assertTrue(session._is_fallback)
        self.assertEqual(session.brain_b, self.brain_a)


if __name__ == "__main__":
    unittest.main()
