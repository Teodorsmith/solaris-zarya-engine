import pytest
from unittest.mock import MagicMock
from agent.brains.base import BaseBrain, QuotaExceededError
from agent.brains.factory import BrainManager

class FakeGeminiBrain(BaseBrain):
    def generate(self, prompt: str) -> str:
        raise QuotaExceededError("Gemini quota exhausted")
    def embed(self, text: str) -> list[float]:
        return [0.1]

class FakeGroqBrain(BaseBrain):
    def generate(self, prompt: str) -> str:
        return '{"result": "groq fallback success"}'
    def embed(self, text: str) -> list[float]:
        return [0.2]

def test_brain_failover_switch():
    gemini = FakeGeminiBrain()
    manager = BrainManager(brain=gemini)
    
    # Mock the internal builders directly to avoid environment checks during testing
    import agent.brains.factory
    original_builders = agent.brains.factory.BRAIN_BUILDERS.copy()
    
    agent.brains.factory.BRAIN_BUILDERS["gemini"] = lambda e: FakeGeminiBrain()
    agent.brains.factory.BRAIN_BUILDERS["groq"] = lambda e: FakeGroqBrain()
    
    try:
        assert manager.brain.__class__.__name__ == "FakeGeminiBrain"
        # Trigger failover
        next_brain = manager.switch_to_next_available()
        assert next_brain.__class__.__name__ == "FakeGroqBrain"
        assert manager.brain.__class__.__name__ == "FakeGroqBrain"
        
        # Test generation with new brain
        result = manager.brain.generate("test")
        assert "groq fallback success" in result
    finally:
        agent.brains.factory.BRAIN_BUILDERS = original_builders
