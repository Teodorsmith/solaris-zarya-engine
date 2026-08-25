"""Tests for the Brain implementations and Factory."""

import httpx
import pytest

from agent.brains.factory import get_brain
from agent.brains.gemini_brain import BrainError, GeminiBrain
from agent.brains.mock_brain import MockBrain
from agent.memory.embeddings import EmbeddingEngine


@pytest.fixture
def embedder():
    return EmbeddingEngine(force_fallback=True)


def test_factory_fallback_to_mock(embedder, monkeypatch):
    monkeypatch.setenv("AI_BRAIN", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Missing API key should cleanly fallback to MockBrain with a warning
    with pytest.warns(UserWarning, match="No GEMINI_API_KEY"):
        brain = get_brain(embedder)

    assert isinstance(brain, MockBrain)


def test_factory_loads_gemini(embedder, monkeypatch):
    monkeypatch.setenv("AI_BRAIN", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    brain = get_brain(embedder)
    assert isinstance(brain, GeminiBrain)
    assert brain.api_key == "fake_key"


def test_gemini_brain_parses_success(monkeypatch):
    brain = GeminiBrain(api_key="fake")

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]
                },
            )

    monkeypatch.setattr(httpx, "Client", MockClient)
    assert brain.generate("test") == "Hello World"


def test_gemini_brain_safety_block_raises_brainerror(monkeypatch):
    brain = GeminiBrain(api_key="fake")

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return httpx.Response(
                200, json={"candidates": [{"finishReason": "SAFETY"}]}
            )

    monkeypatch.setattr(httpx, "Client", MockClient)
    with pytest.raises(BrainError, match="policy: SAFETY"):
        brain.generate("test")


def test_gemini_brain_retry_logic(monkeypatch):
    brain = GeminiBrain(api_key="fake")

    class MockClientState:
        calls = 0

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            MockClientState.calls += 1
            if MockClientState.calls < 3:
                resp = httpx.Response(503)
                resp.request = httpx.Request("POST", "http://fake")
                return resp

            resp = httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Success"}]}}]},
            )
            resp.request = httpx.Request("POST", "http://fake")
            return resp

    monkeypatch.setattr(httpx, "Client", MockClient)

    # Monkeypatch time.sleep to run instantly in tests
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)

    result = brain.generate("test")
    assert result == "Success"
