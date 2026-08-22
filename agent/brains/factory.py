"""Brain factory for instantiating the right LLM provider."""
from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)

from agent.brains.base import BaseBrain
from agent.brains.mock_brain import MockBrain
from agent.memory.embeddings import EmbeddingEngine


def _build_mock_brain(embedder: EmbeddingEngine | None) -> BaseBrain:
    return MockBrain(embedder=embedder)


def _build_gemini_brain(embedder: EmbeddingEngine | None) -> BaseBrain:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        warnings.warn("No GEMINI_API_KEY or GOOGLE_API_KEY found. Falling back to MockBrain.")
        return _build_mock_brain(embedder)
        
    model = os.getenv("GEMINI_MODEL", "auto")
    rpm_limit = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
    
    # Lazy import to avoid httpx requirement if not using Gemini
    from agent.brains.gemini_brain import GeminiBrain
    return GeminiBrain(api_key=api_key, model=model, rpm_limit=rpm_limit)


def _build_groq_brain(embedder) -> BaseBrain:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        warnings.warn("No GROQ_API_KEY found. Falling back to MockBrain.")
        return _build_mock_brain(embedder)
        
    model = os.getenv("GROQ_MODEL", "auto")
    rpm_limit = int(os.getenv("GROQ_RPM_LIMIT", "30"))
    
    from agent.brains.groq_brain import GroqBrain
    return GroqBrain(api_key=api_key, model=model, rpm_limit=rpm_limit)

def _build_openai_brain(embedder) -> BaseBrain:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        warnings.warn("No OPENAI_API_KEY found. Falling back to MockBrain.")
        return _build_mock_brain(embedder)
        
    model = os.getenv("OPENAI_MODEL", "auto")
    rpm_limit = int(os.getenv("OPENAI_RPM_LIMIT", "500"))
    base_url = os.getenv("OPENAI_BASE_URL")
    
    from agent.brains.openai_brain import OpenAIBrain
    return OpenAIBrain(api_key=api_key, model=model, rpm_limit=rpm_limit, base_url=base_url)


def _build_local_brain(
    embedder,
    model: str = "auto",
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseBrain:
    """Build an OpenAILikeBrain for local providers (Ollama, LM Studio, vLLM).

    Never fails startup — if /models is unavailable the brain still boots and
    falls back to the 'default' model name.
    """
    api_key = (
        os.getenv("LOCAL_LLM_API_KEY")
        or os.getenv("OPENAI_LIKE_API_KEY")
        or "ollama"
    )
    from agent.brains.openai_like_brain import OpenAILikeBrain
    brain = OpenAILikeBrain(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.info(
        "Local brain ready: provider=local base_url=%s model=%s",
        brain.base_url, brain.model,
    )
    return brain


# Simple registry — each builder accepts (embedder, **kwargs)
BRAIN_BUILDERS = {
    "gemini": _build_gemini_brain,
    "groq": _build_groq_brain,
    "openai": _build_openai_brain,
    "local": _build_local_brain,
    "mock": _build_mock_brain,
}


def get_brain(embedder: EmbeddingEngine | None = None) -> BaseBrain:
    """Instantiates the requested brain, falling back to mock safely."""
    brain_type = os.getenv("AI_BRAIN", "mock").lower()

    builder = BRAIN_BUILDERS.get(brain_type)
    if not builder:
        warnings.warn(f"Unknown AI_BRAIN '{brain_type}'. Falling back to MockBrain.")
        builder = BRAIN_BUILDERS["mock"]

    return builder(embedder)


class BrainManager:
    """Hot-swap the active brain at runtime without restarting the agent.

    Preserves the same fallback contract as ``get_brain()``:

    * Unknown provider → warning + MockBrain (no crash).
    * Missing API key  → warning + MockBrain.
    * ``"mock"``       → always succeeds.
    * ``"local"``      → never fails startup even if /models is down.
    """

    def __init__(
        self,
        embedder: EmbeddingEngine | None = None,
        brain: BaseBrain | None = None,
    ) -> None:
        self._embedder = embedder
        self._brain: BaseBrain = brain if brain is not None else get_brain(embedder)

    @property
    def brain(self) -> BaseBrain:
        return self._brain

    def switch_brain(
        self,
        provider: str,
        *,
        model: str = "auto",
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BaseBrain:
        """Switch the active brain to *provider*.

        Args:
            provider:    One of "gemini", "groq", "openai", "local", "mock".
            model:       Model name / tag.  Pass ``"auto"`` for discovery.
            base_url:    Override base URL (local only).
            temperature: Override generation temperature.
            max_tokens:  Override max output tokens.

        Returns:
            The newly active brain (also stored in ``self.brain``).
        """
        provider = provider.strip().lower()
        builder = BRAIN_BUILDERS.get(provider)

        if builder is None:
            warnings.warn(
                f"BrainManager: unknown provider '{provider}'. "
                "Falling back to MockBrain."
            )
            logger.warning("brain switch: unknown provider=%s — using mock", provider)
            self._brain = _build_mock_brain(self._embedder)
            return self._brain

        try:
            if provider == "local":
                new_brain = _build_local_brain(
                    self._embedder,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif provider == "mock":
                new_brain = _build_mock_brain(self._embedder)
            else:
                # Existing cloud builders only accept (embedder); model/key come
                # from env vars.  Future builders may accept kwargs.
                new_brain = builder(self._embedder)

            # Safe log — never print base_url query strings or Bearer tokens.
            safe_url = (base_url or "").split("?")[0] or "env"
            logger.info(
                "brain switch: provider=%s base_url=%s model=%s",
                provider,
                new_brain.base_url if hasattr(new_brain, "base_url") else safe_url,
                getattr(new_brain, "model", model),
            )
            self._brain = new_brain
            return new_brain

        except Exception as exc:
            warnings.warn(
                f"BrainManager: failed to build '{provider}' brain ({exc}). "
                "Falling back to MockBrain."
            )
            logger.warning(
                "brain switch: provider=%s failed (%s) — using mock",
                provider, exc,
            )
            self._brain = _build_mock_brain(self._embedder)
            return self._brain

    def list_available(self) -> list[str]:
        """Return known provider names from the registry."""
        return list(BRAIN_BUILDERS.keys())
