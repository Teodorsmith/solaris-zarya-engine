"""Brain factory for instantiating the right LLM provider."""
import os
import warnings

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

# Simple registry for future expansion
BRAIN_BUILDERS = {
    "gemini": _build_gemini_brain,
    "groq": _build_groq_brain,
    "openai": _build_openai_brain,
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
