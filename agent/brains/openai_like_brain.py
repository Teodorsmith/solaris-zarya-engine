"""Generic OpenAI-Compatible & Local LLM integration (Ollama, LM Studio, vLLM, OpenRouter)."""
from __future__ import annotations

import os
import time
import random
import threading
import httpx
import logging
from typing import Optional

from agent.brains.base import BaseBrain
from agent.brains.gemini_brain import BrainError

logger = logging.getLogger(__name__)

# Provider-level generation defaults — can be overridden per-instance.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096


class OpenAILikeBrain(BaseBrain):
    PREFERRED_MODELS = [
        "qwen2.5-coder:7b",
        "qwen2.5-coder",
        "deepseek-r1:7b",
        "deepseek-r1",
        "llama3.2",
        "llama3.1",
        "mistral",
        "phi3",
        "default",
    ]

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "auto",
        rpm_limit: int = 0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        raw_base = (
            base_url
            or os.getenv("LOCAL_LLM_BASE_URL")
            or os.getenv("OPENAI_LIKE_BASE_URL")
            or "http://localhost:11434/v1"
        )
        self.base_url = self._normalize_base_url(raw_base)
        self.url = f"{self.base_url}/chat/completions"

        self.api_key = (
            api_key
            or os.getenv("LOCAL_LLM_API_KEY")
            or os.getenv("OPENAI_LIKE_API_KEY")
            or "ollama"
        )
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._lock = threading.Lock()

        # Per-instance generation parameters
        self.temperature: float = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens: int = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "Bearer none",
        }

        self.timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        self.max_retries = 3
        self.retryable_statuses = {408, 429, 500, 502, 503, 504}

        configured_model = os.getenv("LOCAL_LLM_MODEL", model).strip()
        if configured_model.lower() in {"auto", ""}:
            self.model = self._discover_best_model()
            # Safe log — no keys/tokens, only endpoint and model name
            logger.info(
                "OpenAILike auto-discovery selected: provider=local base_url=%s model=%s",
                self.base_url, self.model,
            )
        else:
            self.model = configured_model

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Strip trailing slashes and ensure base_url does not end with /chat/completions."""
        cleaned = url.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            cleaned = cleaned[:-17].rstrip("/")
        return cleaned

    def list_models(self) -> list[str]:
        """Best-effort fetch of model names from /models endpoint.

        Never raises — if the server is unavailable, requires different auth,
        or returns an unexpected schema, an empty list is returned.
        Model discovery must never block agent startup.
        """
        try:
            # Short independent timeout so a slow local server can't stall the agent.
            discovery_timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
            with httpx.Client(timeout=discovery_timeout) as client:
                res = client.get(f"{self.base_url}/models", headers=self.headers)
                if res.status_code != 200:
                    logger.debug(
                        "Model discovery: HTTP %s from %s — skipping.",
                        res.status_code, self.base_url,
                    )
                    return []

                data = res.json()
                models: list[str] = []

                # 1. Standard OpenAI format: {"data": [{"id": ...}]}
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        if isinstance(item, dict) and "id" in item:
                            models.append(str(item["id"]))
                        elif isinstance(item, str):
                            models.append(item)
                # 2. Ollama / custom format: {"models": [{"name": ...} or {"id": ...}]}
                elif isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                    for item in data["models"]:
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("id") or item.get("model")
                            if name:
                                models.append(str(name))
                        elif isinstance(item, str):
                            models.append(item)
                # 3. Direct list format: [{"id": ...}] or ["model1", "model2"]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "id" in item:
                            models.append(str(item["id"]))
                        elif isinstance(item, str):
                            models.append(item)

                return models

        except Exception as exc:
            # Broad catch is intentional: connection refused, DNS failure,
            # auth errors, JSON decode errors — all are non-fatal here.
            logger.debug(
                "Model discovery failed for %s (%s) — continuing without it.",
                self.base_url, exc,
            )
            return []

    def _discover_best_model(self) -> str:
        available = self.list_models()
        if available:
            for candidate in self.PREFERRED_MODELS:
                if candidate in available:
                    return candidate
            return available[0]
        return "default"

    def _apply_rate_limit(self) -> None:
        if self.rpm_limit <= 0:
            return
        with self._lock:
            min_interval = 60.0 / self.rpm_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()

    @staticmethod
    def _extract_final_answer(raw: str) -> str:
        """Strip <think>...</think> chain-of-thought blocks if present."""
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        return raw.strip()

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            self._apply_rate_limit()

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.url, headers=self.headers, json=payload)

                    if response.status_code == 200:
                        # Robust extraction — some local models may omit fields
                        try:
                            data = response.json()
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError, ValueError) as parse_err:
                            raise BrainError(
                                f"Malformed response from provider at {self.base_url}: "
                                f"{parse_err} — raw: {response.text[:200]}"
                            ) from parse_err

                        if not content:
                            raise BrainError(
                                f"Provider returned empty content. "
                                f"model={self.model} base_url={self.base_url}"
                            )

                        return self._extract_final_answer(content)

                    if response.status_code in self.retryable_statuses:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_duration = float(retry_after)
                        else:
                            sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)

                        logger.warning(
                            "Provider returned %s. Retrying in %.2fs... (attempt %d/%d)",
                            response.status_code, sleep_duration, attempt + 1, self.max_retries,
                        )
                        time.sleep(sleep_duration)
                        continue

                    logger.error(
                        "Non-retryable HTTP %s from provider at %s",
                        response.status_code, self.base_url,
                    )
                    response.raise_for_status()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Network error (%s). Retrying in %.2fs... (attempt %d/%d)",
                    exc.__class__.__name__, sleep_duration, attempt + 1, self.max_retries,
                )
                time.sleep(sleep_duration)

        raise BrainError(f"Provider API failed after {self.max_retries} retries. Last error: {last_error}")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings should be handled by FastEmbed via EmbeddingEngine, not OpenAILikeBrain.")