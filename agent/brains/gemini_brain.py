"""Google Gemini integration via direct httpx REST calls with dynamic model discovery."""
from __future__ import annotations

import os
import time
import httpx
import logging
from typing import Optional
from agent.brains.base import BaseBrain

logger = logging.getLogger(__name__)


class BrainError(Exception):
    """Structured error for brain failures (e.g. safety blocks, API limits)."""
    pass


class GeminiBrain(BaseBrain):
    PREFERRED_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def __init__(self, api_key: str, model: str = "auto", rpm_limit: int = 15):
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)

        # Resolve model dynamically if set to 'auto' or empty
        configured_model = os.getenv("GEMINI_MODEL", model).strip()
        if configured_model.lower() in {"auto", ""}:
            self.model = self._discover_best_model()
            logger.info(f"Gemini auto-discovery selected model: {self.model}")
        else:
            self.model = configured_model.removeprefix("models/")

    def _discover_best_model(self) -> str:
        """Fetch available models for this Gemini API key and pick the best available."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/models?key={self.api_key}")
                if res.status_code == 200:
                    data = res.json()
                    # Filter for models that support text generation (generateContent)
                    available = {
                        m["name"].removeprefix("models/")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    }

                    # 1. Match against preferred priority order
                    for candidate in self.PREFERRED_MODELS:
                        if candidate in available:
                            return candidate

                    # 2. Fallback to any model containing 'flash' or 'pro'
                    for candidate in available:
                        if "flash" in candidate or "pro" in candidate:
                            return candidate

                    # 3. Fallback to first available candidate
                    if available:
                        return next(iter(available))
        except Exception as e:
            logger.warning(f"Gemini auto-discovery failed: {e}")

        # Safe static fallback if discovery fails or network is offline
        return "gemini-2.5-flash"

    def _rate_limit_wait(self) -> None:
        """Enforce spacing between calls to stay within RPM bounds."""
        if self.rpm_limit <= 0:
            return
        min_interval = 60.0 / max(1, self.rpm_limit)
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """Generate text with automated 429/503 retry backoff."""
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
            }
        }

        backoff = 2.0
        for attempt in range(max_retries):
            self._rate_limit_wait()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise BrainError("Gemini returned empty candidates list.")

                        candidate = candidates[0]
                        finish_reason = candidate.get("finishReason")
                        if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST"}:
                            raise BrainError(f"Gemini generation blocked by policy: {finish_reason}")

                        parts = candidate.get("content", {}).get("parts", [])
                        if not parts:
                            raise BrainError("Gemini candidate contains no content parts.")

                        return parts[0].get("text", "").strip()

                    elif response.status_code in (429, 503):
                        logger.warning(f"Gemini API returned {response.status_code}. Retrying in {backoff:.2f}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2.0, 30.0)
                        continue

                    else:
                        raise BrainError(
                            f"HTTP {response.status_code} from Gemini: {response.text}"
                        )

            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise BrainError(f"Network error calling Gemini: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

        raise BrainError(f"Failed to generate output from Gemini after {max_retries} retries.")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings should be handled by FastEmbed via EmbeddingEngine, not Gemini.")
