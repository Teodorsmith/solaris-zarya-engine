"""Groq integration via direct httpx REST calls (OpenAI-compatible schema)."""
import time
import random
import threading
import httpx
import logging
from typing import Optional

from agent.brains.base import BaseBrain
from agent.brains.gemini_brain import BrainError  # Reuse the same error type

logger = logging.getLogger(__name__)

class GroqBrain(BaseBrain):
    PREFERRED_MODELS = [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen3.6-27b",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self, api_key: str, model: str = "auto", rpm_limit: int = 30):
        import os
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self.base_url = "https://api.groq.com/openai/v1"
        self.url = f"{self.base_url}/chat/completions"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        self.timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)
        self.max_retries = 3
        self.retryable_statuses = {408, 429, 500, 502, 503, 504}
        
        configured_model = os.getenv("GROQ_MODEL", model).strip()
        if configured_model.lower() in {"auto", ""}:
            self.model = self._discover_best_model()
            logger.info(f"Groq auto-discovery selected model: {self.model}")
        else:
            self.model = configured_model

    def _discover_best_model(self) -> str:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/models", headers=self.headers)
                if res.status_code == 200:
                    available = {m["id"] for m in res.json().get("data", [])}
                    for candidate in self.PREFERRED_MODELS:
                        if candidate in available:
                            return candidate
                    if available:
                        return next(iter(available))
        except Exception as e:
            logger.warning(f"Groq auto-discovery failed: {e}")
        return "llama-3.3-70b-versatile"

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

    def _get_fallback_brain(self) -> Optional[BaseBrain]:
        import os
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            from agent.brains.gemini_brain import GeminiBrain
            model = os.getenv("GEMINI_MODEL", "auto")
            rpm = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
            return GeminiBrain(api_key=gemini_key, model=model, rpm_limit=rpm)
            
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from agent.brains.openai_brain import OpenAIBrain
            model = os.getenv("OPENAI_MODEL", "auto")
            rpm = int(os.getenv("OPENAI_RPM_LIMIT", "500"))
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAIBrain(api_key=openai_key, model=model, rpm_limit=rpm, base_url=base_url)
            
        return None

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 4096
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            self._apply_rate_limit()

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.url, headers=self.headers, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if not choices:
                            raise BrainError(f"Groq returned empty choices: {data}")
                        
                        message = choices[0].get("message", {})
                        if "content" not in message:
                            raise BrainError(f"Malformed response structure: {data}")
                        
                        return self._extract_final_answer(message["content"])

                    if response.status_code in self.retryable_statuses:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_duration = float(retry_after)
                        else:
                            sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                        
                        # Provider Failover on long rate limit pause
                        if response.status_code == 429 and sleep_duration > 30.0:
                            fallback = self._get_fallback_brain()
                            if fallback:
                                logger.warning(
                                    f"Groq rate limit delay ({sleep_duration:.1f}s > 30s). "
                                    f"Failing over to {fallback.__class__.__name__}..."
                                )
                                return fallback.generate(prompt)
                            else:
                                raise BrainError(
                                    f"Groq 429 rate limit reached ({sleep_duration:.1f}s wait required). "
                                    "No fallback provider (GEMINI_API_KEY/OPENAI_API_KEY) found."
                                )

                        logger.warning(f"Groq API returned {response.status_code}. Retrying in {sleep_duration:.2f}s...")
                        time.sleep(sleep_duration)
                        continue
                    
                    logger.error(f"HTTPStatusError from Groq: {response.text}")
                    response.raise_for_status()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"Network error ({exc.__class__.__name__}). Retrying in {sleep_duration:.2f}s...")
                time.sleep(sleep_duration)

        # Retries exhausted -> attempt fallback before throwing error
        fallback = self._get_fallback_brain()
        if fallback:
            logger.warning(f"Groq API retries exhausted. Failing over to {fallback.__class__.__name__}...")
            return fallback.generate(prompt)

        raise BrainError(f"Groq API failed after {self.max_retries} retries. Last error: {last_error}")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings should be handled by FastEmbed via EmbeddingEngine, not Groq.")
