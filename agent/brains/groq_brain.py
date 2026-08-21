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
    def __init__(self, api_key: str, model: str = "qwen/qwen3.6-27b", rpm_limit: int = 30):
        self.api_key = api_key
        self.model = model
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        self.timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)
        self.max_retries = 3
        self.retryable_statuses = {408, 429, 500, 502, 503, 504}

    def _apply_rate_limit(self) -> None:
        if self.rpm_limit <= 0:
            return
        
        with self._lock:
            min_interval = 60.0 / self.rpm_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192
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
                        
                        return message["content"]

                    if response.status_code in self.retryable_statuses:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_duration = float(retry_after)
                        else:
                            sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                        
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

        raise BrainError(f"Groq API failed after {self.max_retries} retries. Last error: {last_error}")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings should be handled by FastEmbed via EmbeddingEngine, not Groq.")
