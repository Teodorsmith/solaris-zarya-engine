"""Google Gemini integration via direct httpx REST calls."""
import time
import random
import threading
import httpx
import logging
from typing import Optional

from agent.brains.base import BaseBrain

logger = logging.getLogger(__name__)

class BrainError(Exception):
    """Structured error for brain failures (e.g. safety blocks)."""
    pass

class GeminiBrain(BaseBrain):
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", rpm_limit: int = 15):
        self.api_key = api_key
        self.model = model
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        
        # Explicit timeouts
        self.timeout = httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=20.0,
            pool=10.0
        )
        self.max_retries = 3
        self.retryable_statuses = {408, 429, 500, 502, 503, 504}

    def _apply_rate_limit(self) -> None:
        """Enforce strict client-side RPM pacing."""
        if self.rpm_limit <= 0:
            return
        
        with self._lock:
            min_interval = 60.0 / self.rpm_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            self._apply_rate_limit()

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.url, headers=self.headers, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise BrainError(f"Gemini returned empty candidates: {data}")
                        
                        candidate = candidates[0]
                        # Handle safety/recitation blocks gracefully
                        finish_reason = candidate.get("finishReason")
                        if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST"}:
                            raise BrainError(f"Gemini generation blocked by policy: {finish_reason}")
                        
                        parts = candidate.get("content", {}).get("parts", [])
                        if not parts or "text" not in parts[0]:
                            raise BrainError(f"Malformed response structure: {data}")
                        
                        return parts[0]["text"]

                    if response.status_code in self.retryable_statuses:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        
                        # Respect standard Retry-After header if provided
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_duration = float(retry_after)
                        else:
                            # Exponential backoff (1s, 2s, 4s...) + full jitter
                            sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                        
                        logger.warning(f"Gemini API returned {response.status_code}. Retrying in {sleep_duration:.2f}s...")
                        time.sleep(sleep_duration)
                        continue
                    
                    # Unrecoverable error (e.g. 400 Bad Request, 401 Invalid Key)
                    logger.error(f"HTTPStatusError from Gemini: {response.text}")
                    response.raise_for_status()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                sleep_duration = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"Network error ({exc.__class__.__name__}). Retrying in {sleep_duration:.2f}s...")
                time.sleep(sleep_duration)

        raise BrainError(f"Gemini API failed after {self.max_retries} retries. Last error: {last_error}")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings should be handled by FastEmbed via EmbeddingEngine, not Gemini.")
