# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""OpenAI integration via direct httpx REST calls."""

import logging
import random
import threading
import time

import httpx

from agent.brains.base import BaseBrain
from agent.brains.gemini_brain import BrainError  # Reuse the same error type

logger = logging.getLogger(__name__)


class OpenAIBrain(BaseBrain):
    PREFERRED_MODELS = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "auto",
        rpm_limit: int = 500,
        base_url: str | None = None,
    ):
        import os

        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._lock = threading.Lock()

        # Default to OpenAI's endpoint, but allow custom OpenRouter / OpenCode endpoints
        self.base_url = (
            base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        )
        self.base_url = self.base_url.removesuffix(
            "/chat/completions"
        )  # remove /chat/completions

        self.url = f"{self.base_url}/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)
        self.max_retries = 3
        self.retryable_statuses = {408, 429, 500, 502, 503, 504}

        configured_model = os.getenv("OPENAI_MODEL", model).strip()
        if configured_model.lower() in {"auto", ""}:
            self.model = self._discover_best_model()
            logger.info(f"OpenAI auto-discovery selected model: {self.model}")
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
            logger.warning(f"OpenAI auto-discovery failed: {e}")
        return "gpt-4o-mini"

    def _apply_rate_limit(self) -> None:
        if self.rpm_limit <= 0:
            return

        with self._lock:
            min_interval = 60.0 / self.rpm_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.2),
        }
        if kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        if "repetition_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["repetition_penalty"]

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
                            raise BrainError(f"OpenAI returned empty choices: {data}")

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
                            sleep_duration = (2.0**attempt) + random.uniform(0.5, 1.5)

                        logger.warning(
                            f"OpenAI API returned {response.status_code}. Retrying in {sleep_duration:.2f}s..."
                        )
                        time.sleep(sleep_duration)
                        continue

                    logger.error(f"HTTPStatusError from OpenAI: {response.text}")
                    response.raise_for_status()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                sleep_duration = (2.0**attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    f"Network error ({exc.__class__.__name__}). Retrying in {sleep_duration:.2f}s..."
                )
                time.sleep(sleep_duration)

        raise BrainError(
            f"OpenAI API failed after {self.max_retries} retries. Last error: {last_error}"
        )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Embeddings should be handled by FastEmbed via EmbeddingEngine, not OpenAI."
        )
