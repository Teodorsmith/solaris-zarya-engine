# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Local embedding engine via FastEmbed (ONNX, no PyTorch/GPU needed).

Falls back to a deterministic, non-semantic hash embedding if FastEmbed
can't load its model (no internet on first run, offline CI, sandboxed
dev environments — confirmed to happen in at least one real environment:
FastEmbed needs huggingface.co, which isn't always reachable). The
fallback keeps the whole system structurally testable; it just won't do
real semantic matching, only lexical overlap. Force it explicitly with
EmbeddingEngine(force_fallback=True) for fast, deterministic, offline
unit tests — you want that regardless of network access, so your test
suite doesn't depend on downloading model weights.
"""

from __future__ import annotations

import hashlib
import math
import warnings

from agent.config import EMBEDDING_DIM, EMBEDDING_MODEL


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _fallback_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic, dependency-free stand-in: same text -> same vector,
    via a hashed bag-of-words. NOT semantically meaningful — it won't
    recognize paraphrases the way a real embedding model does, only
    literal shared words. Exists purely so retrieval logic is testable
    without a model download."""
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    return _l2_normalize(vec)


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL, force_fallback: bool = False):
        self.model_name = model_name
        self._model = None
        self.using_fallback = force_fallback

        if not force_fallback:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=model_name)
            except Exception as e:  # noqa: BLE001 - any failure here means "use the fallback"
                warnings.warn(
                    f"FastEmbed unavailable ({e!r}); falling back to a non-semantic "
                    f"hash embedding. Retrieval will still run, but similarity scores "
                    f"won't reflect real meaning until FastEmbed can reach its model host.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.using_fallback = True

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = next(iter(self._model.embed([text])))
            return _l2_normalize(vec.tolist())
        return _fallback_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [_l2_normalize(v.tolist()) for v in self._model.embed(texts)]
        return [_fallback_embed(t) for t in texts]

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        # Both vectors are L2-normalized, so dot product == cosine similarity.
        return sum(x * y for x, y in zip(a, b))
