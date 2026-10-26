"""Perception backends: deterministic hash (default) and sentence-transformers."""
from __future__ import annotations

import hashlib
import re
from typing import List, Protocol, runtime_checkable

import numpy as np

DIM = 128
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@runtime_checkable
class Perceiver(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


class PerceptionLayer:
    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            idx = int.from_bytes(hashlib.md5(tok.encode()).digest(), "little") % self.dim
            vec[idx] += 1.0
        n = float(np.linalg.norm(vec))
        return vec / n if n else vec

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class SentenceTransformerPerception:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME, fallback: bool = True) -> None:
        self._fallback = PerceptionLayer()
        self._model = None
        self._dim = 384
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            if not fallback:
                raise

    @property
    def dim(self) -> int:
        return self._dim if self._model else self._fallback.dim

    @property
    def using_semantic_model(self) -> bool:
        return self._model is not None

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            return self._fallback.embed(text)
        return np.asarray(
            self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float32,
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


def make_perception(backend: str = "hash", **kw) -> Perceiver:
    if backend in ("hash", "default"):
        return PerceptionLayer(**kw)
    if backend in ("sentence-transformers", "st", "semantic"):
        return SentenceTransformerPerception(**kw)
    raise ValueError(f"unknown perception backend: {backend!r}")