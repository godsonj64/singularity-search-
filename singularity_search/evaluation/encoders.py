"""Text encoders for retrieval experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from singularity_search.utils.math_ops import l2_normalize


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _fnv1a_index(token: str, n_dims: int) -> int:
    h = 2166136261
    for ch in token:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h % n_dims


class TextEncoder(Protocol):
    name: str

    def fit(self, documents: list[str]) -> None: ...

    def encode_documents(self, documents: list[str]) -> np.ndarray: ...

    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


@dataclass
class HashingTextEncoder:
    """Dependency-light lexical hashing encoder.

    This encoder is intended as a reproducible control, not as a state-of-the-art
    semantic encoder.
    """

    n_dims: int = 384
    name: str = "hash"

    def fit(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")

    def _encode_one(self, text: str) -> np.ndarray:
        out = np.zeros(self.n_dims, dtype=np.float64)
        for token in tokenize(text):
            out[_fnv1a_index(token, self.n_dims)] += 1.0
        return out

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._encode_one(doc) for doc in documents]), axis=1)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._encode_one(q) for q in queries]), axis=1)


@dataclass
class TfidfLiteEncoder:
    """Small deterministic TF-IDF encoder without external dependencies."""

    max_features: int = 30000
    name: str = "tfidf_lite"

    def fit(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        vocab_terms = [term for term, _ in df.most_common(self.max_features)]
        self.vocab_: dict[str, int] = {term: i for i, term in enumerate(vocab_terms)}
        n_docs = len(documents)
        self.idf_ = np.ones(len(self.vocab_), dtype=np.float64)
        for term, idx in self.vocab_.items():
            self.idf_[idx] = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0

    def _encode_one(self, text: str) -> np.ndarray:
        if not hasattr(self, "vocab_"):
            raise RuntimeError("encoder must be fitted before encoding.")
        vec = np.zeros(len(self.vocab_), dtype=np.float64)
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            idx = self.vocab_.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf_[idx]
        return vec

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._encode_one(doc) for doc in documents]), axis=1)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._encode_one(q) for q in queries]), axis=1)


class SentenceTransformerEncoder:
    """Optional sentence-transformer encoder for stronger baselines."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> None:
        self.name = f"sentence_transformer:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise ImportError(
                "sentence-transformers is required for this encoder. Install with: "
                "pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        arr = self.model.encode(documents, batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float64)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        arr = self.model.encode(queries, batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float64)


def build_encoder(name: str, model_name: str | None = None) -> TextEncoder:
    key = name.strip().lower()
    if key == "hash":
        return HashingTextEncoder()
    if key in {"tfidf", "tfidf_lite"}:
        return TfidfLiteEncoder()
    if key in {"sentence-transformer", "sentence_transformer", "bge"}:
        return SentenceTransformerEncoder(model_name or "BAAI/bge-small-en-v1.5")
    raise ValueError(f"Unsupported encoder: {name}")
