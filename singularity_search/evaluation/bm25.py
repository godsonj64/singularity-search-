"""A compact BM25 implementation for baseline evaluation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from singularity_search.evaluation.encoders import tokenize


@dataclass
class BM25Index:
    documents: list[str]
    doc_ids: list[str]
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if len(self.documents) != len(self.doc_ids):
            raise ValueError("documents and doc_ids must have the same length.")
        if not self.documents:
            raise ValueError("documents must not be empty.")
        self.doc_tokens = [tokenize(doc) for doc in self.documents]
        self.doc_len = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_len)) if len(self.doc_len) else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        out = np.zeros(len(self.documents), dtype=np.float64)
        if not query_terms:
            return out
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            dl = self.doc_len[i]
            denom_const = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12))
            for term in query_terms:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                score += self.idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_const)
            out[i] = score
        return out

    def search(self, query: str, top_k: int = 100) -> list[str]:
        scores = self.scores(query)
        limit = min(top_k, len(scores))
        order = np.argsort(-scores)[:limit]
        return [self.doc_ids[int(i)] for i in order]
