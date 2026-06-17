"""Origin-Coupled Spectral Diffusion Search.

This module implements the mathematically corrected classical algorithm:

    ANN/candidate retrieval -> graph diffusion -> origin coupling ->
    multiplicative relevance sharpening -> top-k.

The implementation does not claim quantum speedup. It is a practical semantic
re-ranking operator grounded in graph diffusion and stable probability updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from singularity_search.classical.graph import (
    build_knn_weight_matrix,
    diffuse_probability,
    random_walk_transition,
)
from singularity_search.utils.math_ops import (
    as_1d_float_array,
    as_2d_float_array,
    cosine_matrix,
    l2_normalize,
    normalize_l1,
    softmax,
    squared_distances_to_vector,
)


@dataclass(frozen=True)
class OCSDConfig:
    """Configuration for Origin-Coupled Spectral Diffusion Search."""

    graph_k: int = 12
    diffusion_tau: float = 0.35
    diffusion_steps: int = 8
    beta: float = 12.0
    origin_count: int = 4
    origin_temperature: float = 0.35
    origin_strength: float = 0.7
    direct_weight: float = 0.65
    sharpening_strength: float = 1.25
    sharpening_steps: int = 2
    candidate_count: int | None = None

    def validate(self) -> None:
        if self.graph_k < 1:
            raise ValueError("graph_k must be at least 1.")
        if self.diffusion_tau < 0:
            raise ValueError("diffusion_tau must be non-negative.")
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be at least 1.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.origin_count < 1:
            raise ValueError("origin_count must be at least 1.")
        if self.origin_temperature <= 0:
            raise ValueError("origin_temperature must be positive.")
        if self.origin_strength < 0:
            raise ValueError("origin_strength must be non-negative.")
        if not 0 <= self.direct_weight <= 1:
            raise ValueError("direct_weight must lie in [0, 1].")
        if self.sharpening_strength < 0:
            raise ValueError("sharpening_strength must be non-negative.")
        if self.sharpening_steps < 0:
            raise ValueError("sharpening_steps must be non-negative.")
        if self.candidate_count is not None and self.candidate_count < 2:
            raise ValueError("candidate_count must be None or at least 2.")


class OriginCoupledSpectralDiffusionSearch:
    """Classical search engine for pre-computed embeddings.

    Parameters
    ----------
    embeddings:
        Matrix with shape `(n_items, embedding_dim)`.
    items:
        Optional payload list with one item per embedding. If omitted, integer
        indices are returned.
    config:
        Algorithm hyperparameters.
    """

    def __init__(
        self,
        embeddings: np.ndarray | list[list[float]],
        items: list[object] | None = None,
        config: OCSDConfig | None = None,
    ) -> None:
        self.config = config or OCSDConfig()
        self.config.validate()

        self.embeddings = as_2d_float_array(embeddings, "embeddings")
        if self.embeddings.shape[0] < 2:
            raise ValueError("at least two embeddings are required.")

        if items is None:
            self.items: list[object] = list(range(self.embeddings.shape[0]))
        else:
            if len(items) != self.embeddings.shape[0]:
                raise ValueError("items length must match number of embeddings.")
            self.items = list(items)

        self.normalized_embeddings = l2_normalize(self.embeddings, axis=1)

    def search(self, query_embedding: np.ndarray | list[float], top_k: int = 10) -> list[dict[str, object]]:
        """Return top-k results with diagnostic scores."""
        query = as_1d_float_array(query_embedding, "query_embedding")
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query_embedding dimension must match embedding dimension.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        candidate_indices = self._candidate_indices(query)
        candidate_vectors = self.embeddings[candidate_indices]

        direct_scores = cosine_matrix(query, candidate_vectors)
        psi = softmax(self.config.beta * direct_scores)

        weights = build_knn_weight_matrix(candidate_vectors, k=self.config.graph_k)
        transition = random_walk_transition(weights)
        psi = diffuse_probability(
            psi,
            transition,
            tau=self.config.diffusion_tau,
            steps=self.config.diffusion_steps,
        )

        origin_scores = self._multi_origin_scores(query, candidate_vectors)
        psi = self._apply_origin_coupling(psi, origin_scores)

        combined_scores = (
            self.config.direct_weight * direct_scores
            + (1.0 - self.config.direct_weight) * origin_scores
        )
        psi = self._sharpen(psi, combined_scores)

        order = np.argsort(-psi)[: min(top_k, len(candidate_indices))]
        results: list[dict[str, object]] = []
        for rank, local_idx in enumerate(order, start=1):
            global_idx = int(candidate_indices[local_idx])
            results.append(
                {
                    "rank": rank,
                    "index": global_idx,
                    "item": self.items[global_idx],
                    "probability": float(psi[local_idx]),
                    "direct_score": float(direct_scores[local_idx]),
                    "origin_score": float(origin_scores[local_idx]),
                    "combined_score": float(combined_scores[local_idx]),
                }
            )
        return results

    def _candidate_indices(self, query: np.ndarray) -> np.ndarray:
        direct = cosine_matrix(query, self.embeddings)
        n = self.embeddings.shape[0]
        m = self.config.candidate_count or n
        m = min(max(m, 2), n)
        if m == n:
            return np.arange(n)
        candidate = np.argpartition(-direct, kth=m - 1)[:m]
        return candidate[np.argsort(-direct[candidate])]

    def _multi_origin_scores(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        origins = self._estimate_origins(vectors)
        normalized_origins = l2_normalize(origins, axis=1)
        normalized_vectors = l2_normalize(vectors, axis=1)
        normalized_query = query / max(float(np.linalg.norm(query)), 1e-12)

        query_origin_similarity = normalized_origins @ normalized_query
        origin_attention = softmax(query_origin_similarity, temperature=self.config.origin_temperature)
        item_origin_similarity = normalized_vectors @ normalized_origins.T
        scores = item_origin_similarity @ (origin_attention * query_origin_similarity)
        return np.clip(scores, -1.0, 1.0)

    def _estimate_origins(self, vectors: np.ndarray) -> np.ndarray:
        """Estimate multi-origin fields using deterministic farthest-point seeding."""
        x = as_2d_float_array(vectors, "vectors")
        n, d = x.shape
        k = min(self.config.origin_count, n)
        mean = np.mean(x, axis=0)
        distances = squared_distances_to_vector(x, mean)
        first = int(np.argmin(distances))
        centers = [x[first]]
        chosen = {first}

        while len(centers) < k:
            center_matrix = np.asarray(centers, dtype=np.float64)
            dist_to_centers = np.min(
                np.sum((x[:, None, :] - center_matrix[None, :, :]) ** 2, axis=2), axis=1
            )
            for idx in chosen:
                dist_to_centers[idx] = -np.inf
            next_idx = int(np.argmax(dist_to_centers))
            centers.append(x[next_idx])
            chosen.add(next_idx)

        centers_arr = np.asarray(centers, dtype=np.float64)
        dist2 = np.sum((x[:, None, :] - centers_arr[None, :, :]) ** 2, axis=2)
        scale = max(float(np.median(dist2[np.isfinite(dist2)])), 1e-12)
        assignment = np.exp(-(dist2 - np.min(dist2, axis=1, keepdims=True)) / scale)
        assignment /= np.maximum(np.sum(assignment, axis=1, keepdims=True), 1e-12)

        refined = np.zeros((k, d), dtype=np.float64)
        for j in range(k):
            weights = assignment[:, j]
            denom = max(float(np.sum(weights)), 1e-12)
            refined[j] = (weights[:, None] * x).sum(axis=0) / denom
        return refined

    def _apply_origin_coupling(self, psi: np.ndarray, origin_scores: np.ndarray) -> np.ndarray:
        coupling = np.exp(self.config.origin_strength * origin_scores)
        return normalize_l1(psi * coupling)

    def _sharpen(self, psi: np.ndarray, scores: np.ndarray) -> np.ndarray:
        current = normalize_l1(psi)
        if self.config.sharpening_steps == 0 or self.config.sharpening_strength == 0:
            return current
        centered = scores - np.mean(scores)
        for _ in range(self.config.sharpening_steps):
            current = normalize_l1(current * np.exp(self.config.sharpening_strength * centered))
        return current
