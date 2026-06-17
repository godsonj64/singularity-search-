"""Graph construction and random-walk diffusion."""

from __future__ import annotations

import numpy as np

from singularity_search.utils.math_ops import as_2d_float_array, normalize_l1


def build_knn_weight_matrix(
    vectors: np.ndarray,
    k: int = 12,
    sigma: float | None = None,
) -> np.ndarray:
    """Build a symmetric k-nearest-neighbour Gaussian weight matrix.

    This pure NumPy implementation is intended for small and medium candidate
    pools. For very large corpora, use an external ANN index first, then apply
    this re-ranking method to the retrieved candidate pool.
    """
    x = as_2d_float_array(vectors, "vectors")
    n = x.shape[0]
    if n < 2:
        raise ValueError("at least two vectors are required.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    k_eff = min(k, n - 1)

    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(sq_norm + sq_norm.T - 2.0 * (x @ x.T), 0.0)
    np.fill_diagonal(dist2, np.inf)

    finite_distances = dist2[np.isfinite(dist2)]
    if sigma is None:
        sigma = float(np.sqrt(np.median(finite_distances))) if finite_distances.size else 1.0
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0

    weights = np.zeros((n, n), dtype=np.float64)
    nearest = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
    for i in range(n):
        for j in nearest[i]:
            weights[i, j] = np.exp(-dist2[i, j] / (sigma * sigma))

    weights = np.maximum(weights, weights.T)
    np.fill_diagonal(weights, 0.0)
    return weights


def random_walk_transition(weights: np.ndarray) -> np.ndarray:
    """Return row-stochastic transition matrix P = D^{-1} W."""
    w = as_2d_float_array(weights, "weights")
    if w.shape[0] != w.shape[1]:
        raise ValueError("weights must be square.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    degree = np.sum(w, axis=1)
    p = np.zeros_like(w, dtype=np.float64)
    nonzero = degree > 1e-12
    p[nonzero] = w[nonzero] / degree[nonzero, None]
    p[~nonzero, ~nonzero] = 1.0
    return p


def diffuse_probability(
    psi: np.ndarray,
    transition: np.ndarray,
    tau: float = 0.35,
    steps: int = 8,
) -> np.ndarray:
    """Approximate continuous-time random-walk diffusion.

    The stable explicit update is
        psi <- (1 - step_tau) psi + step_tau P^T psi
    repeated `steps` times. This approximates exp[-tau(I - P^T)] psi
    without constructing a dense matrix exponential.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    p = as_2d_float_array(transition, "transition")
    v = normalize_l1(np.asarray(psi, dtype=np.float64))
    if p.shape[0] != p.shape[1] or p.shape[0] != v.shape[0]:
        raise ValueError("transition and probability vector dimensions do not match.")

    step_tau = min(float(tau) / float(steps), 1.0)
    for _ in range(steps):
        v = (1.0 - step_tau) * v + step_tau * (p.T @ v)
        v = normalize_l1(v)
    return v
