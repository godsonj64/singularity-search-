"""Layout utilities for OCSD map displays."""

from __future__ import annotations

import numpy as np

from singularity_search.utils.math_ops import as_2d_float_array, l2_normalize


def pca2(vectors: np.ndarray) -> np.ndarray:
    """Project vectors to two dimensions using deterministic PCA."""
    x = as_2d_float_array(vectors, "vectors")
    centered = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        raise ValueError("at least two vectors are required for PCA layout.")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = min(2, vh.shape[0])
    proj = centered @ vh[:components].T
    if components == 1:
        proj = np.column_stack([proj[:, 0], np.zeros(x.shape[0])])
    scale = np.std(proj, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return proj / scale


def knn_edges(vectors: np.ndarray, k: int = 6, max_edges: int | None = None) -> list[tuple[int, int, float]]:
    """Return undirected weighted kNN edges from cosine similarity."""
    x = l2_normalize(as_2d_float_array(vectors, "vectors"), axis=1)
    n = x.shape[0]
    if n < 2:
        return []
    k_eff = min(max(1, k), n - 1)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        nearest = np.argpartition(-sim[i], kth=k_eff - 1)[:k_eff]
        for j in nearest:
            a, b = sorted((int(i), int(j)))
            edges[(a, b)] = max(edges.get((a, b), -1.0), float(sim[i, j]))
    out = [(a, b, w) for (a, b), w in edges.items()]
    out.sort(key=lambda item: item[2], reverse=True)
    if max_edges is not None:
        out = out[:max_edges]
    return out


def scale_to_canvas(coords: np.ndarray, width: int, height: int, padding: int = 44) -> np.ndarray:
    """Scale 2D coordinates into canvas coordinates."""
    xy = as_2d_float_array(coords, "coords")
    if xy.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2).")
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    norm = (xy - lo[None, :]) / span[None, :]
    w = max(width - 2 * padding, 1)
    h = max(height - 2 * padding, 1)
    out = np.empty_like(norm)
    out[:, 0] = padding + norm[:, 0] * w
    out[:, 1] = padding + norm[:, 1] * h
    return out
