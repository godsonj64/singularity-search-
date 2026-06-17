"""Numerically stable mathematical primitives."""

from __future__ import annotations

import numpy as np


def as_2d_float_array(values: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def as_1d_float_array(values: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def l2_normalize(matrix: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_matrix(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = as_1d_float_array(query, "query")
    x = as_2d_float_array(vectors, "vectors")
    if q.shape[0] != x.shape[1]:
        raise ValueError("query dimension must match vectors dimension.")
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    xn = l2_normalize(x, axis=1)
    return xn @ qn


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    vals = as_1d_float_array(values, "values")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scaled = vals / temperature
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom <= 0 or not np.isfinite(denom):
        raise FloatingPointError("softmax normalization failed.")
    return exp_vals / denom


def normalize_l1(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vals = as_1d_float_array(values, "values")
    vals = np.maximum(vals, 0.0)
    total = float(np.sum(vals))
    if total <= eps:
        return np.full(vals.shape[0], 1.0 / vals.shape[0], dtype=np.float64)
    return vals / total


def squared_distances_to_vector(vectors: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = as_2d_float_array(vectors, "vectors")
    v = as_1d_float_array(vector, "vector")
    if x.shape[1] != v.shape[0]:
        raise ValueError("vector dimension mismatch.")
    diff = x - v[None, :]
    return np.sum(diff * diff, axis=1)
