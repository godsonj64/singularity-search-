"""Convert classical search scores into a small Grover marked-set oracle.

This is a research demonstration. Real semantic search still runs classically
because loading large embedding databases into quantum amplitudes is the main
bottleneck on current quantum hardware.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def marked_bitstrings_from_scores(
    scores: Sequence[float],
    quantile: float = 0.75,
) -> list[str]:
    """Return bitstrings whose scores exceed a quantile threshold.

    The number of states must be a power of two because the bitstrings encode
    computational basis states on n qubits.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contain NaN or infinite values.")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1].")

    n = arr.size
    n_qubits_float = math.log2(n)
    if int(n_qubits_float) != n_qubits_float:
        raise ValueError("number of scores must be a power of two.")
    n_qubits = int(n_qubits_float)
    threshold = float(np.quantile(arr, quantile))
    marked = [format(i, f"0{n_qubits}b") for i, value in enumerate(arr) if value >= threshold]
    if not marked:
        marked = [format(int(np.argmax(arr)), f"0{n_qubits}b")]
    return marked
