"""Palette helpers for OCSD maps."""

from __future__ import annotations


def mix_hex(left: str, right: str, amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    lr, lg, lb = int(left[1:3], 16), int(left[3:5], 16), int(left[5:7], 16)
    rr, rg, rb = int(right[1:3], 16), int(right[3:5], 16), int(right[5:7], 16)
    r = int(lr + (rr - lr) * amount)
    g = int(lg + (rg - lg) * amount)
    b = int(lb + (rb - lb) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def judgment_base(relevance: float | None) -> str:
    if relevance is None:
        return "#4d7cc9"
    if relevance > 0:
        return "#2fbf71"
    return "#d14b5a"


def node_fill(base: str, probability: float, peak: float) -> str:
    rel = 0.0 if peak <= 0 else max(0.0, min(1.0, probability / peak))
    return mix_hex("#17233d", base, 0.25 + 0.75 * rel)
