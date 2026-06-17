"""Run a small classical Origin-Coupled Spectral Diffusion Search demo."""

from __future__ import annotations

import numpy as np

from singularity_search import OCSDConfig, OriginCoupledSpectralDiffusionSearch


ITEMS = [
    "graph heat kernel ranking",
    "quantum amplitude amplification",
    "hotel banquet menu planning",
    "spectral graph theory and diffusion maps",
    "semantic vector database retrieval",
    "origin-coupled latent field search",
    "restaurant point-of-sale software",
    "Grover search oracle construction",
]


def toy_embed(text: str) -> np.ndarray:
    """Deterministic toy embedding. Replace with a real embedding model in production."""
    terms = [
        "graph",
        "heat",
        "quantum",
        "amplitude",
        "hotel",
        "spectral",
        "semantic",
        "origin",
        "search",
        "oracle",
    ]
    text_lower = text.lower()
    vec = np.array([text_lower.count(term) for term in terms], dtype=np.float64)
    vec += (sum(ord(ch) for ch in text_lower) % 17) * 1e-3
    return vec


def main() -> None:
    embeddings = np.vstack([toy_embed(item) for item in ITEMS])
    query = toy_embed("origin quantum graph search")

    engine = OriginCoupledSpectralDiffusionSearch(
        embeddings,
        items=ITEMS,
        config=OCSDConfig(graph_k=3, origin_count=3, candidate_count=None),
    )
    for result in engine.search(query, top_k=5):
        print(result)


if __name__ == "__main__":
    main()
