from __future__ import annotations

import numpy as np

from singularity_search import OCSDConfig, OriginCoupledSpectralDiffusionSearch
from singularity_search.classical.graph import build_knn_weight_matrix, diffuse_probability, random_walk_transition


def test_probability_diffusion_preserves_normalization() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    weights = build_knn_weight_matrix(x, k=2)
    transition = random_walk_transition(weights)
    psi = np.array([0.7, 0.2, 0.05, 0.05])
    out = diffuse_probability(psi, transition, tau=0.5, steps=4)
    assert np.all(out >= 0)
    assert np.isclose(out.sum(), 1.0)


def test_ocsd_search_returns_relevant_result() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    items = ["alpha", "alpha-near", "beta", "gamma"]
    engine = OriginCoupledSpectralDiffusionSearch(
        embeddings,
        items=items,
        config=OCSDConfig(graph_k=2, origin_count=2, sharpening_steps=1),
    )
    results = engine.search([1.0, 0.0, 0.0], top_k=2)
    assert results[0]["item"] in {"alpha", "alpha-near"}
    assert len(results) == 2
    assert results[0]["probability"] >= results[1]["probability"]
