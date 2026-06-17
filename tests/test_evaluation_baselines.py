from __future__ import annotations

from singularity_search.evaluation.beir_io import Document, Query
from singularity_search.evaluation.encoders import HashingTextEncoder
from singularity_search.evaluation.runner import run_benchmark_suite


def test_benchmark_suite_runs_on_tiny_corpus() -> None:
    docs = [
        Document("d1", "graph search", "spectral diffusion ranking"),
        Document("d2", "hotel menu", "banquet catering planning"),
        Document("d3", "quantum search", "grover oracle amplitude"),
        Document("d4", "semantic retrieval", "vector cosine benchmark"),
    ]
    queries = [Query("q1", "graph diffusion search"), Query("q2", "banquet menu")]
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    results, payload = run_benchmark_suite(docs, queries, qrels, HashingTextEncoder(n_dims=64), top_k=4, graph_k=2)
    assert len(results) >= 5
    assert "runs" in payload
    assert any(result.method.endswith("OCSD-full") for result in results)
