from __future__ import annotations

from singularity_search.evaluation.beir_io import Query, filter_to_loaded_corpus


def test_filter_to_loaded_corpus_keeps_only_evaluable_queries() -> None:
    queries = [Query("q1", "alpha"), Query("q2", "beta"), Query("q3", "gamma")]
    qrels = {
        "q1": {"d1": 1.0, "missing": 1.0},
        "q2": {"missing": 1.0},
        "q3": {"d3": 0.0},
    }
    filtered_queries, filtered_qrels = filter_to_loaded_corpus(queries, qrels, {"d1"})
    assert [q.query_id for q in filtered_queries] == ["q1"]
    assert filtered_qrels == {"q1": {"d1": 1.0}}
