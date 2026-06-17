from __future__ import annotations

from singularity_search.evaluation.bm25 import BM25Index


def test_bm25_ranks_matching_document_first() -> None:
    docs = ["graph diffusion retrieval", "hotel banquet catering", "quantum oracle search"]
    ids = ["d1", "d2", "d3"]
    index = BM25Index(docs, ids)
    assert index.search("banquet catering", top_k=3)[0] == "d2"
