from __future__ import annotations

import numpy as np

from singularity_search.evaluation.encoders import HashingTextEncoder, TfidfLiteEncoder


def test_hash_encoder_shapes() -> None:
    encoder = HashingTextEncoder(n_dims=32)
    docs = ["alpha beta", "gamma delta"]
    encoder.fit(docs)
    x = encoder.encode_documents(docs)
    q = encoder.encode_queries(["alpha"])
    assert x.shape == (2, 32)
    assert q.shape == (1, 32)
    assert np.isfinite(x).all()


def test_tfidf_encoder_shapes() -> None:
    encoder = TfidfLiteEncoder(max_features=8)
    docs = ["alpha beta", "gamma delta"]
    encoder.fit(docs)
    x = encoder.encode_documents(docs)
    q = encoder.encode_queries(["alpha"])
    assert x.shape[0] == 2
    assert q.shape[0] == 1
    assert x.shape[1] == q.shape[1]
