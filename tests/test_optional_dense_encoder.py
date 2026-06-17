from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="optional dense encoder dependency not installed")
def test_dense_encoder_constructs_when_available() -> None:
    from singularity_search.evaluation.encoders import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder("BAAI/bge-small-en-v1.5", batch_size=1)
    assert encoder.name.startswith("sentence_transformer:")
