# Method summary

## Origin-Coupled Spectral Diffusion Search

OCSD is implemented as a classical retrieval re-ranking method.

Given document embeddings, a query embedding, and a candidate set, the method applies:

1. direct similarity initialization;
2. probability-preserving random-walk diffusion;
3. query-conditioned latent-origin coupling;
4. multiplicative relevance sharpening;
5. top-k ranking.

## Evaluation design

The evaluation pipeline uses external qrels rather than internal probability concentration.

The default experiment script runs:

- BM25;
- encoder cosine retrieval;
- encoder cosine plus graph diffusion;
- encoder cosine plus latent-origin coupling;
- OCSD without sharpening;
- full OCSD.

Metrics are nDCG@10, nDCG@100, Recall@10, Recall@100, MRR@10, MAP@100, mean latency, and p95 latency.

## Statistical comparison

The experiment report includes a paired bootstrap comparison of OCSD-full against BM25 on nDCG@10.
