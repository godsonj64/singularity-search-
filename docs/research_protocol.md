# Research protocol for a publishable OCSD evaluation

This protocol keeps the paper claim narrow and testable.

## Claim

Origin-Coupled Spectral Diffusion Search is evaluated as a classical second-stage retrieval and re-ranking operator. The valid claim is not quantum speedup and not cosmological entanglement computation.

## Required baselines

The evaluation runner includes:

1. BM25.
2. Encoder cosine retrieval.
3. Encoder cosine plus graph diffusion.
4. Encoder cosine plus latent-origin coupling.
5. OCSD without sharpening.
6. Full OCSD.

## Required metrics

The benchmark report computes:

- nDCG@10.
- nDCG@100.
- Recall@10.
- Recall@100.
- MRR@10.
- MAP@100.
- Mean query latency.
- P95 query latency.

## Statistical test

The report includes a paired bootstrap comparison of `OCSD-full` against `BM25` on nDCG@10.

## Corpus discipline

If `--max-docs` is used, results must be described as a truncated-corpus diagnostic evaluation. Full BEIR claims require the full corpus and full qrels split.

## Encoder discipline

The default `tfidf_lite` encoder is a reproducible control. Stronger claims require a stronger encoder such as BGE, E5, or another sentence-transformer model.

## Recommended paper wording

> We introduce Origin-Coupled Spectral Diffusion Search, a query-conditioned graph re-ranking operator that combines local manifold diffusion with latent-origin coupling. We evaluate it as a classical second-stage retrieval method on standard information-retrieval benchmarks using qrels-based metrics and ablations.

## Wording to avoid

- Do not claim a universal super-fast search algorithm.
- Do not claim physical cosmological entanglement is used for computation.
- Do not claim Grover-like speedup for the classical operator.
