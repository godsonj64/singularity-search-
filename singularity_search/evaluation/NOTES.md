# Evaluation module

This module adds measurable retrieval experiments to Singularity Search.

## Components

- `beir_io.py`: loads BEIR-format corpus, queries, and qrels.
- `metrics.py`: computes nDCG, Recall, MRR, and MAP.
- `bm25.py`: dependency-light BM25 baseline.
- `encoders.py`: hash, TF-IDF-lite, and optional sentence-transformer encoders.
- `runner.py`: runs baselines, ablations, full OCSD, paired bootstrap, and reports.

## Baseline and ablation set

- BM25.
- Encoder cosine.
- Encoder cosine plus graph diffusion.
- Encoder cosine plus latent-origin coupling.
- OCSD without sharpening.
- Full OCSD.

## Scientific note

Retrieval quality is computed against external qrels, not from the model confidence values alone.
