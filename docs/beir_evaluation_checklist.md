# BEIR evaluation checklist

Use this checklist before presenting results in a paper.

- Use the full corpus unless the report clearly says diagnostic subset.
- Report nDCG@10, Recall@100, MRR@10, MAP@100, mean latency, and p95 latency.
- Include BM25 and cosine retrieval baselines.
- Include diffusion, origin-coupling, and no-sharpening ablations.
- Include paired bootstrap confidence intervals.
- State the encoder used and whether it is hash, TF-IDF-lite, or a dense encoder.
- Keep quantum claims separate from the classical retrieval evaluation.
