# Paper claim boundaries

## Defensible claim

Singularity Search proposes Origin-Coupled Spectral Diffusion Search, a classical retrieval re-ranking operator combining local graph diffusion, latent-origin coupling, and stable multiplicative sharpening.

## Evaluation claim

The repository evaluates OCSD against BM25, cosine retrieval, diffusion ablations, origin-coupling ablations, and a no-sharpening OCSD variant using BEIR-format qrels.

## Non-claims

The repository does not claim:

- physical cosmological entanglement computation;
- faster-than-light information transfer;
- Grover-like speedup for the classical algorithm;
- universal superiority over all retrieval systems;
- full BEIR performance when `--max-docs` truncation is used.

## Publication status

The repository is now suitable for generating the evidence needed for a workshop, demo, or software artifact submission. A full research paper still requires running the experiments on full datasets, reporting the produced numbers, and discussing failures as well as improvements.
