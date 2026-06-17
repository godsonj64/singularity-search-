# Singularity Search

**Singularity Search** is a research-grade Python repository for **Origin-Coupled Spectral Diffusion Search**.

The name comes from a physics analogy: distant objects may preserve structure from a shared origin. In this repository, that idea is converted into a defensible computational operator. The shared origin is not claimed to be physical cosmological entanglement. It is a learned latent prior over a database.

The repository contains two parts:

1. A practical **classical search engine** based on graph diffusion, latent-origin coupling, and multiplicative relevance sharpening.
2. A small **Qiskit quantum proof-of-concept** showing a correct Grover-style amplitude amplification circuit over indexed states.

The production algorithm is classical. The quantum module is for small demonstrations, circuit export, and theoretical experiments.

---

## Mathematical core

Given candidate embeddings `z_i`, query embedding `z_q`, and a random-walk transition matrix `P`, the corrected classical operator is

```text
psi_T = M_q^T D_Omega(q) exp[-tau(I - P^T)] psi_0.
```

The initial distribution is

```text
psi_0(i) = exp(beta cos(z_i, z_q)) / sum_j exp(beta cos(z_j, z_q)).
```

The multi-origin coupling score is

```text
s(i, q) = alpha cos(z_i, z_q)
        + (1 - alpha) sum_m a_m(q) cos(z_i, Omega_m) cos(z_q, Omega_m).
```

The sharpening step is a stable multiplicative-weights update:

```text
M_q[psi](i) = psi(i) exp(xi s(i, q)) / sum_j psi(j) exp(xi s(j, q)).
```

This avoids the mathematical error of treating a classical probability reflection as Grover amplitude amplification.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the Qiskit quantum demonstration:

```bash
pip install -e '.[quantum]'
```

Qiskit is the SDK used for building quantum circuits, and Qiskit Aer provides a high-performance simulator. IBM's QPY format is used only for saving and loading `QuantumCircuit` objects, not for execution.

---

## Run the classical demo

```bash
python examples/classical_demo.py
```

Expected output is a ranked list of semantically related toy items with diagnostic scores.

---

## Run the quantum Grover demo

```bash
pip install -e '.[quantum]'
python examples/quantum_grover_demo.py
```

This builds a small Grover circuit for the marked state `101`, runs it in Qiskit Aer, and saves the circuit as:

```text
grover_101.qpy
```

---

## CLI usage

### Classical search

Create `embeddings.json`:

```json
[[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]]
```

Create `query.json`:

```json
[1, 0, 0]
```

Run:

```bash
singularity-search classical --embeddings embeddings.json --query query.json --top-k 2
```

### Quantum proof-of-concept

```bash
singularity-search grover --marked 101 --shots 2048 --qpy grover_101.qpy
```

---

## Repository structure

```text
singularity-search/
  singularity_search/
    classical/
      graph.py
      ocsd.py
    quantum/
      grover.py
      oracle_from_scores.py
      qpy_io.py
    utils/
      math_ops.py
    cli.py
  examples/
    classical_demo.py
    quantum_grover_demo.py
  tests/
    test_classical.py
    test_quantum_optional.py
  pyproject.toml
  README.md
  LICENSE
```

---

## Scientific limits

This project does not claim that cosmological entanglement can be used for faster-than-light search. It also does not claim that the classical OCSD operator has Grover's quadratic quantum speedup.

The defensible claim is narrower and stronger:

> Singularity Search implements a classical semantic re-ranking operator that combines approximate candidate retrieval, graph diffusion, query-conditioned latent-origin coupling, and stable contrastive sharpening. A separate Qiskit module demonstrates the correct quantum circuit structure for small marked-set search.

---

## Development

```bash
pip install -e '.[dev]'
pytest
```

Optional quantum tests are skipped unless Qiskit is installed.

---

## Next research directions

1. Replace the toy embedding function with sentence-transformer, OpenAI, or local model embeddings.
2. Add HNSW candidate retrieval for million-scale corpora.
3. Evaluate against cosine search, BM25, hybrid search, diffusion re-ranking, and graph PageRank baselines.
4. Add ablations for graph diffusion, origin coupling, and multiplicative sharpening.
5. Explore block-encoding or QRAM-assumption versions only as theoretical quantum research, not as near-term product claims.
