from __future__ import annotations

import importlib.util

import pytest

from singularity_search.quantum.oracle_from_scores import marked_bitstrings_from_scores


def test_marked_bitstrings_from_scores() -> None:
    marked = marked_bitstrings_from_scores([0.1, 0.2, 0.9, 0.3], quantile=0.75)
    assert "10" in marked


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_grover_builds_when_qiskit_available() -> None:
    from singularity_search.quantum.grover import build_grover_circuit

    circuit = build_grover_circuit(["101"], iterations=1)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
