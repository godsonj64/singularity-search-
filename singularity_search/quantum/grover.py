"""Small, correct Grover-search proof-of-concept using Qiskit.

This module is intentionally finite and explicit. It does not load a large
classical embedding database into a quantum computer. Instead, it demonstrates
how a Boolean marked-set oracle can be encoded as a phase oracle and amplified.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _import_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover - only triggered without optional deps.
        raise ImportError(
            "Quantum features require optional dependencies. Install with: "
            "pip install 'singularity-search[quantum]'"
        ) from exc
    return QuantumCircuit, transpile, AerSimulator


def _validate_bitstring(bitstring: str) -> None:
    if not bitstring:
        raise ValueError("bitstring must be non-empty.")
    if any(bit not in "01" for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'.")


def phase_oracle_for_marked_states(n_qubits: int, marked_states: Iterable[str]):
    """Create a phase oracle that flips each marked computational basis state."""
    QuantumCircuit, _, _ = _import_qiskit()
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")

    marked = list(marked_states)
    if not marked:
        raise ValueError("marked_states must not be empty.")

    qc = QuantumCircuit(n_qubits, name="Oq")
    for state in marked:
        _validate_bitstring(state)
        if len(state) != n_qubits:
            raise ValueError("all marked states must match n_qubits.")

        # Qiskit uses little-endian qubit indexing for basis strings. Reversing
        # maps the leftmost printed bit to the highest-index qubit.
        for qubit, bit in enumerate(reversed(state)):
            if bit == "0":
                qc.x(qubit)

        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        for qubit, bit in enumerate(reversed(state)):
            if bit == "0":
                qc.x(qubit)
    return qc


def diffusion_operator(n_qubits: int):
    """Return Grover's reflection about the uniform superposition."""
    QuantumCircuit, _, _ = _import_qiskit()
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")

    qc = QuantumCircuit(n_qubits, name="D")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_grover_iterations(n_states: int, marked_count: int = 1) -> int:
    """Return the standard near-optimal Grover iteration count."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2.")
    if marked_count < 1 or marked_count > n_states:
        raise ValueError("marked_count must lie in [1, n_states].")
    return max(1, int(round((math.pi / 4.0) * math.sqrt(n_states / marked_count))))


def build_grover_circuit(marked_states: Iterable[str], iterations: int | None = None):
    """Build a measured Grover circuit for the supplied marked states."""
    QuantumCircuit, _, _ = _import_qiskit()
    marked = list(marked_states)
    if not marked:
        raise ValueError("marked_states must not be empty.")
    for state in marked:
        _validate_bitstring(state)
    n_qubits = len(marked[0])
    if any(len(state) != n_qubits for state in marked):
        raise ValueError("all marked states must have the same length.")

    n_states = 2**n_qubits
    if len(set(marked)) > n_states:
        raise ValueError("too many marked states.")
    if iterations is None:
        iterations = optimal_grover_iterations(n_states, marked_count=len(set(marked)))
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    qc = QuantumCircuit(n_qubits, n_qubits, name="singularity_grover_search")
    qc.h(range(n_qubits))

    oracle = phase_oracle_for_marked_states(n_qubits, sorted(set(marked)))
    diffuser = diffusion_operator(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_grover_simulation(marked_states: Iterable[str], shots: int = 2048) -> dict[str, int]:
    """Run the Grover circuit on Qiskit Aer and return measurement counts."""
    _, transpile, AerSimulator = _import_qiskit()
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    circuit = build_grover_circuit(marked_states)
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    result = simulator.run(compiled, shots=shots).result()
    return dict(result.get_counts())
