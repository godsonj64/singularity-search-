"""Run the Qiskit Grover proof-of-concept and save a QPY circuit."""

from __future__ import annotations

from pathlib import Path

from singularity_search.quantum.grover import build_grover_circuit, run_grover_simulation
from singularity_search.quantum.qpy_io import save_qpy


def main() -> None:
    marked = ["101"]
    circuit = build_grover_circuit(marked)
    output = save_qpy(circuit, Path("grover_101.qpy"))
    counts = run_grover_simulation(marked, shots=2048)
    print(f"Saved: {output}")
    print(counts)
    print("Most likely:", max(counts, key=counts.get))


if __name__ == "__main__":
    main()
