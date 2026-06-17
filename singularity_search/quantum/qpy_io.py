"""QPY save/load helpers for Qiskit circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_qpy():
    try:
        from qiskit import qpy
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "QPY support requires Qiskit. Install with: "
            "pip install 'singularity-search[quantum]'"
        ) from exc
    return qpy


def save_qpy(circuit: Any, path: str | Path) -> Path:
    """Save one QuantumCircuit to a QPY file."""
    qpy = _import_qpy()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        qpy.dump(circuit, handle)
    return output


def load_qpy(path: str | Path) -> Any:
    """Load the first QuantumCircuit from a QPY file."""
    qpy = _import_qpy()
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    with input_path.open("rb") as handle:
        circuits = qpy.load(handle)
    if not circuits:
        raise ValueError("QPY file did not contain any circuits.")
    return circuits[0]
