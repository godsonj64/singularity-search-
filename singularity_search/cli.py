"""Command line interface for Singularity Search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from singularity_search.classical.ocsd import OCSDConfig, OriginCoupledSpectralDiffusionSearch


def _load_json_array(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _classical_search(args: argparse.Namespace) -> int:
    embeddings = np.asarray(_load_json_array(Path(args.embeddings)), dtype=np.float64)
    query = np.asarray(_load_json_array(Path(args.query)), dtype=np.float64)

    items = None
    if args.items:
        items = _load_json_array(Path(args.items))

    config = OCSDConfig(
        graph_k=args.graph_k,
        diffusion_tau=args.diffusion_tau,
        diffusion_steps=args.diffusion_steps,
        origin_count=args.origin_count,
        candidate_count=args.candidate_count,
    )
    engine = OriginCoupledSpectralDiffusionSearch(embeddings, items=items, config=config)
    results = engine.search(query, top_k=args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def _grover(args: argparse.Namespace) -> int:
    from singularity_search.quantum.grover import build_grover_circuit, run_grover_simulation
    from singularity_search.quantum.qpy_io import save_qpy

    marked = args.marked
    circuit = build_grover_circuit(marked)
    if args.qpy:
        output = save_qpy(circuit, args.qpy)
        print(f"Saved QPY circuit: {output}")
    counts = run_grover_simulation(marked, shots=args.shots)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="singularity-search",
        description="Origin-Coupled Spectral Diffusion Search and quantum proof-of-concept.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    classical = sub.add_parser("classical", help="Run classical OCSD search over JSON embeddings.")
    classical.add_argument("--embeddings", required=True, help="JSON file containing a 2D numeric array.")
    classical.add_argument("--query", required=True, help="JSON file containing a 1D numeric query vector.")
    classical.add_argument("--items", help="Optional JSON file containing item payloads.")
    classical.add_argument("--top-k", type=int, default=5)
    classical.add_argument("--graph-k", type=int, default=12)
    classical.add_argument("--diffusion-tau", type=float, default=0.35)
    classical.add_argument("--diffusion-steps", type=int, default=8)
    classical.add_argument("--origin-count", type=int, default=4)
    classical.add_argument("--candidate-count", type=int)
    classical.set_defaults(func=_classical_search)

    grover = sub.add_parser("grover", help="Run small Qiskit Grover proof-of-concept.")
    grover.add_argument("--marked", nargs="+", required=True, help="Marked bitstrings, e.g. 101.")
    grover.add_argument("--shots", type=int, default=2048)
    grover.add_argument("--qpy", help="Optional path for saving the circuit as QPY.")
    grover.set_defaults(func=_grover)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
