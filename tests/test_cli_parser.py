from __future__ import annotations

from singularity_search.cli import build_parser


def test_cli_parser_has_classical_and_grover() -> None:
    parser = build_parser()
    args = parser.parse_args(["classical", "--embeddings", "x.json", "--query", "q.json"])
    assert args.command == "classical"
    args = parser.parse_args(["grover", "--marked", "101"])
    assert args.command == "grover"
