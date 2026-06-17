"""Evaluate Singularity Search on BEIR-format retrieval benchmarks.

Example:
    python experiments/evaluate_beir.py \
        --dataset scifact \
        --max-docs 2000 \
        --max-queries 300 \
        --encoder tfidf_lite \
        --output-dir reports/beir_scifact

For a full-corpus result, omit --max-docs and --max-queries if memory permits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from singularity_search.evaluation.beir_io import (
    ensure_beir_dataset,
    filter_to_loaded_corpus,
    load_corpus,
    load_qrels,
    load_queries,
)
from singularity_search.evaluation.encoders import build_encoder
from singularity_search.evaluation.runner import (
    paired_bootstrap_delta,
    run_benchmark_suite,
    write_markdown_report,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qrels-based BEIR evaluation for Singularity Search.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name to download/use.")
    parser.add_argument("--dataset-dir", type=Path, help="Existing BEIR dataset directory.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/beir"))
    parser.add_argument("--split", default="test", help="qrels split, usually test or dev.")
    parser.add_argument("--max-docs", type=int, help="Optional diagnostic corpus truncation.")
    parser.add_argument("--max-queries", type=int, help="Optional query truncation after qrels filtering.")
    parser.add_argument("--encoder", default="tfidf_lite", choices=["hash", "tfidf_lite", "bge", "sentence-transformer"])
    parser.add_argument("--model-name", help="Sentence-transformer model name for --encoder bge/sentence-transformer.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--graph-k", type=int, default=14)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/beir_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or ensure_beir_dataset(args.dataset, args.cache_dir)

    docs = load_corpus(dataset_dir, max_docs=args.max_docs)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, split=args.split)

    loaded_doc_ids = {doc.doc_id for doc in docs}
    queries, qrels = filter_to_loaded_corpus(queries, qrels, loaded_doc_ids)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
        kept = {query.query_id for query in queries}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in kept}
    if not queries:
        raise RuntimeError("No evaluable queries remain after qrels/corpus filtering.")

    encoder = build_encoder(args.encoder, model_name=args.model_name)
    results, payload = run_benchmark_suite(
        docs=docs,
        queries=queries,
        qrels=qrels,
        encoder=encoder,
        top_k=args.top_k,
        graph_k=args.graph_k,
    )

    runs = payload["runs"]
    full_key = next(key for key in runs if key.endswith(":OCSD-full"))
    bootstrap = None
    if "BM25" in runs and full_key in runs:
        bootstrap = paired_bootstrap_delta(
            runs[full_key],
            runs["BM25"],
            qrels,
            metric="nDCG@10",
            samples=args.bootstrap_samples,
        )

    write_results(args.output_dir, results, payload)
    write_markdown_report(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        encoder_name=encoder.name,
        results=results,
        bootstrap=bootstrap,
        truncated=args.max_docs is not None,
    )
    print(f"Wrote BEIR evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
