"""Launch the qrels-aware OCSD map on a BEIR-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from singularity_search.evaluation.beir_io import ensure_beir_dataset, filter_to_loaded_corpus, load_corpus, load_qrels, load_queries
from singularity_search.evaluation.encoders import build_encoder
from singularity_search.maps.viewer import MapConfig, OCSDMapViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch qrels-aware OCSD map.")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/beir"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-docs", type=int, default=800)
    parser.add_argument("--max-queries", type=int, default=80)
    parser.add_argument("--encoder", default="tfidf_lite", choices=["hash", "tfidf_lite", "bge", "sentence-transformer"])
    parser.add_argument("--model-name")
    parser.add_argument("--graph-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or ensure_beir_dataset(args.dataset, args.cache_dir)
    docs = load_corpus(dataset_dir, max_docs=args.max_docs)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, split=args.split)
    queries, qrels = filter_to_loaded_corpus(queries, qrels, {doc.doc_id for doc in docs})
    queries = queries[: args.max_queries]
    qrels = {query.query_id: qrels[query.query_id] for query in queries if query.query_id in qrels}
    if not queries:
        raise RuntimeError("No evaluable queries remain. Increase --max-docs or use the full corpus.")
    encoder = build_encoder(args.encoder, model_name=args.model_name)
    app = OCSDMapViewer(docs, queries, qrels, encoder, args.dataset, config=MapConfig(graph_k=args.graph_k))
    app.run()


if __name__ == "__main__":
    main()
