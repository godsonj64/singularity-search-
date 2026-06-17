"""Benchmark runner for OCSD and retrieval baselines."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from singularity_search import OCSDConfig, OriginCoupledSpectralDiffusionSearch
from singularity_search.classical.graph import build_knn_weight_matrix, diffuse_probability, random_walk_transition
from singularity_search.evaluation.beir_io import Document, Query
from singularity_search.evaluation.bm25 import BM25Index
from singularity_search.evaluation.encoders import TextEncoder
from singularity_search.evaluation.metrics import RetrievalMetrics, evaluate_run
from singularity_search.utils.math_ops import cosine_matrix, normalize_l1, softmax


@dataclass(frozen=True)
class MethodResult:
    method: str
    metrics: RetrievalMetrics
    mean_latency_ms: float
    p95_latency_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            **self.metrics.as_dict(),
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


def _rank_by_scores(doc_ids: list[str], scores: np.ndarray, top_k: int) -> list[str]:
    order = np.argsort(-scores)[: min(top_k, len(doc_ids))]
    return [doc_ids[int(i)] for i in order]


def _origin_scores(query: np.ndarray, doc_embeddings: np.ndarray, origin_count: int = 5) -> np.ndarray:
    # Deterministic light version of multi-origin scoring for ablation evaluation.
    n = doc_embeddings.shape[0]
    k = min(origin_count, n)
    mean_vec = doc_embeddings.mean(axis=0)
    distances = np.sum((doc_embeddings - mean_vec[None, :]) ** 2, axis=1)
    first = int(np.argmin(distances))
    selected = [first]
    while len(selected) < k:
        centers = doc_embeddings[selected]
        dist_to_centers = np.min(
            np.sum((doc_embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
        )
        dist_to_centers[selected] = -np.inf
        selected.append(int(np.argmax(dist_to_centers)))
    origins = doc_embeddings[selected]
    query_origin = origins @ query
    attn = softmax(query_origin, temperature=0.35)
    return np.clip((doc_embeddings @ origins.T) @ (attn * query_origin), -1.0, 1.0)


def run_cosine(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    queries: list[Query],
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[dict[str, list[str]], list[float]]:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query, qvec in zip(queries, query_embeddings):
        t0 = time.perf_counter()
        scores = cosine_matrix(qvec, doc_embeddings)
        run[query.query_id] = _rank_by_scores(doc_ids, scores, top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return run, latencies


def run_diffusion_ablation(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    queries: list[Query],
    query_embeddings: np.ndarray,
    top_k: int,
    graph_k: int,
) -> tuple[dict[str, list[str]], list[float]]:
    weights = build_knn_weight_matrix(doc_embeddings, k=min(graph_k, len(doc_ids) - 1))
    transition = random_walk_transition(weights)
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query, qvec in zip(queries, query_embeddings):
        t0 = time.perf_counter()
        direct = cosine_matrix(qvec, doc_embeddings)
        psi = softmax(12.0 * direct)
        psi = diffuse_probability(psi, transition, tau=0.35, steps=8)
        run[query.query_id] = _rank_by_scores(doc_ids, psi, top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return run, latencies


def run_origin_ablation(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    queries: list[Query],
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[dict[str, list[str]], list[float]]:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query, qvec in zip(queries, query_embeddings):
        t0 = time.perf_counter()
        direct = cosine_matrix(qvec, doc_embeddings)
        origin = _origin_scores(qvec, doc_embeddings)
        scores = 0.65 * direct + 0.35 * origin
        run[query.query_id] = _rank_by_scores(doc_ids, scores, top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return run, latencies


def run_ocsd(
    docs: list[Document],
    doc_embeddings: np.ndarray,
    queries: list[Query],
    query_embeddings: np.ndarray,
    top_k: int,
    graph_k: int,
    enable_diffusion: bool = True,
    enable_origin: bool = True,
    enable_sharpening: bool = True,
) -> tuple[dict[str, list[str]], list[float]]:
    items = docs
    config = OCSDConfig(
        graph_k=min(graph_k, len(docs) - 1),
        diffusion_tau=0.35 if enable_diffusion else 0.0,
        diffusion_steps=8,
        origin_count=5,
        origin_strength=0.75 if enable_origin else 0.0,
        direct_weight=0.62 if enable_origin else 1.0,
        sharpening_strength=1.35 if enable_sharpening else 0.0,
        sharpening_steps=2 if enable_sharpening else 0,
        candidate_count=None,
    )
    engine = OriginCoupledSpectralDiffusionSearch(doc_embeddings, items=items, config=config)
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query, qvec in zip(queries, query_embeddings):
        t0 = time.perf_counter()
        rows = engine.search(qvec, top_k=top_k)
        run[query.query_id] = [row["item"].doc_id for row in rows]
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return run, latencies


def run_bm25(
    docs: list[Document],
    queries: list[Query],
    top_k: int,
) -> tuple[dict[str, list[str]], list[float]]:
    index = BM25Index([doc.combined_text for doc in docs], [doc.doc_id for doc in docs])
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query in queries:
        t0 = time.perf_counter()
        run[query.query_id] = index.search(query.text, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return run, latencies


def summarize_method(
    method: str,
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    latencies_ms: list[float],
) -> MethodResult:
    metrics = evaluate_run(run, qrels)
    lat = np.asarray(latencies_ms, dtype=np.float64)
    return MethodResult(
        method=method,
        metrics=metrics,
        mean_latency_ms=float(np.mean(lat)),
        p95_latency_ms=float(np.percentile(lat, 95)),
    )


def paired_bootstrap_delta(
    run_a: dict[str, list[str]],
    run_b: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    metric: str = "nDCG@10",
    samples: int = 1000,
    seed: int = 13,
) -> dict[str, float]:
    from singularity_search.evaluation.metrics import evaluate_query

    qids = [qid for qid in qrels if qid in run_a and qid in run_b]
    if not qids:
        raise ValueError("No overlapping queries for bootstrap.")
    diffs = []
    per_query = []
    for qid in qids:
        a = evaluate_query(run_a[qid], qrels[qid])[metric]
        b = evaluate_query(run_b[qid], qrels[qid])[metric]
        per_query.append(a - b)
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_query, dtype=np.float64)
    for _ in range(samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        diffs.append(float(np.mean(arr[idx])))
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "delta_mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(diffs_arr, 2.5)),
        "ci95_high": float(np.percentile(diffs_arr, 97.5)),
        "p_delta_le_0": float(np.mean(diffs_arr <= 0.0)),
    }


def run_benchmark_suite(
    docs: list[Document],
    queries: list[Query],
    qrels: dict[str, dict[str, float]],
    encoder: TextEncoder,
    top_k: int = 100,
    graph_k: int = 14,
) -> tuple[list[MethodResult], dict[str, dict[str, list[str]]]]:
    texts = [doc.combined_text for doc in docs]
    query_texts = [q.text for q in queries]
    encoder.fit(texts)
    doc_embeddings = encoder.encode_documents(texts)
    query_embeddings = encoder.encode_queries(query_texts)
    doc_ids = [doc.doc_id for doc in docs]

    runs: dict[str, dict[str, list[str]]] = {}
    results: list[MethodResult] = []

    bm25_run, bm25_lat = run_bm25(docs, queries, top_k)
    runs["BM25"] = bm25_run
    results.append(summarize_method("BM25", bm25_run, qrels, bm25_lat))

    cosine_run, cosine_lat = run_cosine(doc_ids, doc_embeddings, queries, query_embeddings, top_k)
    runs[f"{encoder.name}:cosine"] = cosine_run
    results.append(summarize_method(f"{encoder.name}:cosine", cosine_run, qrels, cosine_lat))

    diffusion_run, diffusion_lat = run_diffusion_ablation(doc_ids, doc_embeddings, queries, query_embeddings, top_k, graph_k)
    runs[f"{encoder.name}:cosine+diffusion"] = diffusion_run
    results.append(summarize_method(f"{encoder.name}:cosine+diffusion", diffusion_run, qrels, diffusion_lat))

    origin_run, origin_lat = run_origin_ablation(doc_ids, doc_embeddings, queries, query_embeddings, top_k)
    runs[f"{encoder.name}:cosine+origin"] = origin_run
    results.append(summarize_method(f"{encoder.name}:cosine+origin", origin_run, qrels, origin_lat))

    ocsd_no_sharp_run, ocsd_no_sharp_lat = run_ocsd(
        docs, doc_embeddings, queries, query_embeddings, top_k, graph_k, True, True, False
    )
    runs[f"{encoder.name}:OCSD-no-sharpen"] = ocsd_no_sharp_run
    results.append(summarize_method(f"{encoder.name}:OCSD-no-sharpen", ocsd_no_sharp_run, qrels, ocsd_no_sharp_lat))

    full_run, full_lat = run_ocsd(docs, doc_embeddings, queries, query_embeddings, top_k, graph_k, True, True, True)
    runs[f"{encoder.name}:OCSD-full"] = full_run
    results.append(summarize_method(f"{encoder.name}:OCSD-full", full_run, qrels, full_lat))

    return results, {"runs": runs}


def write_results(output_dir: Path, results: list[MethodResult], runs_payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    rows = [result.as_dict() for result in results]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "runs.json").write_text(json.dumps(runs_payload, indent=2), encoding="utf-8")


def write_markdown_report(
    output_dir: Path,
    dataset_name: str,
    encoder_name: str,
    results: list[MethodResult],
    bootstrap: dict[str, float] | None,
    truncated: bool,
) -> None:
    lines = [
        "# Singularity Search BEIR Evaluation Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Encoder: `{encoder_name}`",
        f"- Corpus mode: `{'truncated diagnostic subset' if truncated else 'full or user-provided corpus'}`",
        "",
        "## Metrics",
        "",
        "| Method | nDCG@10 | Recall@100 | MRR@10 | MAP@100 | Mean latency ms | P95 latency ms | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        data = result.as_dict()
        lines.append(
            f"| {data['method']} | {data['nDCG@10']:.4f} | {data['Recall@100']:.4f} | "
            f"{data['MRR@10']:.4f} | {data['MAP@100']:.4f} | {data['mean_latency_ms']:.2f} | "
            f"{data['p95_latency_ms']:.2f} | {data['queries_evaluated']} |"
        )
    if bootstrap:
        lines.extend(
            [
                "",
                "## Paired bootstrap comparison",
                "",
                "The reported delta is `OCSD-full minus BM25` on nDCG@10.",
                "",
                f"- Mean delta: `{bootstrap['delta_mean']:.6f}`",
                f"- 95% CI: `[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]`",
                f"- Bootstrap probability delta <= 0: `{bootstrap['p_delta_le_0']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim discipline",
            "",
            "These results evaluate OCSD as a classical second-stage retrieval/re-ranking operator. They do not establish quantum speedup or cosmological-entanglement computation.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
