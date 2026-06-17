"""Standard information-retrieval metrics.

The functions in this module evaluate a ranked run against external qrels.
Internal confidence values such as top probability are intentionally not used
as quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from statistics import mean


Qrels = dict[str, dict[str, float]]
Run = dict[str, list[str]]


@dataclass(frozen=True)
class RetrievalMetrics:
    ndcg_at_10: float
    ndcg_at_100: float
    recall_at_10: float
    recall_at_100: float
    mrr_at_10: float
    map_at_100: float
    queries_evaluated: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nDCG@10": self.ndcg_at_10,
            "nDCG@100": self.ndcg_at_100,
            "Recall@10": self.recall_at_10,
            "Recall@100": self.recall_at_100,
            "MRR@10": self.mrr_at_10,
            "MAP@100": self.map_at_100,
            "queries_evaluated": self.queries_evaluated,
        }


def _relevant_docs(qrel: dict[str, float]) -> set[str]:
    return {doc_id for doc_id, rel in qrel.items() if rel > 0}


def dcg(relevances: list[float]) -> float:
    return sum(((2.0**rel) - 1.0) / log2(rank + 2.0) for rank, rel in enumerate(relevances))


def ndcg_at(ranking: list[str], qrel: dict[str, float], k: int) -> float:
    if not qrel:
        return 0.0
    observed = [float(qrel.get(doc_id, 0.0)) for doc_id in ranking[:k]]
    ideal = sorted((float(rel) for rel in qrel.values()), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg(observed) / ideal_dcg


def recall_at(ranking: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = _relevant_docs(qrel)
    if not relevant:
        return 0.0
    retrieved = set(ranking[:k])
    return len(relevant & retrieved) / len(relevant)


def mrr_at(ranking: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = _relevant_docs(qrel)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at(ranking: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = _relevant_docs(qrel)
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def evaluate_query(ranking: list[str], qrel: dict[str, float]) -> dict[str, float]:
    return {
        "nDCG@10": ndcg_at(ranking, qrel, 10),
        "nDCG@100": ndcg_at(ranking, qrel, 100),
        "Recall@10": recall_at(ranking, qrel, 10),
        "Recall@100": recall_at(ranking, qrel, 100),
        "MRR@10": mrr_at(ranking, qrel, 10),
        "MAP@100": average_precision_at(ranking, qrel, 100),
    }


def evaluate_run(run: Run, qrels: Qrels) -> RetrievalMetrics:
    rows: list[dict[str, float]] = []
    for qid, qrel in qrels.items():
        if qid not in run:
            continue
        if not _relevant_docs(qrel):
            continue
        rows.append(evaluate_query(run[qid], qrel))

    if not rows:
        raise ValueError("No overlapping queries with positive qrels were evaluated.")

    return RetrievalMetrics(
        ndcg_at_10=mean(row["nDCG@10"] for row in rows),
        ndcg_at_100=mean(row["nDCG@100"] for row in rows),
        recall_at_10=mean(row["Recall@10"] for row in rows),
        recall_at_100=mean(row["Recall@100"] for row in rows),
        mrr_at_10=mean(row["MRR@10"] for row in rows),
        map_at_100=mean(row["MAP@100"] for row in rows),
        queries_evaluated=len(rows),
    )
