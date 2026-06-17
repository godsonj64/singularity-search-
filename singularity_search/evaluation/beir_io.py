"""BEIR-format input/output helpers.

Expected layout:

    dataset/
      corpus.jsonl
      queries.jsonl
      qrels/test.tsv

This module deliberately keeps data loading simple and explicit so benchmark
subsetting decisions are visible in the generated reports.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def combined_text(self) -> str:
        return f"{self.title} {self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def ensure_beir_dataset(dataset_name: str, cache_dir: Path) -> Path:
    dataset_name = dataset_name.strip().lower()
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty.")
    dataset_dir = cache_dir / dataset_name
    if (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "queries.jsonl").exists():
        return dataset_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{dataset_name}.zip"
    url = f"{BEIR_BASE_URL}/{dataset_name}.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(cache_dir)
    if not (dataset_dir / "corpus.jsonl").exists():
        raise RuntimeError(f"Downloaded {dataset_name}, but corpus.jsonl was not found.")
    return dataset_dir


def load_corpus(dataset_dir: Path, max_docs: int | None = None) -> list[Document]:
    path = dataset_dir / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            title = str(row.get("title", "") or "")
            text = str(row.get("text", "") or "")
            doc_id = str(row.get("_id", len(docs)))
            if not (title.strip() or text.strip()):
                continue
            docs.append(Document(doc_id=doc_id, title=title.strip(), text=text.strip()))
            if max_docs is not None and len(docs) >= max_docs:
                break
    if not docs:
        raise ValueError(f"No documents loaded from {path}.")
    return docs


def load_queries(dataset_dir: Path, max_queries: int | None = None) -> list[Query]:
    path = dataset_dir / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    queries: list[Query] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row.get("_id", ""))
            text = str(row.get("text", "") or "").strip()
            if qid and text:
                queries.append(Query(query_id=qid, text=text))
            if max_queries is not None and len(queries) >= max_queries:
                break
    if not queries:
        raise ValueError(f"No queries loaded from {path}.")
    return queries


def load_qrels(dataset_dir: Path, split: str = "test") -> dict[str, dict[str, float]]:
    path = dataset_dir / "qrels" / f"{split}.tsv"
    if not path.exists():
        alt = dataset_dir / "qrels" / f"{split}.txt"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(path)

    qrels: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.lower().startswith("query-id") or line.lower().startswith("qid"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            try:
                rel = float(score)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = rel
    if not qrels:
        raise ValueError(f"No qrels loaded from {path}.")
    return qrels


def filter_to_loaded_corpus(
    queries: list[Query],
    qrels: dict[str, dict[str, float]],
    loaded_doc_ids: set[str],
) -> tuple[list[Query], dict[str, dict[str, float]]]:
    filtered_qrels: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        kept = {doc_id: rel for doc_id, rel in rels.items() if doc_id in loaded_doc_ids}
        if any(rel > 0 for rel in kept.values()):
            filtered_qrels[qid] = kept
    filtered_queries = [q for q in queries if q.query_id in filtered_qrels]
    return filtered_queries, filtered_qrels
