"""Compact qrels-aware OCSD map viewer."""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np

from singularity_search import OCSDConfig, OriginCoupledSpectralDiffusionSearch
from singularity_search.evaluation.beir_io import Document, Query
from singularity_search.evaluation.encoders import TextEncoder
from singularity_search.maps.layout import knn_edges, pca2, scale_to_canvas
from singularity_search.maps.palette import judgment_base, mix_hex, node_fill
from singularity_search.maps.telemetry import MapTelemetry


@dataclass(frozen=True)
class MapConfig:
    graph_k: int = 12
    top_k: int = 30
    frame_ms: int = 33
    telemetry_root: Path = Path("reports/livemap")


class OCSDMapViewer:
    def __init__(self, documents: list[Document], queries: list[Query], qrels: dict[str, dict[str, float]], encoder: TextEncoder, dataset_name: str, config: MapConfig | None = None) -> None:
        if len(documents) < 2:
            raise ValueError("at least two documents are required")
        if not queries:
            raise ValueError("at least one query is required")
        self.documents = documents
        self.queries = queries
        self.qrels = qrels
        self.encoder = encoder
        self.dataset_name = dataset_name
        self.config = config or MapConfig()
        texts = [doc.combined_text for doc in documents]
        self.encoder.fit(texts)
        self.doc_embeddings = self.encoder.encode_documents(texts)
        self.coords = pca2(self.doc_embeddings)
        self.edges = knn_edges(self.doc_embeddings, k=min(self.config.graph_k, len(documents) - 1), max_edges=3000)
        self.engine = OriginCoupledSpectralDiffusionSearch(self.doc_embeddings, items=self.documents, config=OCSDConfig(graph_k=min(self.config.graph_k, len(documents) - 1), candidate_count=None))
        self.telemetry = MapTelemetry(self.config.telemetry_root, run_name="ocsd_livemap")
        self.query_index = 0
        self.frame = 0
        self.playing = True
        self.selected_index: int | None = None
        self.current_probs = np.full(len(documents), 1.0 / len(documents), dtype=np.float64)
        self.target_probs = self.current_probs.copy()
        self.last_results: list[dict[str, object]] = []
        self.root = tk.Tk()
        self.root.title("OCSD LiveMap")
        self.root.geometry("1500x900")
        self.root.configure(bg="#0a1222")
        self._build_ui()
        self._compute()
        self._animate()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Panel.TFrame", background="#0a1222")
        style.configure("Panel.TLabel", background="#0a1222", foreground="#dce7ff")
        outer = ttk.Frame(self.root, style="Panel.TFrame")
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)
        left = ttk.Frame(outer, style="Panel.TFrame", width=380)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="OCSD LiveMap", style="Panel.TLabel", font=("Helvetica", 18, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.query_var = tk.StringVar(value=self.queries[0].text)
        self.query_id_var = tk.StringVar(value=self.queries[0].query_id)
        ttk.Entry(left, textvariable=self.query_var, width=48).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(left, text="Recompute", command=self._compute).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(left, text="Next Query", command=self._next_query).grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.status_var = tk.StringVar(value="green=relevant, red=non-relevant, blue=unjudged")
        ttk.Label(left, textvariable=self.status_var, style="Panel.TLabel", wraplength=350).grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.detail_var = tk.StringVar(value="Click a node for metadata.")
        ttk.Label(left, textvariable=self.detail_var, style="Panel.TLabel", wraplength=350).grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.tree = ttk.Treeview(left, columns=("rank", "doc", "prob", "rel", "title"), show="headings", height=24)
        for key, label, width in [("rank", "R", 38), ("doc", "Doc", 90), ("prob", "Prob", 70), ("rel", "Rel", 60), ("title", "Title", 170)]:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width)
        self.tree.grid(row=6, column=0, sticky="nsew")
        left.rowconfigure(6, weight=1)
        self.canvas = tk.Canvas(outer, bg="#08101f", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._render())
        self.canvas.bind("<Button-1>", self._click)

    def _active_qrels(self) -> dict[str, float]:
        return self.qrels.get(self.query_id_var.get(), {})

    def _next_query(self) -> None:
        self.query_index = (self.query_index + 1) % len(self.queries)
        q = self.queries[self.query_index]
        self.query_id_var.set(q.query_id)
        self.query_var.set(q.text)
        self._compute()

    def _compute(self) -> None:
        query = self.query_var.get().strip()
        if not query:
            return
        start = time.perf_counter()
        qvec = self.encoder.encode_queries([query])[0]
        self.last_results = self.engine.search(qvec, top_k=len(self.documents))
        latency_ms = (time.perf_counter() - start) * 1000.0
        probs = np.zeros(len(self.documents), dtype=np.float64)
        for row in self.last_results:
            probs[int(row["index"])] = float(row["probability"])
        self.target_probs = probs / max(float(probs.sum()), 1e-12)
        self._refresh_table()
        top_doc = self.last_results[0]["item"]
        rel = self._active_qrels().get(top_doc.doc_id)
        self.status_var.set(f"top={top_doc.doc_id} rel={rel if rel is not None else 'unjudged'} latency={latency_ms:.2f}ms")
        self.telemetry.log_query(self.query_id_var.get(), query, top_doc.doc_id, float(self.last_results[0]["probability"]), rel, latency_ms)
        self._render()

    def _refresh_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        qrel = self._active_qrels()
        for row in self.last_results[: self.config.top_k]:
            idx = int(row["index"])
            doc = self.documents[idx]
            rel = qrel.get(doc.doc_id)
            self.tree.insert("", tk.END, iid=str(idx), values=(row["rank"], doc.doc_id[:12], f"{float(row['probability']):.4f}", "?" if rel is None else f"{rel:g}", doc.title[:60]))

    def _render(self) -> None:
        self.canvas.delete("all")
        xy = scale_to_canvas(self.coords, max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height()))
        qrel = self._active_qrels()
        peak = float(max(self.current_probs.max(), 1e-12))
        for i, j, w in self.edges:
            self.canvas.create_line(*xy[i], *xy[j], fill=mix_hex("#14213b", "#355f9c", max(0.0, min(1.0, (w + 1.0) / 2.0))), width=1)
        for idx, doc in enumerate(self.documents):
            p = float(self.current_probs[idx])
            radius = 3.0 + 15.0 * ((p / peak) ** 0.72)
            x, y = xy[idx]
            fill = node_fill(judgment_base(qrel.get(doc.doc_id)), p, peak)
            outline = "#f7fbff" if idx == self.selected_index else "#0c1428"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, width=1.2)
        for rank, row in enumerate(self.last_results[:5], start=1):
            idx = int(row["index"])
            x, y = xy[idx]
            rr = 18 + rank * 4
            self.canvas.create_oval(x - rr, y - rr, x + rr, y + rr, outline="#eaf4ff", width=1.2)
            self.canvas.create_text(x + 8, y - 10, text=str(rank), fill="#eaf4ff", anchor="w", font=("Menlo", 10))
        self.canvas.create_text(14, 12, anchor="nw", fill="#dce7ff", font=("Menlo", 10), text="OCSD LiveMap: qrels-aware retrieval dynamics")

    def _animate(self) -> None:
        if self.playing:
            self.frame += 1
            self.current_probs = 0.86 * self.current_probs + 0.14 * self.target_probs
            self.current_probs /= max(float(self.current_probs.sum()), 1e-12)
            if self.frame % 15 == 0:
                focus = int(np.argmax(self.current_probs))
                entropy = -float(np.sum(self.current_probs * np.log(np.maximum(self.current_probs, 1e-12))))
                self.telemetry.log_frame(self.frame, float(self.current_probs.max()), entropy, self.documents[focus].doc_id)
            self._render()
        self.root.after(self.config.frame_ms, self._animate)

    def _click(self, event: tk.Event[tk.Misc]) -> None:
        xy = scale_to_canvas(self.coords, max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height()))
        d = np.sum((xy - np.array([[event.x, event.y]], dtype=np.float64)) ** 2, axis=1)
        self.selected_index = int(np.argmin(d))
        doc = self.documents[self.selected_index]
        rel = self._active_qrels().get(doc.doc_id)
        self.detail_var.set(f"{doc.doc_id} rel={rel if rel is not None else 'unjudged'} title={doc.title} text={doc.text[:240]}")
        self._render()

    def run(self) -> None:
        self.root.mainloop()
