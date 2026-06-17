"""Telemetry and report helpers for OCSD maps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class MapTelemetry:
    root: Path
    run_name: str = "livemap"

    def __post_init__(self) -> None:
        self.run_dir = self.root / f"{utc_slug()}_{self.run_name}"
        self.snapshots_dir = self.run_dir / "snapshots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_csv = self.run_dir / "map_events.csv"
        self.frames_csv = self.run_dir / "frame_metrics.csv"
        self._init(self.events_csv, ["timestamp_utc", "query_id", "query", "top_doc_id", "top_probability", "top_relevance", "latency_ms"])
        self._init(self.frames_csv, ["timestamp_utc", "frame", "peak_probability", "entropy", "focus_doc_id"])

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init(self, path: Path, header: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(header)

    def append(self, path: Path, row: list[object]) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(row)

    def log_query(self, query_id: str, query: str, top_doc_id: str, top_probability: float, top_relevance: float | None, latency_ms: float) -> None:
        rel = "unjudged" if top_relevance is None else f"{top_relevance:.4f}"
        self.append(self.events_csv, [self._now(), query_id, query, top_doc_id, f"{top_probability:.8f}", rel, f"{latency_ms:.4f}"])

    def log_frame(self, frame: int, peak_probability: float, entropy: float, focus_doc_id: str) -> None:
        self.append(self.frames_csv, [self._now(), frame, f"{peak_probability:.8f}", f"{entropy:.8f}", focus_doc_id])

    def write_report(self, dataset: str, query_count: int, document_count: int) -> Path:
        report = self.run_dir / "report.md"
        report.write_text(
            "\n".join(
                [
                    "# OCSD LiveMap Report",
                    "",
                    f"- Dataset: `{dataset}`",
                    f"- Documents: `{document_count}`",
                    f"- Queries: `{query_count}`",
                    f"- Query events: `{self.events_csv.name}`",
                    f"- Frame metrics: `{self.frames_csv.name}`",
                    "",
                    "This report is a visualization diagnostic artifact. Retrieval quality must still be evaluated with qrels-based metrics.",
                ]
            ),
            encoding="utf-8",
        )
        return report
