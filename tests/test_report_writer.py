from __future__ import annotations

from singularity_search.evaluation.metrics import RetrievalMetrics
from singularity_search.evaluation.runner import MethodResult, write_markdown_report


def test_markdown_report_writer(tmp_path) -> None:
    result = MethodResult(
        method="demo",
        metrics=RetrievalMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2),
        mean_latency_ms=1.5,
        p95_latency_ms=2.0,
    )
    write_markdown_report(tmp_path, "toy", "hash", [result], None, truncated=True)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "nDCG@10" in text
    assert "truncated diagnostic subset" in text
