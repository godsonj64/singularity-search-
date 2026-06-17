from __future__ import annotations

import importlib.util
from pathlib import Path


def test_evaluate_beir_script_exists() -> None:
    path = Path("experiments/evaluate_beir.py")
    assert path.exists() or importlib.util.find_spec("singularity_search.evaluation.runner") is not None
