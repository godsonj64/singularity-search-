#!/usr/bin/env bash
set -euo pipefail

python experiments/evaluate_beir.py \
  --dataset scifact \
  --max-docs 2000 \
  --max-queries 300 \
  --encoder tfidf_lite \
  --output-dir reports/beir_scifact_tfidf_lite_diagnostic
