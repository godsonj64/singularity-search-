#!/usr/bin/env bash
set -euo pipefail

python experiments/evaluate_beir.py \
  --dataset scifact \
  --encoder bge \
  --model-name BAAI/bge-small-en-v1.5 \
  --output-dir reports/beir_scifact_bge_full
