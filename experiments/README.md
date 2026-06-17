# Experiments

## BEIR evaluation

Run a diagnostic subset:

```bash
python experiments/evaluate_beir.py \
  --dataset scifact \
  --max-docs 2000 \
  --max-queries 300 \
  --encoder tfidf_lite \
  --output-dir reports/beir_scifact_tfidf_lite
```

Run the full corpus if memory permits:

```bash
python experiments/evaluate_beir.py \
  --dataset scifact \
  --encoder tfidf_lite \
  --output-dir reports/beir_scifact_full
```

Run with a sentence-transformer encoder:

```bash
pip install sentence-transformers
python experiments/evaluate_beir.py \
  --dataset scifact \
  --encoder bge \
  --model-name BAAI/bge-small-en-v1.5 \
  --output-dir reports/beir_scifact_bge
```

The script writes:

- `metrics.csv`
- `runs.json`
- `report.md`

The report includes qrels-based retrieval metrics and a paired bootstrap comparison of OCSD-full against BM25 on nDCG@10.
