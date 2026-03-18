# eval-amazon-search

Baseline ranking and evaluation experiments for the Amazon ESCI dataset, focused on:

- Okapi BM25 retrieval and reranking
- Local NDCG evaluation for ESCI labels
- Comparison against challenge results on AIcrowd

## What This Repo Does

- Loads ESCI data from `tasksource/esci`
- Builds a BM25 model from product text
- Evaluates predicted rankings against ESCI relevance labels
- Produces per-query NDCG outputs in `data/ndcg_scores.parquet`

## Evaluation Label Weights

This project uses the AIcrowd Task 1 gain mapping:

- `Exact = 1.0`
- `Substitute = 0.1`
- `Complement = 0.01`
- `Irrelevant = 0.0`

## Run

```bash
uv sync
python evaluation.py
```

## Current Local Results

| Storefront | Queries | Mean NDCG |
| --- | ---: | ---: |
| us | 22458 | 0.9146 |
| es | 3844 | 0.8798 |
| jp | 4667 | 0.8732 |
| **overall** | **30969** | **0.9039** |

## AIcrowd Reference

Official challenge leaderboard (for comparison context):

- https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search/leaderboards
