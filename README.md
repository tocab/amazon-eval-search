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

## Results

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Storefront</th>
      <th rowspan="2" align="right">Queries</th>
      <th colspan="2" align="center">Mean NDCG</th>
    </tr>
    <tr>
      <th align="right">Random</th>
      <th align="right">Okapi BM25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>us</td>
      <td align="right">8956</td>
      <td align="right">0.7446</td>
      <td align="right">0.8182</td>
    </tr>
    <tr>
      <td>es</td>
      <td align="right">2417</td>
      <td align="right">0.7397</td>
      <td align="right">0.8226</td>
    </tr>
    <tr>
      <td>jp</td>
      <td align="right">3123</td>
      <td align="right">0.7678</td>
      <td align="right">0.8265</td>
    </tr>
    <tr>
      <td><strong>overall</strong></td>
      <td align="right"><strong>14496</strong></td>
      <td align="right">0.7497</td>
      <td align="right"><strong>0.8206</strong></td>
    </tr>
  </tbody>
</table>

## AIcrowd Reference

Official challenge leaderboard (for comparison context):

- https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search/leaderboards
