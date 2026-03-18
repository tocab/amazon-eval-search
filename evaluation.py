"""Evaluation utilities for BM25 retrieval and reranking on ESCI data."""

from typing import cast

import polars as pl

from datasets import load_dataset

from ranking.okapi_bm25 import OkapiBM25
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

LOCALES = ["us", "es", "jp"]

ESCI_WEIGHTS = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}


def load_data(locales: list[str], split: str = "train") -> pl.DataFrame:
    """Load one ESCI split, filter locales, and map labels to numeric gains.

    Args:
        locales: Storefront locales to keep (for example: `["us", "es", "jp"]`).
        split: ESCI split name (for example: `"train"` or `"test"`).

    Returns:
        pl.DataFrame: Filtered split with an added `esci_weight` column.
    """
    ds = cast(pl.DataFrame, load_dataset("tasksource/esci", split=split).to_polars())
    ds = ds.filter(pl.col("product_locale").is_in(locales))
    ds = ds.with_columns(pl.col("esci_label").replace(ESCI_WEIGHTS).cast(pl.Float32).alias("esci_weight"))
    return ds


def create_product_data(dataset: pl.DataFrame) -> pl.DataFrame:
    """Create a unique product table used to initialize BM25.

    Args:
        dataset: Input ESCI rows.

    Returns:
        pl.DataFrame: Deduplicated product table with product text fields.
    """
    product_data = dataset.select(
        [
            "product_id",
            "product_locale",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_text",
        ]
    ).unique()

    return product_data


def evaluate_retrieval() -> pl.DataFrame:
    """Evaluate full-corpus retrieval by joining BM25 results to judged pairs.

    Args:
        None.

    Returns:
        pl.DataFrame: Per-query NDCG table.
    """
    ds_train = load_data(LOCALES, "train")
    ds_test = load_data(LOCALES, "test")
    product_data = create_product_data(ds_train)

    okapi = OkapiBM25(product_data, "product_title", "product_id")

    query_scores = []
    query_groups = ds_test.partition_by(["query_id", "query"], as_dict=True)
    for index, group in tqdm(query_groups.items(), total=len(query_groups), desc="Evaluating queries"):
        query_id = index[0]
        query = index[1]
        result = okapi.query(query)
        comparison = group.join(result.select("product_id", "score"), on="product_id", how="left").fill_null(0.0)
        score = ndcg_score([comparison["esci_weight"]], [comparison["score"]])
        query_scores.append({"query_id": query_id, "query": query, "ndcg_score": score})

    query_scores = pl.from_dicts(query_scores)
    print(query_scores.sort("ndcg_score"))
    query_scores.write_parquet("data/ndcg_scores.parquet")
    return query_scores


def evaluate_rerank() -> pl.DataFrame:
    """Evaluate reranking within each query's provided candidate set.

    Args:
        None.

    Returns:
        pl.DataFrame: Per-query NDCG table.
    """
    ds_train = load_data(LOCALES, "train")
    ds_test = load_data(LOCALES, "test")
    product_data = create_product_data(ds_train)

    okapi = OkapiBM25(product_data, "product_title", "product_id")

    query_scores = []
    query_groups = ds_test.partition_by(["query_id", "query"], as_dict=True)
    for index, group in tqdm(query_groups.items(), total=len(query_groups), desc="Evaluating queries"):
        query_id = index[0]
        query = index[1]
        result = okapi.rerank(query, group)
        comparison = group.join(result.select("product_id", "score"), on="product_id", how="left").fill_null(0.0)
        score = ndcg_score([comparison["esci_weight"]], [comparison["score"]])
        query_scores.append({"query_id": query_id, "query": query, "ndcg_score": score})

    query_scores = pl.from_dicts(query_scores)
    print(query_scores.sort("ndcg_score"))
    print("Avg ndcg score:", query_scores["ndcg_score"].mean())
    query_scores.write_parquet("data/ndcg_scores.parquet")
    return query_scores


def main() -> None:
    """Run the default local evaluation entry point.

    Args:
        None.

    Returns:
        None.
    """
    evaluate_rerank()


if __name__ == "__main__":
    main()
