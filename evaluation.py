"""Evaluation utilities for BM25 retrieval and reranking on ESCI data."""

from dataclasses import dataclass
import logging
from enum import Enum
from pathlib import Path
from typing import Any
from typing import cast

import polars as pl
import yaml

from datasets import load_dataset
from sentence_transformers.cross_encoder import CrossEncoder

from ranking.base_ranker import BaseRanker
from ranking.e5_bi_encoder import E5BiEncoder
from ranking.msmarco import MSMarcoRanker
from ranking.okapi_bm25 import OkapiBM25
from ranking.random_ranker import RandomRanker
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

DEFAULT_CONFIG_PATH = Path("config/evaluation/evaluation.yml")

ESCI_WEIGHTS = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}

logger = logging.getLogger(__name__)


class RankerType(str, Enum):
    """Supported reranker implementations."""

    OKAPI = "okapi"
    RANDOM = "random"
    MSMARCO = "msmarco"
    E5 = "e5"


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation runtime config loaded from YAML."""

    model_name: RankerType
    model_path: str
    locales: list[str]
    text_column_name: str
    id_column_name: str


def load_evaluation_config(config_path: Path = DEFAULT_CONFIG_PATH) -> EvaluationConfig:
    """Load and validate evaluation config from YAML."""
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")

    config_data = cast(dict[str, Any], raw_config)
    model_name_raw = config_data.get("model_name")
    model_path_raw = config_data.get("model_path", "")
    locales_raw = config_data.get("locales")
    text_column_name_raw = config_data.get("text_column_name")
    id_column_name_raw = config_data.get("id_column_name")

    if not isinstance(model_name_raw, str):
        raise ValueError("Config value `model_name` must be a string.")
    if not isinstance(model_path_raw, str):
        raise ValueError("Config value `model_path` must be a string.")
    if not isinstance(locales_raw, list) or any(not isinstance(locale, str) for locale in locales_raw):
        raise ValueError("Config value `locales` must be a list of strings.")
    if not isinstance(text_column_name_raw, str):
        raise ValueError("Config value `text_column_name` must be a string.")
    if not isinstance(id_column_name_raw, str):
        raise ValueError("Config value `id_column_name` must be a string.")

    try:
        model_name = RankerType(model_name_raw.lower())
    except ValueError as error:
        supported = ", ".join(ranker.value for ranker in RankerType)
        raise ValueError(f"Unsupported `model_name`: {model_name_raw}. Supported values: {supported}") from error

    return EvaluationConfig(
        model_name=model_name,
        model_path=model_path_raw,
        locales=locales_raw,
        text_column_name=text_column_name_raw,
        id_column_name=id_column_name_raw,
    )


def load_data(locales: list[str], split: str = "train") -> pl.DataFrame:
    """Load one ESCI split, filter locales, and map labels to numeric gains.

    Args:
        locales: Storefront locales to keep (for example: `["us", "es", "jp"]`).
        split: ESCI split name (for example: `"train"` or `"test"`).

    Returns:
        pl.DataFrame: Filtered split with an added `esci_weight` column.
    """
    ds = cast(pl.DataFrame, load_dataset("tasksource/esci", split=split).to_polars())
    # Task 1: filter by small_version == 1
    ds = ds.filter(pl.col("small_version") == 1)
    ds = ds.filter(pl.col("product_locale").is_in(locales))
    ds = ds.with_columns(pl.col("esci_label").replace(ESCI_WEIGHTS).cast(pl.Float32).alias("esci_weight"))
    return ds


def create_product_data(dataset: pl.DataFrame, id_column_name: str, text_column_name: str) -> pl.DataFrame:
    """Create a unique product table used to initialize BM25.

    Args:
        dataset: Input ESCI rows.

    Returns:
        pl.DataFrame: Deduplicated product table with product text fields.
    """
    selected_columns = []
    for column in [
        id_column_name,
        text_column_name,
        "product_locale",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
        "product_text",
    ]:
        if column in dataset.columns and column not in selected_columns:
            selected_columns.append(column)

    product_data = dataset.select(selected_columns).unique()

    return product_data


def evaluate_retrieval(config: EvaluationConfig) -> pl.DataFrame:
    """Evaluate full-corpus retrieval by joining BM25 results to judged pairs.

    Args:
        None.

    Returns:
        pl.DataFrame: Per-query NDCG table.
    """
    ds_train = load_data(config.locales, "train")
    ds_test = load_data(config.locales, "test")
    product_data = create_product_data(ds_train, config.id_column_name, config.text_column_name)

    okapi = OkapiBM25(product_data, config.text_column_name, config.id_column_name)

    query_scores = []
    query_groups = ds_test.partition_by(["query_id", "query"], as_dict=True)
    for index, group in tqdm(query_groups.items(), total=len(query_groups), desc="Evaluating queries"):
        query_id = index[0]
        query = index[1]
        result = okapi.query(query)
        comparison = group.join(
            result.select(config.id_column_name, "score"),
            on=config.id_column_name,
            how="left",
        ).fill_null(0.0)
        score = ndcg_score([comparison["esci_weight"]], [comparison["score"]])
        query_scores.append({"query_id": query_id, "query": query, "ndcg_score": score})

    query_scores = pl.from_dicts(query_scores)
    print(query_scores.sort("ndcg_score"))
    query_scores.write_parquet("data/ndcg_scores.parquet")
    return query_scores


def _create_reranker(config: EvaluationConfig, product_data: pl.DataFrame) -> BaseRanker:
    """Create a reranker instance by name."""
    ranker_name = config.model_name
    if ranker_name == RankerType.OKAPI:
        return OkapiBM25(product_data, config.text_column_name, config.id_column_name)
    if ranker_name == RankerType.RANDOM:
        return RandomRanker(product_data, config.text_column_name, config.id_column_name)
    if ranker_name == RankerType.MSMARCO:
        ranker = MSMarcoRanker(product_data, config.text_column_name, config.id_column_name)
        if config.model_path:
            ranker.model = CrossEncoder(config.model_path, num_labels=1, max_length=256)
        return ranker
    if ranker_name == RankerType.E5:
        if config.model_path:
            ranker = E5BiEncoder(config.model_path, config.text_column_name, config.id_column_name)
        else:
            ranker = E5BiEncoder(text_column_name=config.text_column_name, id_column_name=config.id_column_name)
        return ranker
    raise ValueError(f"Unknown ranker: {ranker_name}")


def evaluate_rerank(config: EvaluationConfig) -> pl.DataFrame:
    """Evaluate reranking within each query's provided candidate set.

    Args:
        config: Evaluation runtime config.

    Returns:
        pl.DataFrame: Per-query NDCG table.
    """
    ds_test = load_data(config.locales, "test")
    ds_train = load_data(config.locales, "train")

    product_data = create_product_data(ds_train, config.id_column_name, config.text_column_name)
    ranker = _create_reranker(config, product_data)

    query_scores = []
    query_groups = ds_test.partition_by(["query_id", "query"], as_dict=True)
    for index, group in tqdm(query_groups.items(), total=len(query_groups), desc="Evaluating queries"):
        query_id = index[0]
        query = index[1]
        result = ranker.rerank(query, group)
        comparison = group.join(
            result.select(config.id_column_name, "score"),
            on=config.id_column_name,
            how="left",
        ).fill_null(0.0)
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
    config = load_evaluation_config()
    evaluate_rerank(config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
