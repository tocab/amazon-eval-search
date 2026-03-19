"""Defines the random ranker class."""

import numpy as np
import polars as pl

from ranking.base_ranker import BaseRanker


class RandomRanker(BaseRanker):
    """Random ranker that returns random rankings."""

    def __init__(
        self,
        product_data: pl.DataFrame | None = None,
        text_column_name: str = "product_title",
        id_column_name: str = "product_id",
    ) -> None:
        if product_data is None:
            product_data = pl.DataFrame(schema={id_column_name: pl.String, text_column_name: pl.String})
        self.token_pattern = r"\w+"
        super().__init__(product_data, text_column_name, id_column_name)

    def query(self, query: str) -> pl.DataFrame:  # noqa: D102
        return self.empty_result.clone()

    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:  # noqa: D102
        return data.with_columns(pl.lit(np.random.rand(data.height)).alias("score"))
