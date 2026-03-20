"""Defines the ms marco ranker class."""

import polars as pl
from sentence_transformers import CrossEncoder

from ranking.base_ranker import BaseRanker


class MSMarcoRanker(BaseRanker):
    """MSMarco ranker that returns rankings based on cross encodings of texts."""

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

        # Load model
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            num_labels=1,
            max_length=256,
        )

    def query(self, query: str) -> pl.DataFrame:  # noqa: D102
        return self.empty_result.clone()

    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:  # noqa: D102
        pairs = []
        for text in data[self.text_column_name]:
            pairs.append((query, text))
        scores = self.model.predict(pairs)
        return data.with_columns(pl.lit(scores).alias("score"))
