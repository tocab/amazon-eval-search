"""Defines the e5 bi encoder class."""

import polars as pl
from sentence_transformers import SentenceTransformer

from ranking.base_ranker import BaseRanker


class E5BiEncoder(BaseRanker):
    """E5 Bi encoder that returns rankings based on similarity of texts."""

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
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")

    def query(self, query: str) -> pl.DataFrame:  # noqa: D102
        return self.empty_result.clone()

    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:  # noqa: D102
        # Encode query
        query_vector = self.model.encode(f"query: {query}")

        # Encode passages
        passages = []
        for text in data[self.text_column_name]:
            passages.append(f"passage: {text}")

        passage_vectors = self.model.encode(passages)

        # cosine similarity for each passage vs query
        scores = passage_vectors @ query_vector

        return data.with_columns(pl.lit(scores).alias("score"))
