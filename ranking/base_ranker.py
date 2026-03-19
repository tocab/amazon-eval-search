"""Base ranker implementation."""

from abc import ABC, abstractmethod

import polars as pl


class BaseRanker(ABC):
    """BM25 scorer for full-corpus retrieval and candidate-set reranking."""

    def __init__(self, product_data: pl.DataFrame, text_column_name: str, id_column_name: str) -> None:
        """Initialize BM25 state, tokenize documents, and precompute global IDF.

        Args:
            product_data: Product table used as BM25 corpus.
            text_column_name: Text column used for tokenization and scoring.
            id_column_name: Unique product identifier column.

        Returns:
            None.
        """
        self.total_number_of_docs = len(product_data)
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.id_column_dtype = product_data.schema[id_column_name]
        self.text_column_dtype = product_data.schema[text_column_name]
        self.token_pattern = r"\w+"
        # Prepare product data
        prepared_product_data = self._prepare_product_data(product_data)
        self.product_data = prepared_product_data.lazy()
        self.empty_result = pl.DataFrame(
            schema={
                self.id_column_name: self.id_column_dtype,
                self.text_column_name: self.text_column_dtype,
                "score": pl.Float64,
            }
        )

    def _prepare_product_data(self, product_data: pl.DataFrame) -> pl.DataFrame:
        """Tokenize and normalize document text and add document length.

        Args:
            product_data: Input product table.

        Returns:
            pl.DataFrame: Product table with `document_keywords` and `document_length`.
        """
        product_data = product_data.with_columns(
            pl.col(self.text_column_name)
            .str.to_lowercase()
            .str.extract_all(self.token_pattern)
            .alias("document_keywords")
        ).with_columns(pl.col("document_keywords").list.len().alias("document_length"))

        return product_data

    @abstractmethod
    def query(self, query: str) -> pl.DataFrame:
        """Retrieve and score products from the indexed corpus for a raw query.

        Args:
            query: Raw query string.

        Returns:
            pl.DataFrame: Ranked products with BM25 scores.
        """
        raise NotImplementedError

    @abstractmethod
    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:
        """Rerank a provided candidate set for a query using BM25 scores.

        Args:
            query: Raw query string.
            data: Candidate product rows for the query.

        Returns:
            pl.DataFrame: Candidate products ranked by BM25 score.
        """
        raise NotImplementedError
