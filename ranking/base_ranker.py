"""Base ranker implementation."""

from abc import ABC, abstractmethod

import polars as pl


class BaseRanker(ABC):
    """BM25 scorer for full-corpus retrieval and candidate-set reranking."""

    def __init__(self, text_column_name: str, id_column_name: str) -> None:
        """Initialize BM25 state, tokenize documents, and precompute global IDF.

        Args:
            text_column_name: Text column used for tokenization and scoring.
            id_column_name: Unique product identifier column.

        Returns:
            None.
        """
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.token_pattern = r"\w+"
        self.empty_result = pl.DataFrame(
            schema={
                self.id_column_name: pl.Int32,
                self.text_column_name: pl.String,
                "score": pl.Float64,
            }
        )

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

    def fine_tune(
        self,
        data: pl.DataFrame,
        validation_query_rate: float = 0.1,
        epochs: int = 10,
        batch_size: int = 8192,
        num_negatives: int = 0,
    ) -> None:
        """Fine-tune the ranker with labeled training data."""
        raise NotImplementedError
