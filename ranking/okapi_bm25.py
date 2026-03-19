"""Okapi BM25 ranking implementation based on Polars DataFrames."""

import math
import re
import time

import polars as pl

from ranking.base_ranker import BaseRanker


class OkapiBM25(BaseRanker):
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
        super().__init__(product_data, text_column_name, id_column_name)

        self.k_1 = 1.2
        self.b = 0.75

        prepared_product_data = self.product_data.collect()
        self.avg_doc_len = prepared_product_data["document_length"].mean()
        self.idf_by_term = self._build_idf_lookup(prepared_product_data)
        self.oov_idf = self._calculate_idf(0)

    def _calculate_idf(self, num_docs_with_keyword: int) -> float:
        """Compute BM25 IDF from the number of documents containing a term.

        Args:
            num_docs_with_keyword: Document frequency of the term.

        Returns:
            float: BM25 IDF value.
        """
        return math.log((self.total_number_of_docs - num_docs_with_keyword + 0.5) / (num_docs_with_keyword + 0.5) + 1)

    def _build_idf_lookup(self, prepared_product_data: pl.DataFrame) -> dict[str, float]:
        """Build a lookup table of term -> global BM25 IDF.

        Args:
            prepared_product_data: Tokenized corpus with document keywords.

        Returns:
            dict[str, float]: Mapping from token to global BM25 IDF.
        """
        term_doc_freq = (
            prepared_product_data.select(self.id_column_name, "document_keywords")
            .explode("document_keywords")
            .drop_nulls("document_keywords")
            .unique([self.id_column_name, "document_keywords"])
            .group_by("document_keywords")
            .len(name="doc_freq")
        )

        return {row[0]: self._calculate_idf(row[1]) for row in term_doc_freq.iter_rows()}

    def _calculate_ranking(self, query_keywords: list[str], data: pl.LazyFrame) -> pl.DataFrame:
        """Score documents for a tokenized query and return sorted BM25 results.

        Args:
            query_keywords: Tokenized query terms.
            data: Candidate documents prepared as a lazy frame.

        Returns:
            pl.DataFrame: Ranked products with BM25 scores.
        """
        pl_keyword_matches = []
        for query_keyword in query_keywords:
            pl_keyword_matches.append(
                data.with_columns(pl.col("document_keywords").list.count_matches(query_keyword).alias("count_matches"))
                .filter(pl.col("count_matches") > 0)
                .with_columns(
                    (
                        (pl.col("count_matches") * (self.k_1 + 1))
                        / (
                            pl.col("count_matches")
                            + self.k_1 * (1 - self.b + self.b * pl.col("document_length") / self.avg_doc_len)
                        )
                    ).alias("score")
                )
            )

        pl_keyword_matches = pl.collect_all(pl_keyword_matches)

        pl_scored = []
        for query_keyword, pl_keyword_match in zip(query_keywords, pl_keyword_matches, strict=True):
            idf = self.idf_by_term.get(query_keyword, self.oov_idf)

            pl_keyword_match = pl_keyword_match.with_columns(
                pl.lit(query_keyword).alias("keyword"), pl.col("score") * idf
            )
            pl_scored.append(pl_keyword_match)

        if not pl_scored:
            pl_scored = self.empty_result.clone()
        else:
            pl_scored = (
                pl.concat(pl_scored)
                .group_by(self.id_column_name, maintain_order=False)
                .agg(pl.col(self.text_column_name).first(), pl.col("score").sum())
                .sort("score", descending=True)
            )

        return pl_scored

    def query(self, query: str) -> pl.DataFrame:
        """Retrieve and score products from the indexed corpus for a raw query.

        Args:
            query: Raw query string.

        Returns:
            pl.DataFrame: Ranked products with BM25 scores.
        """
        query_keywords = re.findall(self.token_pattern, query.lower())
        if not query_keywords:
            return self.empty_result.clone()
        pl_scored = self._calculate_ranking(query_keywords, self.product_data)
        return pl_scored

    def timed_query(self, search_query: str) -> tuple[pl.DataFrame, float]:
        """Run retrieval and return `(results, elapsed_ms)`.

        Args:
            search_query: Raw query string.

        Returns:
            tuple[pl.DataFrame, float]: Ranked products and elapsed time in milliseconds.
        """
        start = time.perf_counter()
        result = self.query(search_query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:
        """Rerank a provided candidate set for a query using BM25 scores.

        Args:
            query: Raw query string.
            data: Candidate product rows for the query.

        Returns:
            pl.DataFrame: Candidate products ranked by BM25 score.
        """
        query_keywords = re.findall(self.token_pattern, query.lower())
        if not query_keywords:
            return self.empty_result.clone()
        candidate_data = self._prepare_product_data(data).lazy()
        pl_scored = self._calculate_ranking(query_keywords, candidate_data)
        return pl_scored
