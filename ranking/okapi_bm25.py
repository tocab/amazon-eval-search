"""Okapi BM25 ranking implementation based on Polars DataFrames."""

import math
import re
import time

import polars as pl

pl.Config.set_fmt_str_lengths(1000)


class OkapiBM25:
    """BM25 scorer for full-corpus retrieval and candidate-set reranking."""

    def __init__(self, product_data: pl.DataFrame, text_column_name: str, id_column_name: str):
        """Initialize BM25 state, tokenize documents, and precompute global IDF."""
        self.total_number_of_docs = len(product_data)
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.id_column_dtype = product_data.schema[id_column_name]
        self.text_column_dtype = product_data.schema[text_column_name]
        self.k_1 = 1.2
        self.b = 0.75
        self.token_pattern = r"\w+"
        self.empty_result = pl.DataFrame(
            schema={
                self.id_column_name: self.id_column_dtype,
                self.text_column_name: self.text_column_dtype,
                "score": pl.Float64,
            }
        )

        # Prepare product data
        prepared_product_data = self._prepare_product_data(product_data)
        self.product_data = prepared_product_data.lazy()
        self.avg_doc_len = prepared_product_data["document_length"].mean()
        self.idf_by_term = self._build_idf_lookup(prepared_product_data)
        self.oov_idf = self._calculate_idf(0)

    def _prepare_product_data(self, product_data: pl.DataFrame) -> pl.DataFrame:
        """Tokenize and normalize document text and add document length."""
        product_data = product_data.with_columns(
            pl.col(self.text_column_name)
            .str.to_lowercase()
            .str.extract_all(self.token_pattern)
            .alias("document_keywords")
        ).with_columns(pl.col("document_keywords").list.len().alias("document_length"))

        return product_data

    def _calculate_idf(self, num_docs_with_keyword):
        """Compute BM25 IDF from the number of documents containing a term."""
        return math.log((self.total_number_of_docs - num_docs_with_keyword + 0.5) / (num_docs_with_keyword + 0.5) + 1)

    def _build_idf_lookup(self, prepared_product_data: pl.DataFrame) -> dict[str, float]:
        """Build a lookup table of term -> global BM25 IDF."""
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
        """Score documents for a tokenized query and return sorted BM25 results."""
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
        """Retrieve and score products from the indexed corpus for a raw query."""
        query_keywords = re.findall(self.token_pattern, query.lower())
        if not query_keywords:
            return self.empty_result.clone()
        pl_scored = self._calculate_ranking(query_keywords, self.product_data)
        return pl_scored

    def timed_query(self, search_query: str) -> tuple[pl.DataFrame, float]:
        """Run retrieval and return `(results, elapsed_ms)`."""
        start = time.perf_counter()
        result = self.query(search_query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def rerank(self, query: str, data: pl.DataFrame) -> pl.DataFrame:
        """Rerank a provided candidate set for a query using BM25 scores."""
        query_keywords = re.findall(self.token_pattern, query.lower())
        if not query_keywords:
            return self.empty_result.clone()
        candidate_data = self._prepare_product_data(data).lazy()
        pl_scored = self._calculate_ranking(query_keywords, candidate_data)
        return pl_scored
