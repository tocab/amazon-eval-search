"""NDCG evaluators for cross-encoder and bi-encoder rankers."""

import logging

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class CrossEncoderNDCGEvaluator(SentenceEvaluator):
    """Compute per-query NDCG for a cross-encoder model."""

    def __init__(
        self,
        dataset: pl.DataFrame,
        at_k: int | None = None,
        text_column_name: str = "product_title",
        score_column_name: str = "esci_weight",
        name: str = "",
        batch_size: int = 64,
    ):
        """Initialize evaluator inputs and scoring configuration."""
        super().__init__()
        self.dataset = dataset
        self.score_column_name = score_column_name
        self.model_input = [[row[0], row[1]] for row in dataset.select(["query", text_column_name]).iter_rows()]
        self.scores = [score for score in dataset[score_column_name]]
        self.at_k = at_k
        self.name = name
        self.batch_size = batch_size
        self.primary_metric = f"{self.name}_ndcg@{self.at_k}"

    def __call__(  # type: ignore[override]
        self, model: CrossEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """Score the model and return NDCG aggregated across query groups."""
        logger.info(f"NDCGEvaluator: Evaluating the model on the {self.name}:")

        pred_scores = model.predict(
            self.model_input, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=True
        )

        eval_df = self.dataset.with_columns(pl.lit(pred_scores).alias("prediction"))

        ndcg_scores = []
        for _query_id, group_df in eval_df.group_by("query_id"):
            if group_df.height < 2:
                continue
            score = ndcg_score(
                [group_df[self.score_column_name].to_list()], [group_df["prediction"].to_list()], k=self.at_k
            )
            ndcg_scores.append(score)
        final_ndcg_score = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

        return {self.primary_metric: final_ndcg_score}


class BiEncoderNDCGEvaluator(SentenceEvaluator):
    """Compute per-query NDCG for a bi-encoder model."""

    def __init__(
        self,
        dataset: pl.DataFrame,
        at_k: int | None = None,
        text_column_name: str = "product_title",
        score_column_name: str = "esci_weight",
        name: str = "",
        batch_size: int = 64,
        query_prefix: str = "query",
        document_prefix: str = "passage",
    ):
        """Initialize evaluator inputs and scoring configuration."""
        super().__init__()
        self.dataset = dataset
        self.score_column_name = score_column_name
        self.query_model_input = [f"{query_prefix}: {query}" for query in dataset["query"]]
        self.document_model_input = [f"{document_prefix}: {document}" for document in dataset[text_column_name]]
        self.scores = [score for score in dataset[score_column_name]]
        self.at_k = at_k
        self.name = name
        self.batch_size = batch_size
        self.primary_metric = f"{self.name}_ndcg@{self.at_k}"

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """Score the model and return NDCG aggregated across query groups."""
        logger.info(f"NDCGEvaluator: Evaluating the model on the {self.name}")
        logger.info("Calculate query embeddings:")
        query_encodings = model.encode(
            self.query_model_input, show_progress_bar=True, batch_size=self.batch_size, normalize_embeddings=True
        )
        logger.info("Calculate document embeddings:")
        document_encodings = model.encode(
            self.document_model_input, show_progress_bar=True, batch_size=self.batch_size, normalize_embeddings=True
        )

        # Calculate cosine similarity
        # Since query and document vectors are normalized, cosine similarity simplifies to dot product
        scores = np.sum(query_encodings * document_encodings, axis=1)

        eval_df = self.dataset.with_columns(pl.lit(scores).alias("prediction"))

        ndcg_scores = []
        for _query_id, group_df in eval_df.group_by("query_id"):
            if group_df.height < 2:
                continue
            score = ndcg_score(
                [group_df[self.score_column_name].to_list()], [group_df["prediction"].to_list()], k=self.at_k
            )
            ndcg_scores.append(score)
        final_ndcg_score = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        return {self.primary_metric: final_ndcg_score}
