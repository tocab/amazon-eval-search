"""Defines the e5 bi encoder class."""

from datetime import datetime
import logging
from pathlib import Path

import polars as pl
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from tqdm.auto import tqdm

from evaluators.ndcg_evaluator import BiEncoderNDCGEvaluator
from ranking.base_ranker import BaseRanker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


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

    def _create_training_data(
        self,
        data: pl.DataFrame,
        num_negatives: int,
        negative_threshold: float = 1.0,
    ) -> Dataset:
        # Clean empty queries and titles
        data = data.filter(
            (pl.col("query").str.strip_chars().str.len_chars() > 0)
            & (pl.col(self.text_column_name).str.strip_chars().str.len_chars() > 0)
        )

        # Assign labels
        data = data.with_columns((pl.col("esci_weight") >= negative_threshold).alias("label"))

        # Create dataset in the format of query, positive, negative
        examples = []
        positive_data = data.filter(pl.col("label"))

        num_groups = positive_data.group_by(["query"]).len().height
        for group_keys, group_data in tqdm(
            positive_data.group_by(["query"]),
            total=num_groups,
            desc="Building training examples",
        ):
            query = group_keys[0]
            group_data = group_data.select(["query", "product_text"])
            group_data.columns = ["query", "positive"]

            negatives = (
                data.filter(~pl.col("label") & (pl.col("query") == query))
                .select(["query", "product_text"])
                .head(num_negatives)
            )
            negatives.columns = ["query", "negative"]
            count_negatives = len(negatives)

            # Find more negatives if necessary
            num_missing_negatives = num_negatives - count_negatives
            if num_missing_negatives > 0:
                additional_negatives = (
                    data.filter(~(pl.col("query") == query))
                    .select(["query", "product_text"])
                    .sample(num_missing_negatives)
                    .with_columns(pl.lit(query).alias("query"))
                )
                additional_negatives.columns = ["query", "negative"]
                negatives = pl.concat([negatives, additional_negatives])

            # Negatives to wide format
            negatives = negatives.with_columns(negative_number=pl.int_range(pl.len())).pivot(
                "negative_number",
                on_columns=[i for i in range(num_negatives)],
                index=["query"],
                values="negative",
                aggregate_function="first",
            )

            # Join on example
            if len(negatives) > 0:
                group_data = group_data.join(negatives, on="query")

            examples.append(group_data)

        examples = pl.concat(examples)
        dataset = Dataset.from_polars(examples)
        return dataset

    def fine_tune(
        self,
        data: pl.DataFrame,
        validation_query_rate: float = 0.1,
        epochs: int = 10,
        batch_size: int = 8192,
        num_negatives: int = 0,
    ) -> None:
        """Fine-tune the E5 bi-encoder and persist the trained model artifacts."""
        output_dir = PROJECT_ROOT / "artifacts" / "e5-bi-encoder"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        unique_queries = data["query"].unique()

        count_train_queries = int(len(unique_queries) * (1 - validation_query_rate))
        train_queries = unique_queries.slice(0, count_train_queries)
        validation_queries = unique_queries.slice(count_train_queries)

        train_data = data.filter(pl.col("query").is_in(train_queries))
        validation_data = data.filter(pl.col("query").is_in(validation_queries))

        # Create training data
        logger.info("Create training data")
        train_dataset = self._create_training_data(train_data, num_negatives)

        logger.info("Initial training data evaluation")
        train_data_evaluator = BiEncoderNDCGEvaluator(train_data, name="train_ndcg_evaluator")
        train_score = train_data_evaluator(self.model)
        logger.info(f"Initial training score: {train_score}")

        logger.info("Initial validation data evaluation")
        validation_data_evaluator = BiEncoderNDCGEvaluator(validation_data, name="validation_ndcg_evaluator")
        validation_score = validation_data_evaluator(self.model)
        logger.info(f"Initial training score: {validation_score}")

        train_args = SentenceTransformerTrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            eval_steps=10,
            save_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            report_to="none",
            logging_steps=10,
            disable_tqdm=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            prompts={"query": "query: ", "positive": "passage: "}
            | {f"negative_{i}": "passage: " for i in range(num_negatives)},
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            loss=losses.CachedMultipleNegativesRankingLoss(model=self.model),
            evaluator=[train_data_evaluator, validation_data_evaluator],
        )

        trainer.train()

        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(final_dir))
        logger.info("Training complete. Final model saved to %s", final_dir)

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
