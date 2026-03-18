"""Small local runner for trying BM25 ranking on the ESCI test split."""

from datasets import load_dataset
import polars as pl

from ranking.okapi_bm25 import OkapiBM25

LOCALE = "us"


def try_okapi():
    """Run a single BM25 query and print results plus query latency."""
    ds = load_dataset("tasksource/esci", split="test").to_polars()
    product_data = (
        ds.filter(pl.col("product_locale") == LOCALE)  # type: ignore
        .select(
            [
                "product_id",
                "product_locale",
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
                "product_text",
            ]
        )
        .unique()
    )

    okapi = OkapiBM25(product_data, "product_title", "product_id")
    result, took = okapi.timed_query("iphone")
    print(result)
    print(f"Took {took:.2f} ms")


if __name__ == "__main__":
    try_okapi()
