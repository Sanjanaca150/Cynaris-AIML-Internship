import json

import pandas as pd

from app.config import (
    CLICKSTREAM_FILE,
    REVIEWS_FILE,
    TEST_CASES_FILE,
)


def load_clickstream() -> pd.DataFrame:
    """Load the e-commerce clickstream dataset."""
    return pd.read_csv(CLICKSTREAM_FILE)


def load_product_reviews() -> pd.DataFrame:
    """Load the product reviews dataset."""
    return pd.read_csv(REVIEWS_FILE)


def load_ecommerce_tests() -> list:
    """Load the e-commerce test cases."""
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    clickstream = load_clickstream()
    reviews = load_product_reviews()
    test_cases = load_ecommerce_tests()

    print("=== OmniRetail AI Data Ingestion ===")
    print()
    print(f"Clickstream: {clickstream.shape}")
    print(f"Product Reviews: {reviews.shape}")
    print(f"E-commerce Test Cases: {len(test_cases)}")