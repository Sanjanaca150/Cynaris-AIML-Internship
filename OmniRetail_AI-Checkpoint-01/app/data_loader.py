import json

import pandas as pd

from app.config import REVIEWS_FILE, TEST_CASES_FILE


def load_product_reviews() -> pd.DataFrame:
    """Load the product reviews dataset."""
    return pd.read_csv(REVIEWS_FILE)


def load_ecommerce_tests() -> list:
    """Load the e-commerce test cases."""
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)