from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def test_reviews_dataset_exists_and_is_valid():
    path = DATA_DIR / "aiml_product_reviews.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 800
    assert len(df.columns) == 11
    assert "review_id" in df.columns
    assert "product_id" in df.columns
    assert "rating" in df.columns
    assert "review_text" in df.columns


def test_reviews_dataset_has_no_missing_values():
    path = DATA_DIR / "aiml_product_reviews.csv"

    df = pd.read_csv(path)

    assert df.isna().sum().sum() == 0


def test_ecommerce_test_cases_exist_and_are_valid():
    path = DATA_DIR / "aiml_ecommerce_tests.json"

    assert path.exists()

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 50

    for item in data:
        assert "test_id" in item
        assert "input" in item
        assert "expected_label" in item
        assert "model_target" in item

def test_data_loader_reviews():
    from app.data_loader import load_product_reviews

    df = load_product_reviews()

    assert df.shape == (800, 11)


def test_data_loader_ecommerce_tests():
    from app.data_loader import load_ecommerce_tests

    data = load_ecommerce_tests()

    assert len(data) == 50