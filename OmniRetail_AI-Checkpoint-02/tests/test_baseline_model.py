from pathlib import Path

import joblib

from app.data_loader import (
    load_clickstream,
    load_ecommerce_tests,
    load_product_reviews,
)
from app.preprocessing import (
    TARGET_COLUMN,
    clean_clickstream,
    prepare_features,
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "baseline_purchase_model.joblib"
METRICS_FILE = BASE_DIR / "docs" / "baseline_metrics.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"


def test_clickstream_dataset():
    """Verify clickstream dataset is loaded correctly."""

    df = load_clickstream()

    assert not df.empty
    assert len(df) == 1000
    assert TARGET_COLUMN in df.columns


def test_product_reviews_dataset():
    """Verify product reviews dataset is available."""

    df = load_product_reviews()

    assert not df.empty
    assert len(df) == 800


def test_ecommerce_test_cases():
    """Verify provided test cases are available."""

    test_cases = load_ecommerce_tests()

    assert isinstance(test_cases, list)
    assert len(test_cases) == 50


def test_preprocessing_pipeline():
    """Verify clickstream preprocessing works."""

    df = load_clickstream()

    cleaned = clean_clickstream(df)

    X, y, preprocessor = prepare_features(cleaned)

    assert len(cleaned) == 1000
    assert X.shape == (1000, 9)
    assert len(y) == 1000
    assert TARGET_COLUMN == "label_purchased"
    assert preprocessor is not None


def test_saved_baseline_model():
    """Verify trained baseline model was saved."""

    assert MODEL_FILE.exists()

    model = joblib.load(MODEL_FILE)

    assert model is not None


def test_baseline_metrics_file():
    """Verify baseline metrics file was created."""

    assert METRICS_FILE.exists()

    import json

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    assert metrics["model"] == "Logistic Regression"
    assert metrics["total_records"] == 1000
    assert metrics["training_records"] == 800
    assert metrics["testing_records"] == 200
    assert metrics["feature_count"] == 9

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics


def test_mlflow_database():
    """Verify MLflow tracking database was created."""

    assert MLFLOW_DB.exists()