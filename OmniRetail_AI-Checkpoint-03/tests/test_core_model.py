import json
import sys
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from app.data_loader import load_clickstream
from app.preprocessing import (
    TARGET_COLUMN,
    clean_clickstream,
    prepare_features,
)


MODEL_FILE = BASE_DIR / "models" / "best_core_model.joblib"
RESULTS_FILE = BASE_DIR / "docs" / "experiment_results.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"


def test_clickstream_dataset():
    """Verify the clickstream dataset is available."""

    df = load_clickstream()

    assert not df.empty
    assert len(df) == 1000
    assert TARGET_COLUMN in df.columns


def test_preprocessing_pipeline():
    """Verify the preprocessing pipeline works."""

    df = load_clickstream()

    cleaned = clean_clickstream(df)

    X, y, preprocessor = prepare_features(cleaned)

    assert len(cleaned) == 1000
    assert X.shape == (1000, 9)
    assert len(y) == 1000
    assert TARGET_COLUMN == "label_purchased"
    assert preprocessor is not None


def test_best_model_exists():
    """Verify the best core model was saved."""

    assert MODEL_FILE.exists()

    model = joblib.load(MODEL_FILE)

    assert model is not None


def test_best_model_prediction():
    """Verify the saved model can generate predictions."""

    df = load_clickstream()

    cleaned = clean_clickstream(df)

    X, _, _ = prepare_features(cleaned)

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X.head(10))

    assert len(predictions) == 10


def test_experiment_results_file():
    """Verify experiment results were saved correctly."""

    assert RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    assert results["experiment"] == (
        "OmniRetail_Checkpoint_3_Core_Model"
    )

    assert results["experiments_completed"] == 3

    assert results["selection_metric"] == "f1_score"

    assert len(results["all_results"]) == 3

    assert "best_configuration" in results


def test_three_configurations_recorded():
    """Verify at least two hyperparameter configurations were tested."""

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    configurations = [
        result["configuration"]
        for result in results["all_results"]
    ]

    assert len(configurations) >= 2
    assert len(set(configurations)) >= 2


def test_best_configuration_is_valid():
    """Verify the selected configuration is the best by F1."""

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    best = results["best_configuration"]

    all_configurations = [
        result["configuration"]
        for result in results["all_results"]
    ]

    assert best["configuration"] in all_configurations

    highest_f1 = max(
        result["f1_score"]
        for result in results["all_results"]
    )

    assert best["f1_score"] == highest_f1


def test_mlflow_database_exists():
    """Verify the MLflow tracking database exists."""

    assert MLFLOW_DB.exists()