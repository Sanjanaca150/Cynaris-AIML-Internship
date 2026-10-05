import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.data_loader import load_clickstream
from app.preprocessing import clean_clickstream, prepare_features


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
MLFLOW_DB = BASE_DIR / "mlflow.db"

MODEL_FILE = MODELS_DIR / "baseline_purchase_model.joblib"
METRICS_FILE = DOCS_DIR / "baseline_metrics.json"


# ============================================================
# Baseline model training
# ============================================================

def train_baseline():
    """Train and evaluate the baseline purchase prediction model."""

    print("Loading clickstream dataset...")

    # --------------------------------------------------------
    # Load and clean data
    # --------------------------------------------------------

    df = load_clickstream()

    print(f"Original dataset shape: {df.shape}")

    df = clean_clickstream(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X, y, preprocessor = prepare_features(df)

    print(f"Feature records: {X.shape}")
    print(f"Target records: {y.shape}")

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # --------------------------------------------------------
    # Create baseline Logistic Regression pipeline
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Prepare project directories
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Configure MLflow SQLite backend
    # --------------------------------------------------------

    mlflow_tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    mlflow.set_tracking_uri(
        mlflow_tracking_uri
    )

    mlflow.set_experiment(
        "OmniRetail_Checkpoint_2_Baseline"
    )

    # --------------------------------------------------------
    # Train and track model
    # --------------------------------------------------------

    print()
    print("Starting baseline model training...")

    with mlflow.start_run(
        run_name="LogisticRegression_Baseline"
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        # ----------------------------------------------------
        # Metrics dictionary
        # ----------------------------------------------------

        metrics = {
            "model": "Logistic Regression",
            "dataset": "aiml_ecommerce_clickstream.csv",
            "total_records": int(len(df)),
            "training_records": int(len(X_train)),
            "testing_records": int(len(X_test)),
            "feature_count": int(X.shape[1]),
            "target": "label_purchased",
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
        }

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            "Logistic Regression",
        )

        mlflow.log_param(
            "test_size",
            0.20,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "total_records",
            len(df),
        )

        mlflow.log_param(
            "feature_count",
            X.shape[1],
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        # ----------------------------------------------------
        # Save model locally
        # ----------------------------------------------------

        joblib.dump(
            model,
            MODEL_FILE,
        )

        # ----------------------------------------------------
        # Save metrics locally
        # ----------------------------------------------------

        with open(
            METRICS_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # Log model to MLflow
        #
        # MLflow 3.x uses skops for sklearn model validation.
        # numpy.dtype is explicitly trusted for this model.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="baseline_purchase_model",
            skops_trusted_types=[
                "numpy.dtype",
            ],
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print()
        print("=" * 55)
        print("        OmniRetail AI Baseline Model")
        print("=" * 55)
        print()

        print(
            f"Dataset records : {len(df)}"
        )

        print(
            f"Training records: {len(X_train)}"
        )

        print(
            f"Testing records : {len(X_test)}"
        )

        print(
            f"Feature count   : {X.shape[1]}"
        )

        print()

        print(
            "Evaluation Metrics"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print()

        print(
            "Classification Report"
        )

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        print(
            f"Model saved   : {MODEL_FILE}"
        )

        print(
            f"Metrics saved : {METRICS_FILE}"
        )

        print(
            f"MLflow DB     : {MLFLOW_DB}"
        )

        print()
        print("Baseline model completed successfully.")
        print("=" * 55)


if __name__ == "__main__":
    train_baseline()