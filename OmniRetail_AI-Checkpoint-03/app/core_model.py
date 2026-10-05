import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import MLFLOW_DB
from app.data_loader import load_clickstream
from app.preprocessing import clean_clickstream, prepare_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

MODEL_FILE = MODELS_DIR / "best_core_model.joblib"
RESULTS_FILE = DOCS_DIR / "experiment_results.json"

EXPERIMENT_NAME = "OmniRetail_Checkpoint_3_Core_Model"


EXPERIMENT_CONFIGS = [
    {
        "name": "RandomForest_Config_1",
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
    },
    {
        "name": "RandomForest_Config_2",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
    },
    {
        "name": "RandomForest_Config_3",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
    },
]


def train_core_model():
    """Train and compare multiple Random Forest configurations."""

    print("Loading clickstream dataset...")

    df = load_clickstream()

    print(f"Original dataset shape: {df.shape}")

    df = clean_clickstream(df)

    print(f"Cleaned dataset shape: {df.shape}")

    X, y, preprocessor = prepare_features(df)

    print(f"Feature records: {X.shape}")
    print(f"Target records: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow_tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    mlflow.set_tracking_uri(
        mlflow_tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    results = []

    print()
    print("=" * 65)
    print("      OmniRetail AI Core Model Experiments")
    print("=" * 65)

    for config in EXPERIMENT_CONFIGS:

        print()
        print(f"Running: {config['name']}")

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=config["n_estimators"],
                        max_depth=config["max_depth"],
                        min_samples_split=config["min_samples_split"],
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        with mlflow.start_run(
            run_name=config["name"]
        ):

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

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

            mlflow.log_param(
                "model",
                "Random Forest",
            )

            mlflow.log_param(
                "n_estimators",
                config["n_estimators"],
            )

            if config["max_depth"] is not None:
                mlflow.log_param(
                    "max_depth",
                    config["max_depth"],
                )
            else:
                mlflow.log_param(
                    "max_depth",
                    "None",
                )

            mlflow.log_param(
                "min_samples_split",
                config["min_samples_split"],
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            mlflow.log_param(
                "test_size",
                0.20,
            )

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

            mlflow.sklearn.log_model(
                model,
                name="core_random_forest_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            experiment_result = {
                "configuration": config["name"],
                "model": "Random Forest",
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "min_samples_split": config["min_samples_split"],
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4),
            }

            results.append(
                experiment_result
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

    best_result = max(
        results,
        key=lambda result: result["f1_score"],
    )

    best_config_name = best_result[
        "configuration"
    ]

    best_config = next(
        config
        for config in EXPERIMENT_CONFIGS
        if config["name"] == best_config_name
    )

    print()
    print("=" * 65)
    print("                 BEST CONFIGURATION")
    print("=" * 65)

    print(
        f"Configuration : {best_config_name}"
    )

    print(
        f"Accuracy      : {best_result['accuracy']:.4f}"
    )

    print(
        f"Precision     : {best_result['precision']:.4f}"
    )

    print(
        f"Recall        : {best_result['recall']:.4f}"
    )

    print(
        f"F1 Score      : {best_result['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC       : {best_result['roc_auc']:.4f}"
    )

    best_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=best_config["n_estimators"],
                    max_depth=best_config["max_depth"],
                    min_samples_split=best_config["min_samples_split"],
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    best_model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        best_model,
        MODEL_FILE,
    )

    output = {
        "experiment": EXPERIMENT_NAME,
        "dataset": "aiml_ecommerce_clickstream.csv",
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "target": "label_purchased",
        "selection_metric": "f1_score",
        "experiments_completed": len(results),
        "best_configuration": best_result,
        "all_results": results,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    print()
    print(
        f"Best model saved   : {MODEL_FILE}"
    )

    print(
        f"Results saved      : {RESULTS_FILE}"
    )

    print(
        f"MLflow database    : {MLFLOW_DB}"
    )

    print()
    print(
        "Core model experimentation completed successfully."
    )


if __name__ == "__main__":
    train_core_model()