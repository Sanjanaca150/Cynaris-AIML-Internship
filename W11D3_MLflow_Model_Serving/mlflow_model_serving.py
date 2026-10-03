import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import pandas as pd


# ============================================================
# W11D3 - MLflow Model Serving & REST API
# ============================================================

EXPERIMENT_NAME = "W11D3_MLflow_Model_Serving"
MODEL_NAME = "W11D3_Iris_RandomForest"

# Use SQLite as the MLflow backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or use the MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 1. Load Dataset
# ============================================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="target"
)


# ============================================================
# 2. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("=" * 70)
print("W11D3 - MLflow Model Serving & REST API")
print("=" * 70)

print("\nDataset:")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Features         : {X.shape[1]}")


# ============================================================
# 3. Five Experiments
# ============================================================

experiments = [
    {
        "n_estimators": 50,
        "max_depth": 3,
        "min_samples_split": 2
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    },
    {
        "n_estimators": 150,
        "max_depth": 7,
        "min_samples_split": 2
    },
    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2
    },
    {
        "n_estimators": 250,
        "max_depth": None,
        "min_samples_split": 2
    }
]


best_run_id = None
best_accuracy = -1
best_params = None


# ============================================================
# 4. Run Experiments
# ============================================================

for experiment_number, params in enumerate(experiments, start=1):

    print("\n" + "-" * 70)
    print(f"EXPERIMENT {experiment_number}")
    print("-" * 70)

    with mlflow.start_run(
        run_name=f"experiment_{experiment_number}"
    ) as run:

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(X_test)

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "experiment_number",
            experiment_number
        )

        mlflow.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"]
        )

        mlflow.log_param(
            "min_samples_split",
            params["min_samples_split"]
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # ----------------------------------------------------
        # Create model signature
        # ----------------------------------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # ----------------------------------------------------
        # Log sklearn model
        #
        # Required because current MLflow/skops validation
        # identifies sklearn.tree._tree.Tree as an untrusted type.
        # This model was created locally in this script.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Print experiment results
        # ----------------------------------------------------

        print(f"Run ID    : {run.info.run_id}")
        print(f"Parameters: {params}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        # ----------------------------------------------------
        # Identify best model
        # ----------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_run_id = run.info.run_id
            best_params = params.copy()


# ============================================================
# 5. Display Best Model
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Run ID : {best_run_id}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {best_params}")


# ============================================================
# 6. Register Best Model
# ============================================================

best_model_uri = f"runs:/{best_run_id}/model"

print("\nRegistering best model...")

registered_model = mlflow.register_model(
    model_uri=best_model_uri,
    name=MODEL_NAME
)

print(f"Registered Model : {MODEL_NAME}")
print(f"Model Version    : {registered_model.version}")


# ============================================================
# 7. Load Registered Model
# ============================================================

model_version_uri = (
    f"models:/{MODEL_NAME}/{registered_model.version}"
)

print("\nLoading registered model...")

loaded_model = mlflow.sklearn.load_model(
    model_version_uri
)

print("Registered model loaded successfully.")


# ============================================================
# 8. Test Loaded Registered Model
# ============================================================

loaded_predictions = loaded_model.predict(X_test)

loaded_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)

print(
    f"Loaded Model Accuracy: {loaded_accuracy:.4f}"
)


# ============================================================
# 9. Sample Predictions
# ============================================================

print("\nSample Predictions:")

sample_data = X_test.head(5)

sample_predictions = loaded_model.predict(
    sample_data
)

for index, prediction in enumerate(
    sample_predictions,
    start=1
):

    print(
        f"Sample {index}: "
        f"{iris.target_names[prediction]}"
    )


# ============================================================
# 10. Model Serving Information
# ============================================================

print("\n" + "=" * 70)
print("MODEL SERVING COMMAND")
print("=" * 70)

print(
    f'mlflow models serve '
    f'-m "models:/{MODEL_NAME}/{registered_model.version}" '
    f'-p 5001'
)

print("\nREST endpoint:")
print("POST http://127.0.0.1:5001/invocations")


# ============================================================
# 11. Completion
# ============================================================

print("\n" + "=" * 70)
print("W11D3 COMPLETED")
print("=" * 70)

print("\nCompleted:")
print("[1] MLflow experiment created")
print("[2] Parameters logged")
print("[3] Metrics logged")
print("[4] Model artifacts logged")
print("[5] Five experiments completed")
print("[6] Best model identified")
print("[7] Best model registered")
print("[8] Registered model loaded")
print("[9] Loaded model tested")
print("[10] Model serving command generated")