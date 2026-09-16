import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------
# MLflow Experiment Tracking — W11D1
# ---------------------------------------------------------

EXPERIMENT_NAME = "W11D1_MLflow_Experiment_Tracking"
MODEL_NAME = "W11D1_RandomForest_Best_Model"

# Set MLflow tracking location
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or get experiment
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Five different hyperparameter configurations
experiments = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 150, "max_depth": 7, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
]

print("=" * 60)
print("W11D1 - MLFLOW EXPERIMENT TRACKING")
print("=" * 60)

best_accuracy = -1
best_run_id = None
best_params = None

for i, params in enumerate(experiments, start=1):

    print(f"\nRunning Experiment {i}")
    print("-" * 40)
    print(f"Parameters: {params}")

    with mlflow.start_run(run_name=f"experiment_{i}") as run:

        # Create model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("min_samples_split", params["min_samples_split"])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test[:2]
        )

        # Tags
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("experiment_number", i)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Run ID   : {run.info.run_id}")

        # Track best run
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run.info.run_id
            best_params = params


# ---------------------------------------------------------
# Register Best Model
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BEST EXPERIMENT")
print("=" * 60)

print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Run ID  : {best_run_id}")
print(f"Best Params  : {best_params}")

# URI for best logged model
model_uri = f"runs:/{best_run_id}/model"

print("\nRegistering best model...")
print(f"Model URI: {model_uri}")

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\n" + "=" * 60)
print("MODEL REGISTRATION COMPLETE")
print("=" * 60)

print(f"Model Name   : {MODEL_NAME}")
print(f"Model Version: {registered_model.version}")
print(f"Best Run ID  : {best_run_id}")

print("\nMLflow experiment completed successfully.")