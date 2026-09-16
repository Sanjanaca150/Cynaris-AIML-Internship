import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# W11D2 - MLflow Model Registry & Versioning
# ============================================================

EXPERIMENT_NAME = "W11D2_Iris_Model_Registry"
REGISTERED_MODEL_NAME = "Iris_RandomForest_Model"

# Local SQLite database for MLflow tracking and registry
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 1. LOAD IRIS DATASET
# ============================================================

print("=" * 70)
print("W11D2 - MLFLOW MODEL REGISTRY & VERSIONING")
print("=" * 70)

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset: Iris")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 2. DEFINE FIVE EXPERIMENT CONFIGURATIONS
# ============================================================

experiments = [
    {
        "n_estimators": 50,
        "max_depth": 2,
        "min_samples_split": 2
    },
    {
        "n_estimators": 100,
        "max_depth": 3,
        "min_samples_split": 2
    },
    {
        "n_estimators": 150,
        "max_depth": 4,
        "min_samples_split": 2
    },
    {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 2
    },
    {
        "n_estimators": 250,
        "max_depth": None,
        "min_samples_split": 2
    }
]


# ============================================================
# 3. RUN FIVE MLFLOW EXPERIMENTS
# ============================================================

run_results = []

print("\n" + "=" * 70)
print("RUNNING 5 MLFLOW EXPERIMENTS")
print("=" * 70)

for experiment_number, params in enumerate(experiments, start=1):

    with mlflow.start_run(
        run_name=f"Experiment_{experiment_number}"
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
        # Calculate accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(params)

        # ----------------------------------------------------
        # Log metric
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
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
        # MLflow 3.16 + skops requires explicit trust for
        # sklearn.tree._tree.Tree when logging RandomForest.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train[:2],
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        run_results.append(
            {
                "run_id": run.info.run_id,
                "experiment": experiment_number,
                "accuracy": accuracy,
                "params": params
            }
        )

        print(f"\nExperiment {experiment_number}")
        print(f"Run ID     : {run.info.run_id}")
        print(f"Parameters : {params}")
        print(f"Accuracy   : {accuracy:.4f}")


# ============================================================
# 4. COMPARE EXPERIMENTS AND FIND BEST RUN
# ============================================================

best_run = max(
    run_results,
    key=lambda result: result["accuracy"]
)

print("\n" + "=" * 70)
print("EXPERIMENT COMPARISON")
print("=" * 70)

for result in run_results:
    print(
        f"Experiment {result['experiment']} "
        f"| Accuracy: {result['accuracy']:.4f} "
        f"| Parameters: {result['params']}"
    )


print("\n" + "=" * 70)
print("BEST EXPERIMENT")
print("=" * 70)

print(f"Experiment : {best_run['experiment']}")
print(f"Run ID     : {best_run['run_id']}")
print(f"Accuracy   : {best_run['accuracy']:.4f}")
print(f"Parameters : {best_run['params']}")


# ============================================================
# 5. REGISTER THE BEST MODEL
# ============================================================

best_model_uri = (
    f"runs:/{best_run['run_id']}/model"
)

print("\n" + "=" * 70)
print("REGISTERING BEST MODEL")
print("=" * 70)

registered_model = mlflow.register_model(
    model_uri=best_model_uri,
    name=REGISTERED_MODEL_NAME
)

print(f"Registered Model : {REGISTERED_MODEL_NAME}")
print(f"Model Version    : {registered_model.version}")


# ============================================================
# 6. LOAD REGISTERED MODEL
# ============================================================

model_version_uri = (
    f"models:/{REGISTERED_MODEL_NAME}/{registered_model.version}"
)

print("\n" + "=" * 70)
print("LOADING REGISTERED MODEL")
print("=" * 70)

loaded_model = mlflow.sklearn.load_model(
    model_version_uri
)

print(f"Model URI   : {model_version_uri}")
print(f"Model Type  : {type(loaded_model).__name__}")


# ============================================================
# 7. TEST REGISTERED MODEL
# ============================================================

test_samples = X_test[:5]

loaded_predictions = loaded_model.predict(
    test_samples
)

print("\n" + "=" * 70)
print("REGISTERED MODEL PREDICTION TEST")
print("=" * 70)

print("Test Predictions:")
print(loaded_predictions)

print("\nActual Values:")
print(y_test[:5])


# ============================================================
# 8. EVALUATE REGISTERED MODEL
# ============================================================

all_predictions = loaded_model.predict(
    X_test
)

registered_model_accuracy = accuracy_score(
    y_test,
    all_predictions
)

print("\n" + "=" * 70)
print("REGISTERED MODEL EVALUATION")
print("=" * 70)

print(
    f"Registered Model Accuracy: "
    f"{registered_model_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        all_predictions,
        target_names=iris.target_names
    )
)


# ============================================================
# 9. FINAL INFORMATION
# ============================================================

print("=" * 70)
print("W11D2 TASK COMPLETED")
print("=" * 70)

print(f"MLflow Tracking URI : {TRACKING_URI}")
print(f"Experiment           : {EXPERIMENT_NAME}")
print(f"Registered Model     : {REGISTERED_MODEL_NAME}")
print(f"Registered Version   : {registered_model.version}")
print(f"Best Accuracy        : {best_run['accuracy']:.4f}")

print("\n" + "=" * 70)
print("MLFLOW UI COMMAND")
print("=" * 70)

print(
    "mlflow server "
    "--backend-store-uri sqlite:///mlflow.db "
    "--host 127.0.0.1 "
    "--port 5000"
)

print("\n" + "=" * 70)
print("MODEL SERVING COMMAND")
print("=" * 70)

print(
    f'mlflow models serve '
    f'--model-uri "{model_version_uri}" '
    f'--port 5001 '
    f'--env-manager local'
)

print("\n" + "=" * 70)
print("MODEL REGISTRY DETAILS")
print("=" * 70)

print(f"Model Name    : {REGISTERED_MODEL_NAME}")
print(f"Model Version : {registered_model.version}")
print(f"Model URI     : {model_version_uri}")