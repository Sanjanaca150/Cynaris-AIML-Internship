from pathlib import Path

import mlflow
from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import SemanticSimilarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Configure MLflow to use the project's local SQLite database.
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)


def run_evaluation():
    """Evaluate customer support responses using Ragas."""

    print("=" * 65)
    print("W10D5 RAGAS CUSTOMER SUPPORT EVALUATION")
    print("=" * 65)

    # --------------------------------------------------
    # LOCAL OLLAMA EMBEDDINGS
    # --------------------------------------------------

    evaluator_embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        evaluator_embeddings
    )

    # --------------------------------------------------
    # EVALUATION DATASET
    # --------------------------------------------------

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": (
                    "My order ORD-1001 has not arrived yet. "
                    "Can you help me?"
                ),
                "response": (
                    "Your request regarding order ORD-1001 "
                    "has been received. We can help review "
                    "the delivery issue."
                ),
                "reference": (
                    "The customer needs help with a delayed "
                    "order and its delivery status."
                ),
            },
            {
                "user_input": (
                    "I want a refund because I have been "
                    "waiting too long."
                ),
                "response": (
                    "Your refund request requires additional "
                    "assistance. It has been marked for human "
                    "support review."
                ),
                "reference": (
                    "The customer's refund request requires "
                    "human support review."
                ),
            },
        ]
    )

    # --------------------------------------------------
    # RAGAS METRIC
    # --------------------------------------------------

    metric = SemanticSimilarity()

    metric.embeddings = ragas_embeddings

    # --------------------------------------------------
    # MLFLOW EXPERIMENT
    # --------------------------------------------------

    mlflow.set_experiment(
        "W10D5_Stateful_Customer_Support_Evaluation"
    )

    with mlflow.start_run():

        # Run the actual Ragas evaluation.
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
        )

        print("\nRAGAS EVALUATION RESULT")
        print("-" * 65)
        print(result)

        # Ragas 0.3.6 returns a list containing one score
        # for each evaluation sample.
        scores = result["semantic_similarity"]

        # Calculate the average semantic similarity.
        average_score = sum(scores) / len(scores)

        print(
            "\nIndividual Scores:",
            scores
        )

        print(
            "Ragas Average Semantic Similarity:",
            round(average_score, 4)
        )

        # --------------------------------------------------
        # MLFLOW LOGGING
        # --------------------------------------------------

        mlflow.log_metric(
            "ragas_semantic_similarity",
            average_score,
        )

        mlflow.log_param(
            "evaluation_framework",
            "Ragas",
        )

        mlflow.log_param(
            "evaluation_metric",
            "SemanticSimilarity",
        )

        mlflow.log_param(
            "embedding_model",
            "nomic-embed-text:latest",
        )

        mlflow.log_param(
            "evaluation_samples",
            len(scores),
        )

        mlflow.set_tag(
            "project",
            "W10D5 Stateful Customer Support Agent",
        )

        mlflow.set_tag(
            "evaluation_status",
            "completed",
        )

        print(
            "\nEvaluation successfully logged to MLflow."
        )

    print("=" * 65)
    print("RAGAS EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()