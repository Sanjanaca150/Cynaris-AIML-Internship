"""
W11D5 - RAG Evaluation with Ragas and MLflow

Uses:
- ChromaDB for retrieval
- Ollama for embeddings
- Ollama for LLM generation
- Ragas for evaluation
- MLflow for experiment tracking
"""

import time
from pathlib import Path

import mlflow
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"
MLFLOW_DB = BASE_DIR / "mlflow.db"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"

COLLECTION_NAME = "w11d5_healthcare"

RETRIEVAL_K = 3

MLFLOW_EXPERIMENT = "W11D5_Tracked_Evaluated_RAG"


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# OLLAMA COMPONENTS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# CHROMADB
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)


# ============================================================
# HEALTHCARE EVALUATION DATASET
# ============================================================

EVALUATION_QUESTIONS = [
    {
        "question": "What are the common symptoms of diabetes?",
        "ground_truth": (
            "Common symptoms of diabetes include increased thirst, "
            "frequent urination, increased hunger, fatigue, and "
            "blurred vision."
        ),
    },
    {
        "question": "What are the common symptoms of hypertension?",
        "ground_truth": (
            "Hypertension means elevated blood pressure. It can be "
            "associated with risk factors such as obesity, high salt "
            "intake, physical inactivity, and family history."
        ),
    },
    {
        "question": "What are the symptoms of asthma?",
        "ground_truth": (
            "Common symptoms of asthma include wheezing, coughing, "
            "chest tightness, and shortness of breath."
        ),
    },
    {
        "question": "What is a common cause of anemia?",
        "ground_truth": (
            "Iron deficiency is a common cause of anemia. Symptoms "
            "can include fatigue, weakness, and pale skin."
        ),
    },
    {
        "question": "Is the common cold caused by bacteria?",
        "ground_truth": (
            "The common cold is a viral infection, and antibiotics "
            "are not effective against viruses."
        ),
    },
    {
        "question": "What is influenza?",
        "ground_truth": (
            "Influenza, commonly called the flu, is a contagious "
            "respiratory illness caused by influenza viruses."
        ),
    },
    {
        "question": "What are some risk factors for heart disease?",
        "ground_truth": (
            "Risk factors for heart disease can include high blood "
            "pressure, high cholesterol, smoking, obesity, physical "
            "inactivity, and diabetes."
        ),
    },
    {
        "question": "What is the purpose of preventive healthcare?",
        "ground_truth": (
            "Preventive healthcare focuses on preventing disease "
            "and detecting health problems early."
        ),
    },
]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):
    """Generate an answer using only retrieved healthcare context."""

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a healthcare information assistant.

Answer the question using ONLY the provided context.

Do not invent information.

If the context does not contain the answer, say:
"I don't have enough information in the provided context."

Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# BUILD EVALUATION DATA
# ============================================================

def build_evaluation_dataset():
    """Retrieve context and generate answers for evaluation."""

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print()
    print("=" * 60)
    print("W11D5 RAGAS EVALUATION DATA")
    print("=" * 60)

    for item in EVALUATION_QUESTIONS:

        question = item["question"]

        documents = retriever.invoke(
            question
        )

        answer = generate_answer(
            question,
            documents
        )

        retrieved_contexts = [
            document.page_content
            for document in documents
        ]

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(
            item["ground_truth"]
        )

        print()
        print("-" * 60)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(
            f"Retrieved documents: "
            f"{len(documents)}"
        )

    return Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        }
    )


# ============================================================
# RUN RAGAS
# ============================================================

def run_evaluation():
    """Run Ragas evaluation and log results to MLflow."""

    start_time = time.perf_counter()

    dataset = build_evaluation_dataset()

    print()
    print("=" * 60)
    print("STARTING RAGAS EVALUATION")
    print("=" * 60)
    print()
    print("Evaluation LLM: Ollama")
    print(f"LLM model: {LLM_MODEL}")
    print()

    # Configure Ragas to use the local Ollama model.
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(
        llm
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        embeddings
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for metric in metrics:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    execution_time = (
        time.perf_counter() - start_time
    )

    # ========================================================
    # MLflow RUN
    # ========================================================

    with mlflow.start_run() as run:

        mlflow.log_param(
            "embedding_model",
            EMBEDDING_MODEL
        )

        mlflow.log_param(
            "llm_model",
            LLM_MODEL
        )

        mlflow.log_param(
            "retrieval_k",
            RETRIEVAL_K
        )

        mlflow.log_param(
            "vector_database",
            "ChromaDB"
        )

        mlflow.log_param(
            "collection_name",
            COLLECTION_NAME
        )

        mlflow.log_param(
            "evaluation_framework",
            "Ragas"
        )

        mlflow.log_param(
            "evaluation_llm",
            "Ollama"
        )

        mlflow.log_metric(
            "execution_time_seconds",
            execution_time
        )

        # ----------------------------------------------------
        # Extract Ragas scores
        # ----------------------------------------------------

        result_dict = result.to_pandas().mean(
            numeric_only=True
        ).to_dict()

        for metric_name, score in result_dict.items():

            if score is not None:

                try:
                    mlflow.log_metric(
                        metric_name,
                        float(score)
                    )
                except (TypeError, ValueError):
                    pass

        # ----------------------------------------------------
        # Save evaluation result
        # ----------------------------------------------------

        evaluation_text = str(
            result
        )

        mlflow.log_text(
            evaluation_text,
            "ragas_evaluation.txt"
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("W11D5 RAGAS EVALUATION RESULTS")
        print("=" * 60)

        print()

        for metric_name, score in result_dict.items():

            print(
                f"{metric_name}: "
                f"{float(score):.4f}"
            )

        print()
        print(
            f"Execution time: "
            f"{execution_time:.2f}s"
        )

        print()
        print(
            f"MLflow experiment: "
            f"{MLFLOW_EXPERIMENT}"
        )

        print(
            f"MLflow run ID: "
            f"{run.info.run_id}"
        )

        print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_evaluation()