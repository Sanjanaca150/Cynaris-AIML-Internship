"""
W11D5 - Tracked and Evaluated RAG Pipeline

Components:
- LangGraph for RAG workflow
- ChromaDB for vector retrieval
- Ollama for embeddings and generation
- MLflow for experiment tracking
"""

import time
from pathlib import Path
from typing import TypedDict

import mlflow
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import END, START, StateGraph


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
# EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# CHROMADB VECTOR STORE
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
)


# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)


# ============================================================
# OLLAMA LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict):
    question: str
    context: list[Document]
    answer: str
    execution_time: float


# ============================================================
# RETRIEVE NODE
# ============================================================

def retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant healthcare documents from ChromaDB."""

    documents = retriever.invoke(
        state["question"]
    )

    return {
        **state,
        "context": documents,
    }


# ============================================================
# GENERATE NODE
# ============================================================

def generate(state: RAGState) -> RAGState:
    """Generate an answer using only retrieved context."""

    context_text = "\n\n".join(
        document.page_content
        for document in state["context"]
    )

    prompt = f"""
You are a healthcare information assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Do not invent medical facts.

Keep the answer clear and concise.

Context:
{context_text}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    return {
        **state,
        "answer": answer,
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(RAGState)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)

rag_graph = graph_builder.compile()


# ============================================================
# RUN RAG PIPELINE
# ============================================================

def run_rag(question: str):
    """Run the complete RAG pipeline and track it with MLflow."""

    start_time = time.perf_counter()

    with mlflow.start_run() as run:

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

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
            "framework",
            "LangGraph"
        )

        mlflow.log_param(
            "question",
            question
        )

        # ----------------------------------------------------
        # Execute graph
        # ----------------------------------------------------

        result = rag_graph.invoke(
            {
                "question": question,
                "context": [],
                "answer": "",
                "execution_time": 0.0,
            }
        )

        execution_time = (
            time.perf_counter() - start_time
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "execution_time_seconds",
            execution_time
        )

        mlflow.log_metric(
            "retrieved_documents",
            len(result["context"])
        )

        # ----------------------------------------------------
        # Log text artifacts
        # ----------------------------------------------------

        mlflow.log_text(
            question,
            "question.txt"
        )

        context_text = "\n\n".join(
            document.page_content
            for document in result["context"]
        )

        mlflow.log_text(
            context_text,
            "retrieved_context.txt"
        )

        mlflow.log_text(
            result["answer"],
            "answer.txt"
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("W11D5 TRACKED RAG PIPELINE")
        print("=" * 70)

        print()
        print("Question:")
        print(question)

        print()
        print("Retrieved Context:")
        print("-" * 70)

        if result["context"]:

            for index, document in enumerate(
                result["context"],
                start=1
            ):
                print(
                    f"\n[{index}] "
                    f"{document.page_content}"
                )

        else:
            print(
                "No relevant documents were retrieved."
            )

        print()
        print("Answer:")
        print("-" * 70)
        print(result["answer"])

        print()
        print(
            f"Execution time: "
            f"{execution_time:.2f}s"
        )

        print()
        print(
            f"Retrieved documents: "
            f"{len(result['context'])}"
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

        print("=" * 70)

        return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("W11D5 - TRACKED AND EVALUATED RAG")
    print("=" * 70)

    print()
    question = input(
        "Enter your healthcare question: "
    ).strip()

    if not question:
        print(
            "Question cannot be empty."
        )

    else:
        run_rag(question)