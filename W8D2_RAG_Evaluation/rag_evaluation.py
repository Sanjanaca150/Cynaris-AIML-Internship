import sys
import types

# ============================================================
# RAGAS LEGACY VERTEX AI IMPORT HOTFIX
# ============================================================
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ModuleNotFoundError:
        class ChatVertexAI:
            pass

    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod


# ============================================================
# IMPORTS
# ============================================================
from datasets import Dataset

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.run_config import RunConfig


# ============================================================
# KNOWLEDGE BASE
# ============================================================
sample_text = """
The RAG (Retrieval-Augmented Generation) framework enhances Large Language Models by fetching relevant facts from an external knowledge base.

ChromaDB is an open-source vector database designed for storing and querying vector embeddings efficiently.

Ragas is a framework that helps evaluate Retrieval-Augmented Generation (RAG) pipelines without requiring extensive human annotations.

Faithfulness measures how grounded the answer is in the retrieved context.

Answer Relevancy measures how pertinent the generated answer is to the user's question.

Context Precision measures the signal-to-noise ratio of the retrieved context.

Context Recall measures if all relevant information needed to answer the question was retrieved.
"""

with open("knowledge_base.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# ============================================================
# BUILD AND EVALUATE RAG
# ============================================================
def build_and_evaluate_rag(
    chunk_size=100,
    chunk_overlap=10,
    k=1
):

    print(
        f"\nBuilding RAG pipeline: "
        f"chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}, "
        f"k={k}"
    )

    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(docs)

    print(f"Created {len(splits)} document chunks")

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"local_rag_{chunk_size}_{k}"
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # --------------------------------------------------------
    # RAG Generation LLM
    # --------------------------------------------------------
    local_llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        timeout=180
    )

    # --------------------------------------------------------
    # Ragas Evaluation LLM
    # --------------------------------------------------------
    evaluator_llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        timeout=180
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question "
        "concisely. If you don't know, say you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # --------------------------------------------------------
    # RAG Chain
    # --------------------------------------------------------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | local_llm
        | StrOutputParser()
    )

    # ========================================================
    # TEST QUESTIONS
    # ========================================================
    qa_data = [
        {
            "question": "What is RAG?",
            "reference":
            "RAG enhances Large Language Models by fetching "
            "relevant facts from an external knowledge base."
        },
        {
            "question": "What is ChromaDB?",
            "reference":
            "ChromaDB is an open-source vector database for "
            "storing and querying vector embeddings."
        },
        {
            "question": "What does Ragas do?",
            "reference":
            "Ragas evaluates RAG pipelines without requiring "
            "human annotations."
        },
        {
            "question": "What is faithfulness in Ragas?",
            "reference":
            "Faithfulness measures how grounded the answer is "
            "in the retrieved context."
        },
        {
            "question": "What does Answer Relevancy measure?",
            "reference":
            "Answer Relevancy measures how pertinent the answer "
            "is to the user's prompt."
        },
        {
            "question": "What is Context Precision?",
            "reference":
            "Context Precision measures the signal-to-noise "
            "ratio of the retrieved context."
        },
        {
            "question": "What is Context Recall?",
            "reference":
            "Context Recall measures if all relevant information "
            "was successfully retrieved."
        },
        {
            "question": "Why use ChromaDB for vector search?",
            "reference":
            "It allows efficient storage and querying of "
            "vector embeddings."
        },
        {
            "question": "How does RAG improve LLMs?",
            "reference":
            "By referencing external facts from a knowledge base."
        },
        {
            "question":
            "What framework helps evaluate RAG without human annotations?",
            "reference":
            "Ragas framework."
        }
    ]

    questions = [
        item["question"]
        for item in qa_data
    ]

    references = [
        item["reference"]
        for item in qa_data
    ]

    answers = []
    contexts = []

    # ========================================================
    # GENERATE ANSWERS
    # ========================================================
    print("\nGenerating answers and retrieving contexts...")

    for index, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"Processing question {index}/10..."
        )

        retrieved_docs = retriever.invoke(question)

        answer = rag_chain.invoke(question)

        answers.append(answer)

        contexts.append(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )

    # ========================================================
    # RAGAS DATASET
    # ========================================================
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": references
    }

    dataset = Dataset.from_dict(data)

    # ========================================================
    # RAGAS RUN CONFIGURATION
    # ========================================================
    run_config = RunConfig(
        timeout=180,
        max_retries=2,
        max_workers=1
    )

    # ========================================================
    # RAGAS EVALUATION
    # ========================================================
    print("\nStarting Ragas evaluation...")
    print("Evaluator model: qwen2.5:3b")
    print("RAG generation model: llama3.2:3b")
    print("Ragas max_workers=1")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False
    )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(
    baseline,
    optimized
):

    with open(
        "evaluation_output.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")
        f.write("W8D2 - RAG EVALUATION USING RAGAS\n")
        f.write("=" * 60 + "\n\n")

        f.write("BASELINE CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write("chunk_size = 100\n")
        f.write("chunk_overlap = 10\n")
        f.write("retrieval k = 1\n")
        f.write("RAG model = llama3.2:3b\n")
        f.write("Evaluator model = qwen2.5:3b\n\n")

        f.write("BASELINE RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(str(baseline))
        f.write("\n\n")

        f.write("OPTIMIZED CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write("chunk_size = 100\n")
        f.write("chunk_overlap = 10\n")
        f.write("retrieval k = 3\n")
        f.write("RAG model = llama3.2:3b\n")
        f.write("Evaluator model = qwen2.5:3b\n\n")

        f.write("OPTIMIZED RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(str(optimized))
        f.write("\n\n")

        f.write("BASELINE VS OPTIMIZED\n")
        f.write("-" * 60 + "\n")
        f.write("Baseline retrieval k = 1\n")
        f.write("Optimized retrieval k = 3\n")
        f.write("Changed variable = retrieval k\n")
        f.write("Chunk size = 100 for both configurations\n\n")

        f.write("RAG Evaluation completed successfully.\n")


# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("W8D2 - RAG EVALUATION USING RAGAS")
    print("=" * 60)

    # ========================================================
    # BASELINE
    # ========================================================
    print("\n--- Running Local Baseline ---")
    print("chunk_size=100, chunk_overlap=10, k=1")

    initial_metrics = build_and_evaluate_rag(
        chunk_size=100,
        chunk_overlap=10,
        k=1
    )

    print("\nBASELINE RESULTS:")
    print(initial_metrics)

    # ========================================================
    # OPTIMIZATION
    # ========================================================
    print("\n--- Running Local Optimization ---")
    print("chunk_size=100, chunk_overlap=10, k=3")
    print("Optimization: retrieval k changed from 1 to 3")

    optimized_metrics = build_and_evaluate_rag(
        chunk_size=100,
        chunk_overlap=10,
        k=3
    )

    print("\nOPTIMIZED RESULTS:")
    print(optimized_metrics)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================
    print("\n" + "=" * 60)
    print("BASELINE VS OPTIMIZED")
    print("=" * 60)

    print("\nBaseline:")
    print(initial_metrics)

    print("\nOptimized:")
    print(optimized_metrics)

    print("\nOptimization Summary:")
    print("Baseline retrieval k = 1")
    print("Optimized retrieval k = 3")
    print("Changed variable = retrieval k")
    print("Chunk size = 100 for both configurations")

    # ========================================================
    # SAVE OUTPUT
    # ========================================================
    save_results(
        initial_metrics,
        optimized_metrics
    )

    print("\nResults saved to evaluation_output.txt")
    print("\nRAG Evaluation completed successfully.")