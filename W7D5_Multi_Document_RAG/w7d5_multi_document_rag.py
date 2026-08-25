"""
W7D5 — Week 7 Project: Multi-document RAG System

Technology Stack:
- LlamaIndex
- Ollama
- nomic-embed-text
- llama3.2:3b

Project Objective:
1. Load multiple text documents.
2. Split documents into searchable chunks.
3. Create embeddings using Ollama.
4. Build a vector index using LlamaIndex.
5. Retrieve relevant chunks for questions.
6. Use a local Ollama LLM to generate answers.
7. Display source documents.
8. Test the system with 10 questions.
9. Persist the index so it can be reused.
"""

import os
import time
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"

# Persistent LlamaIndex storage
INDEX_DIR = BASE_DIR / "storage"

OLLAMA_URL = "http://localhost:11434"

LLM_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "nomic-embed-text:latest"

# Retrieve only the most relevant chunks.
TOP_K = 2

# Larger chunks = fewer embedding requests.
CHUNK_SIZE = 1024

CHUNK_OVERLAP = 100

# Give local Ollama enough time to answer.
LLM_TIMEOUT = 600.0

# Keep the generated answers reasonably short.
LLM_OUTPUT_TOKENS = 256


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    """Print a formatted section header."""

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# STEP 1 — VERIFY DOCUMENTS
# ============================================================

def verify_documents():
    """Verify that the documents directory exists."""

    print_header("STEP 1 — VERIFYING DOCUMENTS")

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENTS_DIR}"
        )

    files = [
        file
        for file in DOCUMENTS_DIR.rglob("*")
        if file.is_file()
    ]

    if not files:

        raise FileNotFoundError(
            "No documents were found in the documents folder."
        )

    print(
        f"Document directory: {DOCUMENTS_DIR}"
    )

    print(
        f"Number of documents: {len(files)}"
    )

    for file in files:

        print(
            f"  ✓ {file.name}"
        )

    return files


# ============================================================
# STEP 2 — CONFIGURE OLLAMA
# ============================================================

def configure_models():
    """
    Configure the local Ollama embedding model and LLM.

    No OpenAI API key is required.
    """

    print_header(
        "STEP 2 — CONFIGURING LOCAL OLLAMA"
    )

    print(
        f"LLM model: {LLM_MODEL}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Ollama URL: {OLLAMA_URL}"
    )

    # --------------------------------------------------------
    # Local LLM
    # --------------------------------------------------------

    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=LLM_TIMEOUT,
        temperature=0.1,
        context_window=4096,
        keep_alive="10m",
    )

    # --------------------------------------------------------
    # Local embedding model
    # --------------------------------------------------------

    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    # --------------------------------------------------------
    # LlamaIndex global configuration
    # --------------------------------------------------------

    Settings.llm = llm

    Settings.embed_model = embed_model

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    Settings.num_output = LLM_OUTPUT_TOKENS

    print()
    print(
        "✓ Ollama LLM configured"
    )

    print(
        "✓ Ollama embedding model configured"
    )

    print(
        "✓ SentenceSplitter configured"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print(
        f"LLM timeout: {LLM_TIMEOUT} seconds"
    )

    return llm, embed_model


# ============================================================
# STEP 3 — LOAD DOCUMENTS
# ============================================================

def load_documents():
    """Load all documents using LlamaIndex."""

    print_header(
        "STEP 3 — LOADING DOCUMENTS"
    )

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        recursive=True,
    ).load_data()

    print(
        f"Loaded document objects: "
        f"{len(documents)}"
    )

    if not documents:

        raise ValueError(
            "No documents were loaded."
        )

    return documents


# ============================================================
# STEP 4 — BUILD OR LOAD VECTOR INDEX
# ============================================================

def build_or_load_index(documents):
    """
    Build the vector index on the first run.

    After successful indexing, persist it to disk so later
    executions do not need to regenerate embeddings.
    """

    print_header(
        "STEP 4 — VECTOR INDEX"
    )

    # --------------------------------------------------------
    # Try loading an existing persistent index.
    # --------------------------------------------------------

    if INDEX_DIR.exists():

        try:

            print(
                "Existing persistent index found."
            )

            print(
                f"Loading index from: {INDEX_DIR}"
            )

            storage_context = (
                StorageContext.from_defaults(
                    persist_dir=str(INDEX_DIR)
                )
            )

            index = load_index_from_storage(
                storage_context
            )

            print()
            print(
                "✓ Existing vector index loaded."
            )

            print(
                "No new embedding generation required."
            )

            return index

        except Exception as error:

            print()
            print(
                "WARNING: Existing index could not be loaded."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "A new index will be created."
            )

    # --------------------------------------------------------
    # Build a new index.
    # --------------------------------------------------------

    print(
        "No usable persistent index found."
    )

    print(
        "Creating a new vector index..."
    )

    print()
    print(
        "Embedding documents using Ollama."
    )

    print(
        "This may take time on the first run."
    )

    start_time = time.perf_counter()

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Persist the index.
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index.storage_context.persist(
        persist_dir=str(INDEX_DIR)
    )

    print()
    print(
        "✓ Vector index created."
    )

    print(
        f"Indexing time: {elapsed:.2f} seconds"
    )

    print(
        f"Index persisted to: {INDEX_DIR}"
    )

    return index


# ============================================================
# STEP 5 — CREATE QUERY ENGINE
# ============================================================

def create_query_engine(index):
    """
    Create the LlamaIndex QueryEngine.

    simple_summarize is used to avoid the long refinement
    process that caused the previous ReadTimeout.
    """

    print_header(
        "STEP 5 — CREATING QUERY ENGINE"
    )

    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=TOP_K,

        # Avoid lengthy refine loops with local LLM.
        response_mode="simple_summarize",
    )

    print(
        "✓ Query engine created"
    )

    print(
        f"Top-K retrieved chunks: {TOP_K}"
    )

    print(
        "Response mode: simple_summarize"
    )

    print(
        f"LLM: {LLM_MODEL}"
    )

    return query_engine


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(response):
    """Display retrieved source documents."""

    source_nodes = getattr(
        response,
        "source_nodes",
        []
    )

    if not source_nodes:

        print(
            "  - No source metadata available."
        )

        return

    displayed_sources = set()

    for number, node in enumerate(
        source_nodes,
        start=1
    ):

        metadata = node.node.metadata

        source = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("file_path")
            or "Unknown source"
        )

        score = node.score

        if source in displayed_sources:

            continue

        displayed_sources.add(source)

        if score is not None:

            print(
                f"  - {source} "
                f"(score={score:.4f})"
            )

        else:

            print(
                f"  - {source}"
            )


# ============================================================
# STEP 6 — RUN SINGLE QUERY
# ============================================================

def run_query(
    query_engine,
    question
):
    """
    Run one RAG query.

    Displays:
    - question
    - generated answer
    - retrieved sources
    - similarity scores
    - query latency
    """

    print_header(
        "RAG QUERY"
    )

    print(
        f"Question:\n{question}"
    )

    start_time = time.perf_counter()

    try:

        response = query_engine.query(
            question
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "ANSWER"
        )

        print(
            "-" * 75
        )

        print(
            str(response)
        )

        print()
        print(
            "SOURCES"
        )

        print(
            "-" * 75
        )

        display_sources(
            response
        )

        print()
        print(
            f"Query time: "
            f"{elapsed:.2f} seconds"
        )

        return {
            "success": True,
            "time": elapsed,
            "response": response,
        }

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "QUERY ERROR"
        )

        print(
            "-" * 75
        )

        print(
            str(error)
        )

        print()
        print(
            f"Time before failure: "
            f"{elapsed:.2f} seconds"
        )

        return {
            "success": False,
            "time": elapsed,
            "response": None,
        }


# ============================================================
# STEP 7 — RUN 10 TEST QUESTIONS
# ============================================================

def run_test_questions(
    query_engine
):
    """Run ten questions across the document collection."""

    print_header(
        "STEP 6 — RUNNING 10 TEST QUESTIONS"
    )

    questions = [

        "What is artificial intelligence "
        "and what are its major applications?",

        "What are the main characteristics "
        "of cloud computing?",

        "What are the important principles "
        "of software design?",

        "What are the major stages involved "
        "in software engineering?",

        "How does software architecture "
        "influence software design?",

        "What are the advantages of using "
        "cloud computing?",

        "What are common techniques used "
        "in artificial intelligence?",

        "What factors should be considered "
        "when designing software systems?",

        "How can software engineering practices "
        "improve software quality?",

        "How are artificial intelligence, "
        "cloud computing, and software engineering related?",
    ]

    results = []

    for number, question in enumerate(
        questions,
        start=1
    ):

        print()
        print(
            "#" * 75
        )

        print(
            f"QUESTION {number}/10"
        )

        print(
            "#" * 75
        )

        result = run_query(
            query_engine,
            question
        )

        results.append({
            "question": question,
            "success": result["success"],
            "time": result["time"],
        })

    return results


# ============================================================
# STEP 8 — DISPLAY TEST SUMMARY
# ============================================================

def display_summary(
    results
):
    """Display final test summary."""

    print_header(
        "FINAL TEST SUMMARY"
    )

    successful = sum(
        result["success"]
        for result in results
    )

    total = len(results)

    failed = total - successful

    print(
        f"Total questions: {total}"
    )

    print(
        f"Successful queries: {successful}"
    )

    print(
        f"Failed queries: {failed}"
    )

    successful_times = [
        result["time"]
        for result in results
        if result["success"]
    ]

    if successful_times:

        average_time = (
            sum(successful_times)
            / len(successful_times)
        )

        print(
            f"Average successful query time: "
            f"{average_time:.2f} seconds"
        )

        print()
        print(
            "Individual query times:"
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            if result["success"]:

                print(
                    f"  Query {number}: "
                    f"{result['time']:.2f} seconds"
                )

    print()

    if successful == total:

        print(
            "✓ ALL 10 TEST QUESTIONS PASSED"
        )

    elif successful > 0:

        print(
            "⚠ SOME QUESTIONS PASSED"
        )

    else:

        print(
            "✗ ALL QUESTIONS FAILED"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print_header(
        "W7D5 — MULTI-DOCUMENT RAG SYSTEM"
    )

    print(
        "Technology: LlamaIndex + Ollama"
    )

    print(
        f"LLM: {LLM_MODEL}"
    )

    print(
        f"Embedding: {EMBEDDING_MODEL}"
    )

    print(
        f"Documents: {DOCUMENTS_DIR}"
    )

    print(
        f"Persistent index: {INDEX_DIR}"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    verify_documents()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    configure_models()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    documents = load_documents()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    index = build_or_load_index(
        documents
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    query_engine = create_query_engine(
        index
    )

    # --------------------------------------------------------
    # Demonstration query
    # --------------------------------------------------------

    demo_result = run_query(
        query_engine,
        "What is artificial intelligence?"
    )

    # --------------------------------------------------------
    # Ten-question evaluation
    # --------------------------------------------------------

    results = run_test_questions(
        query_engine
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    display_summary(
        results
    )

    print_header(
        "W7D5 MULTI-DOCUMENT RAG COMPLETED"
    )

    print(
        "✓ Multiple documents loaded"
    )

    print(
        "✓ Ollama embeddings configured"
    )

    print(
        "✓ Local Ollama LLM configured"
    )

    print(
        "✓ LlamaIndex vector index created/loaded"
    )

    print(
        "✓ QueryEngine created"
    )

    print(
        "✓ Sources displayed"
    )

    print(
        "✓ 10 test questions executed"
    )

    print(
        "✓ Index persisted for future runs"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()