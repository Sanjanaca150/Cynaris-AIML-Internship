"""
W7D3 — LlamaIndex: Document Indexing & Querying

Practical Tasks:
1. Index a folder of text files using LlamaIndex VectorStoreIndex
   with Ollama embeddings.
2. Build a QueryEngine, run 10 queries, and verify sources.
3. Connect LlamaIndex to ChromaDB, rerun 10 queries,
   and compare latency.

Stack:
- LlamaIndex
- Ollama
- nomic-embed-text
- llama3.2:3b
- ChromaDB
"""

import os
import time
import statistics

import chromadb

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "documents"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "llama3.2:3b"

TOP_K = 3

# Smaller context prevents Ollama from requesting huge
# amounts of RAM for the KV cache.
CONTEXT_WINDOW = 2048

# Limit very large extracted PDF text files.
MAX_CHARS_PER_FILE = 50000

CHUNK_SIZE = 512

CHUNK_OVERLAP = 50


# ============================================================
# 10 QUERIES
# ============================================================

QUERIES = [
    "What is software engineering?",
    "What is software design?",
    "What is artificial intelligence?",
    "What is machine learning?",
    "What are neural networks?",
    "What is natural language processing?",
    "What is cloud computing?",
    "What are the benefits of cloud computing?",
    "What are the main principles of software design?",
    "What is the relationship between artificial intelligence and machine learning?",
]


# ============================================================
# DISPLAY HELPER
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# PREPARE SMALLER WORKING DOCUMENTS
# ============================================================

def prepare_documents():

    working_dir = os.path.join(
        BASE_DIR,
        "rag_documents"
    )

    os.makedirs(
        working_dir,
        exist_ok=True
    )

    txt_files = [
        file
        for file in os.listdir(DOCUMENTS_DIR)
        if file.lower().endswith(".txt")
    ]

    if not txt_files:

        print(
            "ERROR: No TXT files found."
        )

        return None

    print()
    print(
        f"TXT files found: {len(txt_files)}"
    )

    for filename in txt_files:

        source_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        destination_path = os.path.join(
            working_dir,
            filename
        )

        with open(
            source_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            text = file.read(
                MAX_CHARS_PER_FILE
            )

        with open(
            destination_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

        original_size = os.path.getsize(
            source_path
        )

        print(
            f"  {filename}: "
            f"{original_size:,} bytes -> "
            f"{len(text):,} characters used"
        )

    return working_dir


# ============================================================
# SOURCE DISPLAY
# ============================================================

def print_sources(response):

    source_nodes = getattr(
        response,
        "source_nodes",
        []
    )

    if not source_nodes:

        print(
            "Sources: NONE"
        )

        return

    print(
        "Sources:"
    )

    for node in source_nodes[:TOP_K]:

        metadata = node.node.metadata

        source = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source")
            or "Unknown"
        )

        score = node.score

        if score is not None:

            print(
                f"  - {source} "
                f"| Score: {score:.4f}"
            )

        else:

            print(
                f"  - {source}"
            )


# ============================================================
# SOURCE VERIFICATION
# ============================================================

def verify_source(response):

    source_nodes = getattr(
        response,
        "source_nodes",
        []
    )

    return len(source_nodes) > 0


# ============================================================
# RUN 10 QUERIES
# ============================================================

def run_queries(
    query_engine,
    title
):

    print_header(title)

    latencies = []

    successful = 0

    source_verified = 0

    for number, query in enumerate(
        QUERIES,
        start=1
    ):

        print()
        print("-" * 70)

        print(
            f"Query {number}/10"
        )

        print(
            f"Question: {query}"
        )

        start = time.perf_counter()

        try:

            response = query_engine.query(
                query
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            latencies.append(
                elapsed
            )

            successful += 1

            print()
            print(
                "Answer:"
            )

            print(
                str(response)
            )

            print()

            print_sources(
                response
            )

            if verify_source(response):

                source_verified += 1

                print(
                    "Source Verification: PASS"
                )

            else:

                print(
                    "Source Verification: FAIL"
                )

            print(
                f"Latency: "
                f"{elapsed:.4f} seconds"
            )

        except Exception as error:

            print()
            print(
                "ERROR executing query:"
            )

            print(
                error
            )

    print()
    print("=" * 70)

    print(
        f"Queries completed: "
        f"{successful}/10"
    )

    print(
        f"Source verification passed: "
        f"{source_verified}/10"
    )

    if latencies:

        print(
            f"Average latency: "
            f"{statistics.mean(latencies):.4f} seconds"
        )

        print(
            f"Minimum latency: "
            f"{min(latencies):.4f} seconds"
        )

        print(
            f"Maximum latency: "
            f"{max(latencies):.4f} seconds"
        )

    return (
        latencies,
        successful,
        source_verified
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "W7D3 — LLAMAINDEX DOCUMENT INDEXING & QUERYING"
    )

    print(
        f"Documents directory: "
        f"{DOCUMENTS_DIR}"
    )

    print(
        f"Ollama embedding model: "
        f"{EMBED_MODEL}"
    )

    print(
        f"Ollama LLM model: "
        f"{LLM_MODEL}"
    )

    print(
        f"Ollama URL: "
        f"{OLLAMA_URL}"
    )

    print(
        f"Context window: "
        f"{CONTEXT_WINDOW}"
    )

    print(
        f"Top-K retrieval: "
        f"{TOP_K}"
    )

    # ========================================================
    # CHECK DOCUMENTS
    # ========================================================

    if not os.path.exists(
        DOCUMENTS_DIR
    ):

        print(
            "ERROR: documents folder not found."
        )

        return

    txt_files = [
        file
        for file in os.listdir(
            DOCUMENTS_DIR
        )
        if file.lower().endswith(".txt")
    ]

    print()
    print(
        f"Text documents found: "
        f"{len(txt_files)}"
    )

    for file in txt_files:

        print(
            f"  - {file}"
        )

    if len(txt_files) < 5:

        print(
            "ERROR: At least 5 TXT files required."
        )

        return

    # ========================================================
    # PREPARE DOCUMENTS
    # ========================================================

    print_header(
        "PREPARING TEXT DOCUMENTS"
    )

    working_dir = prepare_documents()

    if working_dir is None:

        return

    print()
    print(
        f"Working directory: "
        f"{working_dir}"
    )

    # ========================================================
    # OLLAMA EMBEDDING
    # ========================================================

    print_header(
        "CONFIGURING LOCAL OLLAMA"
    )

    try:

        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_URL,
        )

        Settings.embed_model = (
            embed_model
        )

        print(
            "Ollama embedding model configured successfully."
        )

        print(
            f"Embedding model: "
            f"{EMBED_MODEL}"
        )

    except Exception as error:

        print(
            "ERROR configuring embedding model:"
        )

        print(error)

        return

    # ========================================================
    # OLLAMA LLM
    # ========================================================

    try:

        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            request_timeout=120.0,
            context_window=CONTEXT_WINDOW,
            temperature=0.0,
        )

        Settings.llm = llm

        print(
            "Ollama LLM configured successfully."
        )

        print(
            f"LLM model: "
            f"{LLM_MODEL}"
        )

        print(
            f"Context window: "
            f"{CONTEXT_WINDOW}"
        )

    except Exception as error:

        print(
            "ERROR configuring LLM:"
        )

        print(error)

        return

    # ========================================================
    # CHUNKING
    # ========================================================

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    Settings.node_parser = splitter

    print(
        "SentenceSplitter configured successfully."
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP}"
    )

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    print_header(
        "LOADING TEXT DOCUMENTS"
    )

    try:

        documents = (
            SimpleDirectoryReader(
                input_dir=working_dir,
                recursive=True
            ).load_data()
        )

    except Exception as error:

        print(
            "ERROR loading documents:"
        )

        print(error)

        return

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "ERROR: No documents loaded."
        )

        return

    # ========================================================
    # PART 1
    # ========================================================

    print_header(
        "PART 1 — LLAMAINDEX VECTORSTOREINDEX"
    )

    print(
        "Creating VectorStoreIndex..."
    )

    print(
        "Embedding documents using Ollama..."
    )

    try:

        index = VectorStoreIndex.from_documents(
            documents
        )

    except Exception as error:

        print(
            "ERROR creating VectorStoreIndex:"
        )

        print(error)

        return

    print()
    print(
        "VectorStoreIndex created successfully."
    )

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K
    )

    print(
        "QueryEngine created successfully."
    )

    print(
        f"Similarity top-k: "
        f"{TOP_K}"
    )

    print(
        f"LLM: "
        f"{LLM_MODEL}"
    )

    (
        llama_latencies,
        llama_success,
        llama_sources
    ) = run_queries(
        query_engine,
        "PART 1 — LLAMAINDEX QUERYENGINE — 10 QUERIES"
    )

    # ========================================================
    # PART 2 — CHROMADB
    # ========================================================

    print_header(
        "CREATING CHROMADB VECTOR STORE"
    )

    try:

        print(
            "Creating ChromaDB persistent client..."
        )

        chroma_client = (
            chromadb.PersistentClient(
                path=CHROMA_DIR
            )
        )

        collection_name = (
            "w7d3_llamaindex_documents"
        )

        try:

            chroma_client.delete_collection(
                name=collection_name
            )

            print(
                "Previous collection deleted."
            )

        except Exception:

            pass

        chroma_collection = (
            chroma_client.get_or_create_collection(
                name=collection_name
            )
        )

        print(
            f"ChromaDB collection: "
            f"{collection_name}"
        )

        vector_store = (
            ChromaVectorStore(
                chroma_collection=chroma_collection
            )
        )

        print(
            "ChromaVectorStore created successfully."
        )

    except Exception as error:

        print(
            "ERROR creating ChromaDB:"
        )

        print(error)

        return

    print()
    print(
        "Creating LlamaIndex using ChromaDB..."
    )

    try:

        chroma_index = (
            VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store
            )
        )

    except Exception as error:

        print(
            "ERROR creating ChromaDB index:"
        )

        print(error)

        return

    print(
        "ChromaDB-backed VectorStoreIndex "
        "created successfully."
    )

    chroma_query_engine = (
        chroma_index.as_query_engine(
            similarity_top_k=TOP_K
        )
    )

    print(
        "QueryEngine created successfully."
    )

    print(
        f"Similarity top-k: "
        f"{TOP_K}"
    )

    print(
        f"LLM: "
        f"{LLM_MODEL}"
    )

    (
        chroma_latencies,
        chroma_success,
        chroma_sources
    ) = run_queries(
        chroma_query_engine,
        "PART 2 — CHROMADB QUERYENGINE — 10 QUERIES"
    )

    # ========================================================
    # PART 3 — LATENCY
    # ========================================================

    print_header(
        "PART 3 — LLAMAINDEX VS CHROMADB LATENCY COMPARISON"
    )

    if not llama_latencies:

        print(
            "No successful LlamaIndex queries."
        )

        return

    if not chroma_latencies:

        print(
            "No successful ChromaDB queries."
        )

        return

    llama_average = statistics.mean(
        llama_latencies
    )

    chroma_average = statistics.mean(
        chroma_latencies
    )

    difference = abs(
        llama_average
        - chroma_average
    )

    print()
    print(
        f"{'Query':<10}"
        f"{'LlamaIndex':<20}"
        f"{'ChromaDB':<20}"
    )

    print(
        "-" * 50
    )

    for i, (
        llama_time,
        chroma_time
    ) in enumerate(
        zip(
            llama_latencies,
            chroma_latencies
        ),
        start=1
    ):

        print(
            f"{i:<10}"
            f"{llama_time:<20.4f}"
            f"{chroma_time:<20.4f}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "AVERAGE LATENCY"
    )

    print(
        "=" * 70
    )

    print(
        f"LlamaIndex VectorStoreIndex : "
        f"{llama_average:.4f} seconds"
    )

    print(
        f"LlamaIndex + ChromaDB       : "
        f"{chroma_average:.4f} seconds"
    )

    print(
        f"Absolute difference          : "
        f"{difference:.4f} seconds"
    )

    if llama_average < chroma_average:

        faster = (
            "LlamaIndex VectorStoreIndex"
        )

    elif chroma_average < llama_average:

        faster = (
            "LlamaIndex + ChromaDB"
        )

    else:

        faster = (
            "Both are approximately equal"
        )

    print(
        f"Faster average: "
        f"{faster}"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header(
        "W7D3 PRACTICAL COMPLETION SUMMARY"
    )

    task1 = (
        llama_success == 10
    )

    task2 = (
        llama_success == 10
        and llama_sources == 10
    )

    task3 = (
        chroma_success == 10
        and chroma_sources == 10
        and len(llama_latencies) == 10
        and len(chroma_latencies) == 10
    )

    print(
        "Task 1 — LlamaIndex VectorStoreIndex "
        "+ Ollama embeddings : "
        + ("COMPLETED" if task1 else "INCOMPLETE")
    )

    print(
        "Task 2 — QueryEngine + 10 queries "
        "+ source verification : "
        + ("COMPLETED" if task2 else "INCOMPLETE")
    )

    print(
        "Task 3 — ChromaDB + 10 queries "
        "+ latency comparison : "
        + ("COMPLETED" if task3 else "INCOMPLETE")
    )

    print()
    print(
        f"LlamaIndex queries: "
        f"{llama_success}/10"
    )

    print(
        f"LlamaIndex source verification: "
        f"{llama_sources}/10"
    )

    print(
        f"ChromaDB queries: "
        f"{chroma_success}/10"
    )

    print(
        f"ChromaDB source verification: "
        f"{chroma_sources}/10"
    )

    print()

    if task1 and task2 and task3:

        print(
            "=" * 70
        )

        print(
            "W7D3 — ALL PRACTICAL TASKS COMPLETED"
        )

        print(
            "=" * 70
        )

    else:

        print(
            "=" * 70
        )

        print(
            "W7D3 — PRACTICAL TASKS NOT YET COMPLETE"
        )

        print(
            "Fix the failed queries before pushing."
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":

    main()