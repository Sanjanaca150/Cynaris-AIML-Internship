from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers import InMemoryEmbeddingRetriever

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)

from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
)


# ============================================================
# W7D1 — HAYSTACK PIPELINE ARCHITECTURE
# ============================================================

DOCUMENTS_DIR = Path("documents")


# ============================================================
# STEP 1 — FIND THE 5 PDF FILES
# ============================================================

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

print("=" * 70)
print("W7D1 — HAYSTACK PIPELINE")
print("=" * 70)

print(f"\nFound {len(pdf_files)} PDF files:")

for pdf in pdf_files:
    print(f"  - {pdf.name}")


if len(pdf_files) != 5:
    raise RuntimeError(
        f"Expected 5 PDF files, but found {len(pdf_files)}"
    )


# ============================================================
# STEP 2 — PDF CONVERSION
# ============================================================

print("\nConverting PDFs...")

converter = PyPDFToDocument()

conversion_result = converter.run(
    sources=pdf_files
)

documents = conversion_result["documents"]

print(f"Converted documents: {len(documents)}")

# Add source filename to metadata
for document in documents:
    if "file_path" in document.meta:
        document.meta["source"] = Path(
            document.meta["file_path"]
        ).name


# ============================================================
# STEP 3 — CLEAN DOCUMENTS
# ============================================================

print("\nCleaning documents...")

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
)

clean_result = cleaner.run(
    documents=documents
)

clean_documents = clean_result["documents"]

print(f"Clean documents: {len(clean_documents)}")


# ============================================================
# STEP 4 — SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

print("\nSplitting documents into chunks...")

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=5,
    split_overlap=1,
)

split_result = splitter.run(
    documents=clean_documents
)

chunks = split_result["documents"]

print(f"Total chunks: {len(chunks)}")


# ============================================================
# STEP 5 — CREATE DOCUMENT STORE
# ============================================================

print("\nCreating document store...")

document_store = InMemoryDocumentStore()

writer = DocumentWriter(
    document_store=document_store
)

writer.run(
    documents=chunks
)

print(
    f"Documents stored successfully: "
    f"{document_store.count_documents()}"
)


# ============================================================
# STEP 6 — CREATE BM25 RETRIEVER
# ============================================================

print("\nCreating BM25 retriever...")

bm25_retriever = InMemoryBM25Retriever(
    document_store=document_store
)


# ============================================================
# STEP 7 — CREATE HAYSTACK PIPELINE
# ============================================================

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    bm25_retriever
)


print("\nBM25 pipeline created successfully.")


# ============================================================
# STEP 8 — RUN 10 QUESTIONS USING BM25
# ============================================================

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is cloud computing?",
    "What is deep learning?",
    "What is natural language processing?",
    "What are neural networks?",
    "What is data processing?",
    "What is software architecture?",
    "What is software design?",
    "What is the purpose of software engineering?",
]


print("\n")
print("=" * 70)
print("BM25 RETRIEVAL RESULTS")
print("=" * 70)


bm25_results = []


for number, question in enumerate(questions, start=1):

    print(f"\nQuestion {number}: {question}")
    print("-" * 70)

    result = bm25_retriever.run(
        query=question,
        top_k=3
    )

    retrieved_documents = result["documents"]

    bm25_results.append(
        {
            "question": question,
            "documents": retrieved_documents
        }
    )

    if not retrieved_documents:

        print("No documents retrieved.")
        continue


    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source = document.meta.get(
            "source",
            "Unknown"
        )

        score = document.score

        print(
            f"\nResult {rank}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"BM25 Score: {score}"
        )

        preview = document.content[:300]

        preview = preview.replace(
            "\n",
            " "
        )

        print(
            f"Content: {preview}..."
        )


print("\n")
print("=" * 70)
print("BM25 — 10 QUESTIONS COMPLETED")
print("=" * 70)

# ============================================================
# STEP 9 — DENSE RETRIEVAL
# ============================================================



print("\n")
print("=" * 70)
print("DENSE RETRIEVAL")
print("=" * 70)


# ------------------------------------------------------------
# STEP 9.1 — CREATE EMBEDDING MODEL
# ------------------------------------------------------------

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading document embedding model...")

document_embedder = SentenceTransformersDocumentEmbedder(
    model=embedding_model
)

document_embedder.warm_up()

print("Document embedding model loaded.")


# ------------------------------------------------------------
# STEP 9.2 — CREATE DENSE DOCUMENT STORE
# ------------------------------------------------------------

dense_document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)


# ------------------------------------------------------------
# STEP 9.3 — CREATE DOCUMENT EMBEDDINGS
# ------------------------------------------------------------

print("\nCreating embeddings for document chunks...")

embedding_result = document_embedder.run(
    documents=chunks
)

embedded_documents = embedding_result["documents"]

print(
    f"Created embeddings for "
    f"{len(embedded_documents)} chunks."
)


# ------------------------------------------------------------
# STEP 9.4 — WRITE EMBEDDED DOCUMENTS
# ------------------------------------------------------------

dense_writer = DocumentWriter(
    document_store=dense_document_store
)

dense_writer.run(
    documents=embedded_documents
)

print(
    f"Dense document store contains "
    f"{dense_document_store.count_documents()} documents."
)


# ------------------------------------------------------------
# STEP 9.5 — CREATE TEXT EMBEDDER
# ------------------------------------------------------------

print("\nLoading query embedding model...")

text_embedder = SentenceTransformersTextEmbedder(
    model=embedding_model
)

text_embedder.warm_up()

print("Query embedding model loaded.")


# ------------------------------------------------------------
# STEP 9.6 — CREATE DENSE RETRIEVER
# ------------------------------------------------------------

dense_retriever = InMemoryEmbeddingRetriever(
    document_store=dense_document_store
)


# ------------------------------------------------------------
# STEP 9.7 — RUN THE SAME 10 QUESTIONS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("DENSE RETRIEVAL RESULTS")
print("=" * 70)


dense_results = []


for number, question in enumerate(
    questions,
    start=1
):

    print(
        f"\nQuestion {number}: {question}"
    )

    print("-" * 70)


    # Convert question into an embedding
    query_embedding_result = text_embedder.run(
        text=question
    )

    query_embedding = query_embedding_result[
        "embedding"
    ]


    # Retrieve top 3 documents
    retrieval_result = dense_retriever.run(
        query_embedding=query_embedding,
        top_k=3
    )


    retrieved_documents = retrieval_result[
        "documents"
    ]


    dense_results.append(
        {
            "question": question,
            "documents": retrieved_documents
        }
    )


    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source = document.meta.get(
            "source",
            "Unknown"
        )

        score = document.score


        print(
            f"\nResult {rank}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Dense Score: {score:.4f}"
        )


        preview = document.content[:300]

        preview = preview.replace(
            "\n",
            " "
        )


        print(
            f"Content: {preview}..."
        )


print("\n")
print("=" * 70)
print("DENSE RETRIEVAL — 10 QUESTIONS COMPLETED")
print("=" * 70)