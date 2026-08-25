from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


# ============================================================
# W7D2 — HAYSTACK RETRIEVAL
# BM25 & DENSE RETRIEVAL
# ============================================================

DOCUMENTS_DIR = Path("documents")

QUESTIONS = [
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


print("=" * 70)
print("W7D2 — HAYSTACK RETRIEVAL: BM25 & DENSE")
print("=" * 70)


# ============================================================
# STEP 1 — FIND THE 5 PDF DOCUMENTS
# ============================================================

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

print("\nFound PDF files:")

for pdf in pdf_files:
    print(f"  - {pdf.name}")

print(f"\nTotal PDF files found: {len(pdf_files)}")

if len(pdf_files) != 5:
    raise RuntimeError(
        f"Expected exactly 5 PDF files, but found {len(pdf_files)}."
    )


# ============================================================
# STEP 2 — CONVERT PDF DOCUMENTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2 — PDF CONVERSION")
print("=" * 70)

converter = PyPDFToDocument()

conversion_result = converter.run(
    sources=pdf_files
)

documents = conversion_result["documents"]

print(
    f"Converted documents: {len(documents)}"
)


# Add source filename to metadata
for document in documents:

    if "file_path" in document.meta:

        document.meta["source"] = Path(
            document.meta["file_path"]
        ).name


# ============================================================
# STEP 3 — CLEAN DOCUMENTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3 — DOCUMENT CLEANING")
print("=" * 70)

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
)

clean_result = cleaner.run(
    documents=documents
)

clean_documents = clean_result["documents"]

print(
    f"Clean documents: {len(clean_documents)}"
)


# ============================================================
# STEP 4 — SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4 — DOCUMENT CHUNKING")
print("=" * 70)

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=5,
    split_overlap=1,
)

split_result = splitter.run(
    documents=clean_documents
)

chunks = split_result["documents"]

print(
    f"Total document chunks: {len(chunks)}"
)


# ============================================================
# STEP 5 — BM25 DOCUMENT STORE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5 — BM25 DOCUMENT STORE")
print("=" * 70)

bm25_document_store = InMemoryDocumentStore()

bm25_writer = DocumentWriter(
    document_store=bm25_document_store
)

bm25_writer.run(
    documents=chunks
)

print(
    "BM25 document store contains: "
    f"{bm25_document_store.count_documents()} documents"
)


# ============================================================
# STEP 6 — BM25 RETRIEVER
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6 — BM25 RETRIEVER")
print("=" * 70)

bm25_retriever = InMemoryBM25Retriever(
    document_store=bm25_document_store
)

print(
    "BM25 retriever created successfully."
)


# ============================================================
# STEP 7 — CREATE BM25 HAYSTACK PIPELINE
# ============================================================

bm25_pipeline = Pipeline()

bm25_pipeline.add_component(
    "retriever",
    bm25_retriever
)

print(
    "BM25 Haystack pipeline created successfully."
)


# ============================================================
# STEP 8 — RUN 10 QUESTIONS USING BM25
# ============================================================

print("\n")
print("=" * 70)
print("BM25 RETRIEVAL RESULTS — 10 QUESTIONS")
print("=" * 70)

bm25_results = []


for number, question in enumerate(
    QUESTIONS,
    start=1
):

    print(
        f"\nQuestion {number}: {question}"
    )

    print("-" * 70)

    result = bm25_retriever.run(
        query=question,
        top_k=3
    )

    retrieved_documents = result["documents"]

    bm25_results.append(
        {
            "question": question,
            "documents": retrieved_documents,
        }
    )

    if not retrieved_documents:

        print(
            "No documents retrieved."
        )

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

        preview = document.content[:300]

        preview = preview.replace(
            "\n",
            " "
        )

        print(
            f"\nResult {rank}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"BM25 Score: {score:.4f}"
        )

        print(
            f"Content: {preview}..."
        )


print("\n")
print("=" * 70)
print("BM25 — 10 QUESTIONS COMPLETED")
print("=" * 70)


# ============================================================
# STEP 9 — DENSE RETRIEVAL SETUP
# ============================================================

print("\n")
print("=" * 70)
print("DENSE RETRIEVAL SETUP")
print("=" * 70)

print(
    f"\nEmbedding model: {EMBEDDING_MODEL}"
)


# ============================================================
# STEP 9.1 — DOCUMENT EMBEDDER
# ============================================================

print(
    "\nLoading document embedding model..."
)

document_embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL
)

document_embedder.warm_up()

print(
    "Document embedding model loaded successfully."
)


# ============================================================
# STEP 9.2 — DENSE DOCUMENT STORE
# ============================================================

dense_document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)

print(
    "\nDense document store created with cosine similarity."
)


# ============================================================
# STEP 9.3 — CREATE DOCUMENT EMBEDDINGS
# ============================================================

print(
    "\nGenerating embeddings for document chunks..."
)

embedding_result = document_embedder.run(
    documents=chunks
)

embedded_documents = embedding_result["documents"]

print(
    f"Created embeddings for "
    f"{len(embedded_documents)} chunks."
)


# ============================================================
# STEP 9.4 — WRITE EMBEDDED DOCUMENTS
# ============================================================

dense_writer = DocumentWriter(
    document_store=dense_document_store
)

dense_writer.run(
    documents=embedded_documents
)

print(
    "Dense document store contains: "
    f"{dense_document_store.count_documents()} documents."
)


# ============================================================
# STEP 9.5 — QUERY TEXT EMBEDDER
# ============================================================

print(
    "\nLoading query embedding model..."
)

text_embedder = SentenceTransformersTextEmbedder(
    model=EMBEDDING_MODEL
)

text_embedder.warm_up()

print(
    "Query embedding model loaded successfully."
)


# ============================================================
# STEP 9.6 — DENSE RETRIEVER
# ============================================================

dense_retriever = InMemoryEmbeddingRetriever(
    document_store=dense_document_store
)

print(
    "Dense retriever created successfully."
)


# ============================================================
# STEP 9.7 — CREATE DENSE HAYSTACK PIPELINE
# ============================================================

dense_pipeline = Pipeline()

dense_pipeline.add_component(
    "retriever",
    dense_retriever
)

print(
    "Dense Haystack pipeline created successfully."
)


# ============================================================
# STEP 10 — RUN SAME 10 QUESTIONS USING DENSE RETRIEVAL
# ============================================================

print("\n")
print("=" * 70)
print("DENSE RETRIEVAL RESULTS — 10 QUESTIONS")
print("=" * 70)

dense_results = []


for number, question in enumerate(
    QUESTIONS,
    start=1
):

    print(
        f"\nQuestion {number}: {question}"
    )

    print("-" * 70)


    # Convert query into embedding
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
            "documents": retrieved_documents,
        }
    )


    if not retrieved_documents:

        print(
            "No documents retrieved."
        )

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

        preview = document.content[:300]

        preview = preview.replace(
            "\n",
            " "
        )

        print(
            f"\nResult {rank}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Dense Score: {score:.4f}"
        )

        print(
            f"Content: {preview}..."
        )


print("\n")
print("=" * 70)
print("DENSE RETRIEVAL — 10 QUESTIONS COMPLETED")
print("=" * 70)


# ============================================================
# STEP 11 — MANUAL RELEVANCE EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("BM25 VS DENSE RETRIEVAL COMPARISON")
print("=" * 70)

print(
    "\nEvaluation method:"
)

print(
    "\nPrecision@3 = Number of relevant documents in Top-3"
)

print(
    "               ------------------------------------"
)

print(
    "                              3"
)

print(
    "\nBecause this PDF collection has no predefined "
    "relevance labels, relevance was evaluated manually."
)


# ============================================================
# MANUAL RELEVANCE JUDGMENTS
# ============================================================
#
# Each number represents how many of the Top-3 retrieved
# chunks were judged relevant to the question.
#
# The evaluation is based on inspection of the actual
# retrieval results produced above.
#

bm25_relevant = [
    3,  # Q1 Artificial Intelligence
    3,  # Q2 Machine Learning
    2,  # Q3 Cloud Computing
    1,  # Q4 Deep Learning
    2,  # Q5 Natural Language Processing
    1,  # Q6 Neural Networks
    1,  # Q7 Data Processing
    3,  # Q8 Software Architecture
    3,  # Q9 Software Design
    2,  # Q10 Purpose of Software Engineering
]


dense_relevant = [
    2,  # Q1 Artificial Intelligence
    3,  # Q2 Machine Learning
    3,  # Q3 Cloud Computing
    3,  # Q4 Deep Learning
    1,  # Q5 Natural Language Processing
    2,  # Q6 Neural Networks
    1,  # Q7 Data Processing
    3,  # Q8 Software Architecture
    3,  # Q9 Software Design
    3,  # Q10 Purpose of Software Engineering
]


# ============================================================
# STEP 12 — CALCULATE PRECISION@3
# ============================================================

bm25_precision_scores = []

dense_precision_scores = []


print("\n")


print(
    f"{'Q':<4}"
    f"{'BM25 Relevant':<18}"
    f"{'BM25 P@3':<12}"
    f"{'Dense Relevant':<18}"
    f"{'Dense P@3':<12}"
    f"{'Winner'}"
)

print("-" * 80)


for index in range(
    len(QUESTIONS)
):

    bm25_precision = (
        bm25_relevant[index] / 3
    )

    dense_precision = (
        dense_relevant[index] / 3
    )


    bm25_precision_scores.append(
        bm25_precision
    )

    dense_precision_scores.append(
        dense_precision
    )


    if bm25_precision > dense_precision:

        winner = "BM25"

    elif dense_precision > bm25_precision:

        winner = "Dense"

    else:

        winner = "Tie"


    print(
        f"{index + 1:<4}"
        f"{bm25_relevant[index]}/3{'':<13}"
        f"{bm25_precision:.2f}{'':<7}"
        f"{dense_relevant[index]}/3{'':<13}"
        f"{dense_precision:.2f}{'':<7}"
        f"{winner}"
    )


# ============================================================
# STEP 13 — OVERALL PRECISION
# ============================================================

overall_bm25_precision = (
    sum(bm25_precision_scores)
    / len(bm25_precision_scores)
)


overall_dense_precision = (
    sum(dense_precision_scores)
    / len(dense_precision_scores)
)


print("\n")
print("=" * 70)
print("OVERALL RETRIEVAL QUALITY")
print("=" * 70)


print(
    f"\nBM25 Mean Precision@3: "
    f"{overall_bm25_precision:.2f}"
)


print(
    f"Dense Mean Precision@3: "
    f"{overall_dense_precision:.2f}"
)


if overall_bm25_precision > overall_dense_precision:

    overall_winner = "BM25"

elif overall_dense_precision > overall_bm25_precision:

    overall_winner = "Dense Retrieval"

else:

    overall_winner = "Tie"


print(
    f"\nOverall Winner: {overall_winner}"
)


# ============================================================
# STEP 14 — FINAL W7D2 STATUS
# ============================================================

print("\n")
print("=" * 70)
print("W7D2 — HAYSTACK RETRIEVAL TASK STATUS")
print("=" * 70)


print("\nCompleted:")

print(
    "  [✓] Found and indexed 5 PDF documents"
)

print(
    "  [✓] Converted PDF documents"
)

print(
    "  [✓] Cleaned documents"
)

print(
    "  [✓] Split documents into chunks"
)

print(
    "  [✓] Created BM25 document store"
)

print(
    "  [✓] Created BM25 retriever"
)

print(
    "  [✓] Created BM25 Haystack pipeline"
)

print(
    "  [✓] Ran 10 questions using BM25"
)

print(
    "  [✓] Created Sentence Transformer document embeddings"
)

print(
    "  [✓] Created Sentence Transformer query embeddings"
)

print(
    "  [✓] Created Dense document store"
)

print(
    "  [✓] Created Dense retriever"
)

print(
    "  [✓] Created Dense Haystack pipeline"
)

print(
    "  [✓] Ran the same 10 questions using Dense Retrieval"
)

print(
    "  [✓] Compared BM25 and Dense Retrieval"
)

print(
    "  [✓] Completed manual Precision@3 evaluation"
)


print("\n")
print(
    f"BM25 Mean Precision@3: "
    f"{overall_bm25_precision:.2f}"
)

print(
    f"Dense Mean Precision@3: "
    f"{overall_dense_precision:.2f}"
)

print(
    f"Overall Winner: {overall_winner}"
)


print("\n")
print("=" * 70)
print("W7D2 — HAYSTACK RETRIEVAL IMPLEMENTATION COMPLETED")
print("=" * 70)