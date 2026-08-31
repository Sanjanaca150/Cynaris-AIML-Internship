from pathlib import Path

from haystack import Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


# ============================================================
# W8D3 - HAYSTACK DENSE RETRIEVAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CHECK PDF FILES
# ============================================================

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

print("=" * 70)
print("W8D3 - HAYSTACK DENSE RETRIEVAL")
print("=" * 70)

print(f"\nPDF files found: {len(pdf_files)}")

for pdf in pdf_files:
    print(f"  - {pdf.name}")

if len(pdf_files) != 5:
    raise ValueError(
        f"Expected 5 PDF files, but found {len(pdf_files)}"
    )


# ============================================================
# 2. CREATE DOCUMENT STORE
# ============================================================

document_store = InMemoryDocumentStore()

print("\nDocumentStore created.")


# ============================================================
# 3. CONVERT PDF FILES
# ============================================================

converter = PyPDFToDocument()
all_documents = []

print("\nConverting PDF files...")

for pdf_file in pdf_files:
    print(f"  Converting: {pdf_file.name}")

    result = converter.run(
        sources=[pdf_file]
    )

    documents = result["documents"]

    for document in documents:
        document.meta["source"] = pdf_file.name

    all_documents.extend(documents)


print("\nPDF conversion completed.")
print(f"Documents created: {len(all_documents)}")


# ============================================================
# 4. SPLIT DOCUMENTS
# ============================================================

splitter = DocumentSplitter(
    split_by="word",
    split_length=150,
    split_overlap=30
)

split_result = splitter.run(
    documents=all_documents
)

split_documents = split_result["documents"]

print(f"Document chunks created: {len(split_documents)}")


# ============================================================
# 5. CREATE DOCUMENT EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")
print("Model: sentence-transformers/all-MiniLM-L6-v2")

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

document_embedder.warm_up()

print("Embedding model loaded.")

print("\nCreating embeddings for document chunks...")

embedding_result = document_embedder.run(
    documents=split_documents
)

embedded_documents = embedding_result["documents"]

print(
    f"Embeddings created for "
    f"{len(embedded_documents)} chunks."
)


# ============================================================
# 6. STORE EMBEDDED DOCUMENTS
# ============================================================

writer = DocumentWriter(
    document_store=document_store
)

writer.run(
    documents=embedded_documents
)

print(
    f"Documents stored: "
    f"{document_store.count_documents()}"
)


# ============================================================
# 7. CREATE DENSE RETRIEVER
# ============================================================

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store
)

print("Dense Retriever created.")


# ============================================================
# 8. TEXT EMBEDDER FOR QUESTIONS
# ============================================================

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()

print("Question text embedder ready.")


# ============================================================
# 9. SAME 10 QUESTIONS USED FOR BM25
# ============================================================

questions = [
    "What is artificial intelligence?",
    "What are the applications of artificial intelligence?",
    "What is cloud computing?",
    "What are the benefits of cloud computing?",
    "What is software engineering?",
    "What are the phases of the software development life cycle?",
    "What is software design?",
    "What are the principles of software design?",
    "What is the relationship between software engineering and software design?",
    "What are the common challenges in software development?"
]


# ============================================================
# 10. RUN DENSE RETRIEVAL
# ============================================================

output_file = OUTPUTS_DIR / "dense_results.txt"

with open(output_file, "w", encoding="utf-8") as output:

    output.write("W8D3 HAYSTACK DENSE RETRIEVAL RESULTS\n")
    output.write("=" * 70 + "\n\n")

    for question_number, question in enumerate(
        questions,
        start=1
    ):

        print("\n" + "=" * 70)
        print(
            f"QUESTION {question_number}: {question}"
        )
        print("=" * 70)

        query_embedding = text_embedder.run(
            text=question
        )["embedding"]

        result = retriever.run(
            query_embedding=query_embedding,
            top_k=3
        )

        retrieved_documents = result["documents"]

        output.write(
            f"QUESTION {question_number}: {question}\n"
        )

        output.write("-" * 70 + "\n")

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            source = document.meta.get(
                "source",
                "Unknown"
            )

            score = document.score

            content = document.content.replace(
                "\n",
                " "
            )

            print(f"\nRank {rank}")
            print(f"Source: {source}")
            print(f"Score: {score}")
            print(f"Content: {content[:300]}...")

            output.write(f"\nRank {rank}\n")
            output.write(f"Source: {source}\n")
            output.write(f"Score: {score}\n")
            output.write(
                f"Content: {content[:500]}...\n"
            )

        output.write(
            "\n" + "=" * 70 + "\n\n"
        )


# ============================================================
# 11. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DENSE RETRIEVAL COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved to:")
print(output_file)
