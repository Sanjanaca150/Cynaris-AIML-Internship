from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)

DATA_DIR = Path("data/documents")


def load_documents():
    """Load all text documents from the document directory."""
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                },
            )
        )

    return documents


def split_documents(documents):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    """Create embeddings and store them in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def main():
    """Run the complete document ingestion pipeline."""
    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    documents = load_documents()

    if not documents:
        raise FileNotFoundError(
            "No .txt documents found in data/documents/"
        )

    print(f"\nLoaded {len(documents)} documents.")

    for document in documents:
        print(f"  - {document.metadata['source']}")

    chunks = split_documents(documents)

    print(f"\nCreated {len(chunks)} document chunks.")

    create_vector_store(chunks)

    print("\nChromaDB vector store created successfully.")
    print(f"Database location: {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()