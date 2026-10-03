"""
Build a ChromaDB vector store from the W11D5 healthcare knowledge base.
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "healthcare_knowledge.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text:latest"


def build_vectorstore():
    """Load healthcare knowledge and store embeddings in ChromaDB."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Healthcare dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(encoding="utf-8")

    sections = [
        section.strip()
        for section in text.split("\n\n")
        if section.strip()
    ]

    documents = [
        Document(
            page_content=section,
            metadata={"source": "healthcare_knowledge.txt"},
        )
        for section in sections
    ]

    # Remove the old vector store to prevent duplicate documents.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print("Creating healthcare ChromaDB...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="w11d5_healthcare",
        persist_directory=str(CHROMA_DIR),
    )

    print("=" * 60)
    print("W11D5 CHROMA VECTOR STORE")
    print("=" * 60)
    print(f"Healthcare documents indexed: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector store: {CHROMA_DIR}")
    print("=" * 60)

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()