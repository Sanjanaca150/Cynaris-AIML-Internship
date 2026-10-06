from pathlib import Path

from app.ingest import load_documents, split_documents


def test_documents_exist():
    documents = load_documents()

    assert len(documents) >= 3


def test_documents_have_source_metadata():
    documents = load_documents()

    for document in documents:
        assert "source" in document.metadata
        assert document.metadata["source"]


def test_document_splitting():
    documents = load_documents()
    chunks = split_documents(documents)

    assert len(chunks) > len(documents)


def test_document_folder_exists():
    assert Path("data/documents").exists()