"""
W8D4 - API Tests

Tests cover:
- Root endpoint
- Health endpoint
- Successful knowledge-base query
- Unknown query handling
- Input validation
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    """Verify that the root endpoint returns API information."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_health_check():
    """Verify that the health endpoint reports a healthy service."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_machine_learning_query():
    """Verify that a known topic returns an answer."""
    response = client.post(
        "/query",
        json={"question": "What is machine learning?"},
    )

    assert response.status_code == 200
    assert "machine learning" in response.json()["answer"].lower()


def test_rag_query():
    """Verify that the RAG topic returns an appropriate answer."""
    response = client.post(
        "/query",
        json={"question": "Explain RAG"},
    )

    assert response.status_code == 200
    assert "retrieval" in response.json()["answer"].lower()


def test_unknown_query():
    """Verify that an unsupported question returns HTTP 404."""
    response = client.post(
        "/query",
        json={"question": "Tell me about quantum physics"},
    )

    assert response.status_code == 404


def test_short_question_validation():
    """Verify that questions shorter than three characters are rejected."""
    response = client.post(
        "/query",
        json={"question": "AI"},
    )

    assert response.status_code == 422