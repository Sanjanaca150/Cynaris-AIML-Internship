from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():
    """Test the health-check endpoint."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_research_endpoint():
    """Test a valid research question."""

    response = client.post(
        "/research",
        json={"question": "What is Retrieval-Augmented Generation?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is Retrieval-Augmented Generation?"
    assert "Retrieval-Augmented Generation" in data["answer"]
    assert data["source"] == "data/knowledge_base.txt"


def test_empty_question():
    """Test that an empty question is rejected."""

    response = client.post(
        "/research",
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_unknown_question():
    """Test that an unsupported question returns 404."""

    response = client.post(
        "/research",
        json={"question": "What is quantum teleportation?"},
    )

    assert response.status_code == 404