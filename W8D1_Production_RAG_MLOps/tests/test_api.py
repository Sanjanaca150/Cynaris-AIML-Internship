from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Production RAG API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_machine_learning_query():
    response = client.post(
        "/query",
        json={"query": "What is machine learning?"}
    )

    assert response.status_code == 200
    assert "machine learning" in response.json()["answer"].lower()


def test_unknown_query():
    response = client.post(
        "/query",
        json={"query": "What is quantum teleportation?"}
    )

    assert response.status_code == 200
    assert "No relevant information" in response.json()["answer"]