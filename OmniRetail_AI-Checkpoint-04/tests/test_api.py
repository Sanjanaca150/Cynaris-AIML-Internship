import sys
from pathlib import Path

from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "OmniRetail AI"
    assert data["checkpoint"] == "Checkpoint 4"
    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "RandomForestClassifier"


def test_prediction_endpoint():
    payload = {
        "product_views": 5,
        "add_to_cart": 2,
        "session_duration_mins": 8.5,
        "return_visitor": 1,
        "discount_applied": 1,
        "recommendation_clicked": 1,
        "city": "Delhi",
        "device": "mobile",
        "category": "electronics",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "purchase_probability" in data
    assert "model" in data

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["purchase_probability"] <= 1.0
    assert data["model"] == "RandomForestClassifier"


def test_prediction_with_different_input():
    payload = {
        "product_views": 10,
        "add_to_cart": 0,
        "session_duration_mins": 2.0,
        "return_visitor": 0,
        "discount_applied": 0,
        "recommendation_clicked": 0,
        "city": "Mumbai",
        "device": "desktop",
        "category": "fashion",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["purchase_probability"] <= 1.0


def test_invalid_prediction_request():
    payload = {
        "product_views": -5,
        "add_to_cart": 2,
        "session_duration_mins": 8.5,
        "return_visitor": 1,
        "discount_applied": 1,
        "recommendation_clicked": 1,
        "city": "Delhi",
        "device": "mobile",
        "category": "electronics",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422