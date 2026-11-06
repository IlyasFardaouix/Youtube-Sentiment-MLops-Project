import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model

client = TestClient(app)
requires_model = pytest.mark.skipif(model is None, reason="model artifact not available")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_empty_batch_is_rejected():
    assert client.post("/predict_batch", json={"comments": []}).status_code == 422


def test_batch_over_limit_is_rejected():
    assert client.post("/predict_batch", json={"comments": ["x"] * 501}).status_code == 422


def test_invalid_payload_is_rejected():
    assert client.post("/predict_batch", json={"comments": "not a list"}).status_code == 422


@requires_model
def test_blank_comment_is_rejected():
    assert client.post("/predict_batch", json={"comments": [" "]}).status_code == 422


@requires_model
def test_predict_batch_returns_consistent_results():
    comments = ["I love this video", "This is terrible", "It is a video about cats"]
    response = client.post("/predict_batch", json={"comments": comments})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == len(comments)
    for item in data["predictions"]:
        assert item["label"] in (-1, 0, 1)
        assert item["sentiment"] in ("negative", "neutral", "positive")
        assert 0.0 <= item["confidence"] <= 1.0
    stats = data["statistics"]
    assert stats["total"] == len(comments)
    assert sum(stats["counts"].values()) == len(comments)
    assert stats["latency_ms"] >= 0


@requires_model
def test_batch_of_fifty():
    response = client.post("/predict_batch", json={"comments": ["great video"] * 50})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 50
