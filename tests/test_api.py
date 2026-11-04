from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_batch_is_rejected():
    assert client.post("/predict_batch", json={"comments": []}).status_code == 422


def test_blank_comment_is_rejected_or_model_unavailable():
    response = client.post("/predict_batch", json={"comments": [" "]})
    assert response.status_code in (422, 503)
