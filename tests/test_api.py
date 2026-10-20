"""API tests using FastAPI's TestClient (model trained on synthetic data)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_files, monkeypatch, tmp_path):
    model_path, meta_path = model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METADATA_PATH", str(meta_path))
    from api.main import app

    with TestClient(app) as c:  # context manager triggers the lifespan (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid_input(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_name"] == "test_lr"
    assert "X-Request-ID" in r.headers


def test_predict_invalid_input_returns_422(client, sample_record):
    sample_record["thal"] = 5
    assert client.post("/predict", json=sample_record).status_code == 422


def test_predict_missing_field_returns_422(client, sample_record):
    sample_record.pop("chol")
    assert client.post("/predict", json=sample_record).status_code == 422


def test_batch_predict(client, sample_record):
    r = client.post("/predict/batch", json={"records": [sample_record, sample_record]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_metrics_endpoint(client, sample_record):
    client.post("/predict", json=sample_record)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "api_requests_total" in r.text
    assert "model_predictions_total" in r.text
