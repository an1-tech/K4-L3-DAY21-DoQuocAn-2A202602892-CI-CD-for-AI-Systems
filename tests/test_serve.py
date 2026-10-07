from pathlib import Path
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from src import serve


@pytest.fixture
def saved_model(tmp_path):
    path = tmp_path / "model.joblib"
    X = pd.DataFrame([[0] * 10, [1] * 10], columns=serve.FEATURE_NAMES)
    model = DummyClassifier(strategy="constant", constant=1).fit(X, [0, 1])
    joblib.dump(model, path)
    return path


@pytest.fixture
def local_client(saved_model, monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", str(saved_model))
    with TestClient(serve.app) as client:
        yield client


def test_health_and_prediction(local_client):
    assert local_client.get("/healthz").json() == {"status": "ok"}
    response = local_client.post("/score", json={"features": [28, 2, 14, 2, 11, 0, 1, 0, 0, 45]})
    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "label": "thu_nhap_cao"}


@pytest.mark.parametrize("features", [[], [1] * 9, [1] * 11])
def test_wrong_feature_count(local_client, features):
    assert local_client.post("/score", json={"features": features}).status_code == 400


def test_invalid_input(local_client):
    assert local_client.post("/score", json={"features": ["bad"] * 10}).status_code == 422
    # Overflow to infinity is valid JSON but is not a valid model input.
    body = '{"features": [1e999, 2, 14, 2, 11, 0, 1, 0, 0, 45]}'
    response = local_client.post("/score", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400


def test_unloaded_model_is_not_healthy(monkeypatch):
    monkeypatch.setattr(serve.app.state, "model", None, raising=False)
    # Without a context manager, TestClient does not run startup.
    client = TestClient(serve.app)
    try:
        assert client.get("/healthz").status_code == 503
    finally:
        client.close()


def test_s3_startup_loads_downloaded_model(saved_model, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "s3")
    monkeypatch.setenv("ARTIFACT_BUCKET", "test-income-bucket")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "downloaded.joblib"))
    client = Mock()
    client.download_file.side_effect = lambda bucket, key, filename: Path(filename).write_bytes(saved_model.read_bytes())
    monkeypatch.setattr(serve.boto3, "client", Mock(return_value=client))
    with TestClient(serve.app) as api:
        assert api.get("/healthz").status_code == 200
    assert client.download_file.call_args.args[:2] == ("test-income-bucket", serve.MODEL_KEY)
    assert (tmp_path / "downloaded.joblib").read_bytes() == saved_model.read_bytes()


def test_s3_failure_preserves_previous_model(saved_model, monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "s3")
    monkeypatch.setenv("ARTIFACT_BUCKET", "test-income-bucket")
    monkeypatch.setenv("MODEL_PATH", str(saved_model))
    previous = saved_model.read_bytes()
    client = Mock()
    client.download_file.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
    )
    monkeypatch.setattr(serve.boto3, "client", Mock(return_value=client))
    with pytest.raises(ClientError):
        serve.load_model()
    assert saved_model.read_bytes() == previous
    assert not list(saved_model.parent.glob("*.part"))


def test_missing_bucket_fails_without_cloud_request(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "s3")
    monkeypatch.delenv("ARTIFACT_BUCKET", raising=False)
    client_factory = Mock()
    monkeypatch.setattr(serve.boto3, "client", client_factory)
    with pytest.raises(ValueError, match="ARTIFACT_BUCKET"):
        serve.load_model()
    client_factory.assert_not_called()
