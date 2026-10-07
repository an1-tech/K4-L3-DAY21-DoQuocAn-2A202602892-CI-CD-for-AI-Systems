import json
from unittest.mock import Mock

import pytest

from scripts import upload_model


@pytest.fixture
def publication_files(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test-model")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"f1_score": 0.71}), encoding="utf-8")
    return model, report


def test_publish_approved_model(publication_files, monkeypatch):
    model, report = publication_files
    client = Mock()
    monkeypatch.setattr(upload_model.boto3, "client", Mock(return_value=client))
    upload_model.publish_model("test-income-bucket", model, report)
    assert client.upload_file.call_count == 2
    assert client.upload_file.call_args_list[0].args == (
        str(model), "test-income-bucket", "artifacts/current/model.joblib"
    )
    assert client.upload_file.call_args_list[1].args == (
        str(report), "test-income-bucket", "artifacts/current/report.json"
    )


def test_rejected_model_never_contacts_aws(publication_files, monkeypatch):
    model, report = publication_files
    report.write_text(json.dumps({"f1_score": 0.60}), encoding="utf-8")
    client_factory = Mock()
    monkeypatch.setattr(upload_model.boto3, "client", client_factory)
    with pytest.raises(ValueError, match="FAILED"):
        upload_model.publish_model("test-income-bucket", model, report)
    client_factory.assert_not_called()
