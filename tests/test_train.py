import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train import FEATURE_NAMES, train


PARAMS = {
    "n_estimators": 10,
    "learning_rate": 0.1,
    "max_depth": 2,
}


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    # Test ghi vào thư mục tạm, giữ nguyên model và report thật.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow-test.db",
    )
    monkeypatch.setenv(
        "MLFLOW_ARTIFACT_ROOT",
        str(tmp_path / "mlartifacts"),
    )


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)

    X = rng.random((200, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=200)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "holdout.csv"

    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return str(train_path), str(eval_path)


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)

    result = train(
        PARAMS,
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_report_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)

    result = train(
        PARAMS,
        data_path=train_path,
        eval_path=eval_path,
    )

    report_path = Path("outputs/report.json")
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["f1_score"] == pytest.approx(result)
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["train_rows"] == 160
    assert report["eval_rows"] == 40


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)

    train(
        PARAMS,
        data_path=train_path,
        eval_path=eval_path,
    )

    model_path = Path("models/model.joblib")
    assert model_path.is_file()

    model = joblib.load(model_path)
    holdout = pd.read_csv(eval_path)
    predictions = model.predict(holdout[FEATURE_NAMES])

    assert len(predictions) == 40
    assert set(predictions.tolist()).issubset({0, 1})