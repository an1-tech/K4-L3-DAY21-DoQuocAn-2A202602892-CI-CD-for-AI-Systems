import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

F1_THRESHOLD = 0.65

FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    # Đọc dữ liệu và giữ đúng thứ tự đặc trưng.
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train[FEATURE_NAMES]
    y_train = df_train["target"]
    X_eval = df_eval[FEATURE_NAMES]
    y_eval = df_eval["target"]

    # Cấu hình nơi lưu thí nghiệm.
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )

    artifact_root = Path(
        os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")
    ).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    experiment_name = "day21-income"
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    run_name = (
        f"trees-{params['n_estimators']}"
        f"_lr-{params['learning_rate']}"
        f"_depth-{params['max_depth']}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)

        model = GradientBoostingClassifier(
            **params,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)

        # Tính F1 riêng cho lớp dương: thu nhập >50K.
        f1 = float(f1_score(y_eval, predictions, zero_division=0))
        accuracy = float(accuracy_score(y_eval, predictions))

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        Path("outputs").mkdir(exist_ok=True)
        report = {
            "f1_score": f1,
            "accuracy": accuracy,
            "train_rows": int(len(df_train)),
            "eval_rows": int(len(df_eval)),
            "params": params,
            "run_id": run.info.run_id,
        }

        with open("outputs/report.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        print(f"Train rows: {len(df_train)}")
        print(f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
        print(f"MLflow run: {run.info.run_id}")

    return f1


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    train(params)