"""Income API: S3 in production, explicit local model mode for development."""

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import tempfile

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]
MODEL_KEY = "artifacts/current/model.joblib"


def load_model():
    source = os.environ.get("MODEL_SOURCE", "s3")
    if source not in {"s3", "local"}:
        raise ValueError("MODEL_SOURCE must be 's3' or 'local'")
    default_path = (
        Path("models/model.joblib")
        if source == "local"
        else Path.home() / "models" / "model.joblib"
    )
    model_path = Path(os.environ.get("MODEL_PATH", str(default_path))).expanduser()
    if source == "local":
        return joblib.load(model_path)
    bucket = os.environ.get("ARTIFACT_BUCKET")
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET is required when MODEL_SOURCE=s3")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("s3", region_name=region)
    # Validate the new download before replacing the previous local model.
    with tempfile.NamedTemporaryFile(
        dir=model_path.parent, suffix=".joblib.part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        client.download_file(bucket, MODEL_KEY, str(temporary_path))
        model = joblib.load(temporary_path)
        temporary_path.replace(model_path)
        return model
    finally:
        temporary_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = load_model()
    try:
        yield
    finally:
        application.state.model = None


app = FastAPI(title="Adult Income API", lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[float]


def ready_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return model


@app.get("/healthz")
def healthz(request: Request):
    ready_model(request)
    return {"status": "ok"}


@app.post("/score")
def score(payload: ScoreRequest, request: Request):
    if len(payload.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
    if not all(math.isfinite(value) for value in payload.features):
        raise HTTPException(status_code=400, detail="Features must be finite numbers")
    model = ready_model(request)
    inputs = pd.DataFrame([payload.features], columns=FEATURE_NAMES)
    prediction = int(model.predict(inputs)[0])
    return {
        "prediction": prediction,
        "label": "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap",
    }


def main():
    import uvicorn

    host = "127.0.0.1" if os.environ.get("MODEL_SOURCE") == "local" else "0.0.0.0"
    uvicorn.run(app, host=host, port=8080)


if __name__ == "__main__":
    main()
