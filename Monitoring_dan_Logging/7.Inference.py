from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field


class ArtifactLoadError(RuntimeError):
    """Raised when model artifacts required for serving are unavailable."""


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(..., description="Seconds elapsed from first transaction.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "Membangun_model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "amount_time_scaler.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

REQUEST_COUNTER = Counter(
    "fraud_api_prediction_requests_total",
    "Total prediction requests received by the fraud API.",
)
HEALTH_COUNTER = Counter(
    "fraud_api_health_requests_total",
    "Total health-check requests received by the fraud API.",
)
ERROR_COUNTER = Counter(
    "fraud_api_prediction_errors_total",
    "Total failed prediction requests.",
)
PREDICTION_COUNTER = Counter(
    "fraud_api_predictions_total",
    "Prediction count by predicted class.",
    ["label"],
)
TRANSACTION_AMOUNT = Histogram(
    "fraud_api_transaction_amount",
    "Distribution of incoming transaction amounts.",
    buckets=(0, 1, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)
FRAUD_PROBABILITY = Histogram(
    "fraud_api_fraud_probability",
    "Distribution of predicted fraud probabilities.",
    buckets=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
REQUEST_LATENCY = Histogram(
    "fraud_api_prediction_latency_seconds",
    "Prediction request latency in seconds.",
)

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


def load_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        raise ArtifactLoadError(f"Feature column artifact not found: {path}")
    return [str(column) for column in json.loads(path.read_text(encoding="utf-8"))]


def load_artifacts() -> tuple[Any, Any, list[str]]:
    missing_paths = [
        path for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH] if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise ArtifactLoadError(
            "Serving artifacts are missing. Run `python modelling_tuning.py` first. "
            f"Missing: {missing}"
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


MODEL, SCALER, FEATURE_COLUMNS = load_artifacts()


def build_feature_frame(payload: TransactionRequest) -> pd.DataFrame:
    frame = pd.DataFrame([payload.model_dump()])
    frame = frame[FEATURE_COLUMNS].copy()
    frame[["Time", "Amount"]] = SCALER.transform(frame[["Time", "Amount"]])
    return frame


@app.get("/health")
def health() -> dict[str, str]:
    HEALTH_COUNTER.inc()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionRequest) -> PredictionResponse:
    start_time = time.perf_counter()
    REQUEST_COUNTER.inc()
    try:
        TRANSACTION_AMOUNT.observe(payload.Amount)
        features = build_feature_frame(payload)
        fraud_probability = float(MODEL.predict_proba(features)[:, 1][0])
        prediction = int(fraud_probability >= 0.5)
        label = "fraud" if prediction == 1 else "legitimate"
        PREDICTION_COUNTER.labels(label=label).inc()
        FRAUD_PROBABILITY.observe(fraud_probability)
        return PredictionResponse(
            prediction=prediction,
            label=label,
            fraud_probability=fraud_probability,
        )
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
