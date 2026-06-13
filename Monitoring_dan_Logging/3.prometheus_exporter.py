from __future__ import annotations

import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "Membangun_model" / "artifacts"

MODEL_READY = Gauge(
    "fraud_model_artifacts_ready",
    "Whether the production model artifact bundle is available.",
)
ARTIFACT_COUNT = Gauge(
    "fraud_model_artifacts_count",
    "Number of required production model artifacts currently available.",
)
MODEL_FILE_SIZE_BYTES = Gauge(
    "fraud_model_file_size_bytes",
    "Size of the production model artifact in bytes.",
)
SCALER_FILE_SIZE_BYTES = Gauge(
    "fraud_scaler_file_size_bytes",
    "Size of the production scaler artifact in bytes.",
)
FEATURE_COLUMNS_FILE_SIZE_BYTES = Gauge(
    "fraud_feature_columns_file_size_bytes",
    "Size of the production feature-columns artifact in bytes.",
)


def update_model_ready_metric() -> None:
    required_files = [
        ARTIFACTS_DIR / "model.joblib",
        ARTIFACTS_DIR / "amount_time_scaler.joblib",
        ARTIFACTS_DIR / "feature_columns.json",
    ]
    available_files = [path for path in required_files if path.exists()]
    MODEL_READY.set(1 if len(available_files) == len(required_files) else 0)
    ARTIFACT_COUNT.set(len(available_files))
    MODEL_FILE_SIZE_BYTES.set((ARTIFACTS_DIR / "model.joblib").stat().st_size if (ARTIFACTS_DIR / "model.joblib").exists() else 0)
    SCALER_FILE_SIZE_BYTES.set((ARTIFACTS_DIR / "amount_time_scaler.joblib").stat().st_size if (ARTIFACTS_DIR / "amount_time_scaler.joblib").exists() else 0)
    FEATURE_COLUMNS_FILE_SIZE_BYTES.set((ARTIFACTS_DIR / "feature_columns.json").stat().st_size if (ARTIFACTS_DIR / "feature_columns.json").exists() else 0)


def main() -> None:
    start_http_server(8001)
    update_model_ready_metric()
    print("Prometheus exporter running on port 8001")
    while True:
        update_model_ready_metric()
        time.sleep(10)


if __name__ == "__main__":
    main()
