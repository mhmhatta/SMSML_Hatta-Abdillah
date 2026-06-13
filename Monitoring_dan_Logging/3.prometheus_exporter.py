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


def update_model_ready_metric() -> None:
    required_files = [
        ARTIFACTS_DIR / "model.joblib",
        ARTIFACTS_DIR / "amount_time_scaler.joblib",
        ARTIFACTS_DIR / "feature_columns.json",
    ]
    MODEL_READY.set(1 if all(path.exists() for path in required_files) else 0)


def main() -> None:
    start_http_server(8001)
    update_model_ready_metric()
    print("Prometheus exporter running on port 8001")
    while True:
        update_model_ready_metric()
        time.sleep(10)


if __name__ == "__main__":
    main()
