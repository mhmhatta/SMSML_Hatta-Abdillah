from __future__ import annotations

import os

import mlflow


DEFAULT_DAGSHUB_OWNER = "mhmhatta"
DEFAULT_DAGSHUB_REPO = "my-first-repo"


def configure_tracking(use_dagshub: bool) -> str:
    """Configure MLflow tracking and return the active tracking URI."""
    if use_dagshub:
        import dagshub

        dagshub.init(
            repo_owner=DEFAULT_DAGSHUB_OWNER,
            repo_name=DEFAULT_DAGSHUB_REPO,
            mlflow=True,
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    return mlflow.get_tracking_uri()
