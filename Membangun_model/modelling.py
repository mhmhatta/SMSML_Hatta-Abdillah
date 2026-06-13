from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tracking import configure_tracking


class TrainingDataError(ValueError):
    """Raised when processed training data is missing or invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    processed_dir: Path
    artifacts_dir: Path
    experiment_name: str
    max_iter: int
    use_dagshub: bool


TARGET_COLUMN = "Class"


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train baseline fraud model.")
    parser.add_argument(
        "--processed-dir",
        default="cc_preprocessing/processed",
        help="Directory containing processed train.csv and test.csv.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for local model artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        default="credit-card-fraud-detection",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--max-iter",
        default=1000,
        type=int,
        help="Maximum LogisticRegression iterations.",
    )
    parser.add_argument(
        "--dagshub",
        action="store_true",
        help="Initialize DagsHub MLflow tracking for mhmhatta/smsml.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        experiment_name=args.experiment_name,
        max_iter=args.max_iter,
        use_dagshub=args.dagshub,
    )


def load_dataset(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise TrainingDataError(
            "Processed data not found. Run "
            "`python cc_preprocessing/preprocess.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        raise TrainingDataError(f"Processed files must contain `{TARGET_COLUMN}` column.")

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return x_train, y_train, x_test, y_test


def evaluate_model(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)
    return {
        "average_precision": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def save_confusion_matrix(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "baseline_confusion_matrix.png"
    display = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    display.ax_.set_title("Baseline Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_feature_columns(columns: list[str], artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "feature_columns.json"
    output_path.write_text(json.dumps(columns, indent=2), encoding="utf-8")
    return output_path


def train_baseline(config: TrainingConfig) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = load_dataset(config.processed_dir)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=config.max_iter,
        n_jobs=-1,
        random_state=42,
    )

    configure_tracking(config.use_dagshub)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": config.max_iter,
                "feature_count": x_train.shape[1],
                "train_rows": x_train.shape[0],
                "test_rows": x_test.shape[0],
            }
        )
        mlflow.log_metrics(metrics)

        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = config.artifacts_dir / "baseline_model.joblib"
        joblib.dump(model, local_model_path)

        confusion_matrix_path = save_confusion_matrix(model, x_test, y_test, config.artifacts_dir)
        feature_columns_path = save_feature_columns(list(x_train.columns), config.artifacts_dir)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(local_model_path))
        mlflow.log_artifact(str(confusion_matrix_path))
        mlflow.log_artifact(str(feature_columns_path))

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_path": str(local_model_path),
        }


def main() -> None:
    config = parse_args()
    result = train_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
