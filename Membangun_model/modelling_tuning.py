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
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


class TrainingDataError(ValueError):
    """Raised when processed training data is missing or invalid."""


@dataclass(frozen=True)
class TuningConfig:
    processed_dir: Path
    artifacts_dir: Path
    experiment_name: str
    n_iter: int
    cv: int
    max_rows: int | None


TARGET_COLUMN = "Class"


def parse_args() -> TuningConfig:
    parser = argparse.ArgumentParser(description="Train tuned fraud model.")
    parser.add_argument("--processed-dir", default="cc_preprocessing/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--experiment-name", default="credit-card-fraud-detection")
    parser.add_argument("--n-iter", default=8, type=int)
    parser.add_argument("--cv", default=3, type=int)
    parser.add_argument(
        "--max-rows",
        default=0,
        type=int,
        help="Optional cap for faster CI experiments. Use 0 for all rows.",
    )
    args = parser.parse_args()
    return TuningConfig(
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        experiment_name=args.experiment_name,
        n_iter=args.n_iter,
        cv=args.cv,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )


def load_dataset(processed_dir: Path, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise TrainingDataError(
            "Processed data not found. Run "
            "`python cc_preprocessing/preprocess.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if max_rows is not None and max_rows < len(train_df):
        train_df, _ = train_test_split(
            train_df,
            train_size=max_rows,
            random_state=42,
            stratify=train_df[TARGET_COLUMN],
        )

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return x_train, y_train, x_test, y_test


def build_search(config: TuningConfig) -> RandomizedSearchCV:
    estimator = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        scoring="average_precision",
        cv=config.cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )


def evaluate_model(
    model: RandomForestClassifier,
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
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path,
) -> Path:
    output_path = artifacts_dir / "tuned_confusion_matrix.png"
    display = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    display.ax_.set_title("Tuned Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_feature_importance(
    model: RandomForestClassifier,
    columns: list[str],
    artifacts_dir: Path,
) -> Path:
    output_path = artifacts_dir / "feature_importance.png"
    importance_df = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(20), x="importance", y="feature")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def export_production_artifacts(
    model: RandomForestClassifier,
    feature_columns: list[str],
    processed_dir: Path,
    artifacts_dir: Path,
) -> dict[str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    feature_columns_path = artifacts_dir / "feature_columns.json"
    scaler_source_path = processed_dir / "amount_time_scaler.joblib"
    scaler_target_path = artifacts_dir / "amount_time_scaler.joblib"

    joblib.dump(model, model_path)
    feature_columns_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    if not scaler_source_path.exists():
        raise TrainingDataError(
            "Scaler artifact missing. Re-run preprocessing before training tuned model."
        )
    scaler = joblib.load(scaler_source_path)
    joblib.dump(scaler, scaler_target_path)

    return {
        "model": model_path,
        "feature_columns": feature_columns_path,
        "scaler": scaler_target_path,
    }


def train_tuned_model(config: TuningConfig) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = load_dataset(config.processed_dir, config.max_rows)
    search = build_search(config)

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="tuned_random_forest") as run:
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, x_test, y_test)
        artifact_paths = export_production_artifacts(
            model=best_model,
            feature_columns=list(x_train.columns),
            processed_dir=config.processed_dir,
            artifacts_dir=config.artifacts_dir,
        )
        confusion_matrix_path = save_confusion_matrix(best_model, x_test, y_test, config.artifacts_dir)
        feature_importance_path = save_feature_importance(best_model, list(x_train.columns), config.artifacts_dir)

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "search_scoring": "average_precision",
                "n_iter": config.n_iter,
                "cv": config.cv,
                "max_rows": config.max_rows or "all",
                "feature_count": x_train.shape[1],
                **search.best_params_,
            }
        )
        mlflow.log_metric("best_cv_average_precision", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        for artifact_path in [
            *artifact_paths.values(),
            confusion_matrix_path,
            feature_importance_path,
        ]:
            mlflow.log_artifact(str(artifact_path))

        return {
            "run_id": run.info.run_id,
            "best_params": search.best_params_,
            "metrics": metrics,
            "artifacts": {key: str(value) for key, value in artifact_paths.items()},
        }


def main() -> None:
    config = parse_args()
    result = train_tuned_model(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
