from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetSchemaError(ValueError):
    """Raised when the raw fraud dataset does not match the expected schema."""

@dataclass(frozen=True)
class PreprocessConfig:
    raw_path: Path
    output_dir: Path
    test_size: float
    random_state: int

TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Time", "Amount"]
PCA_COLUMNS = [f"V{index}" for index in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess credit card fraud dataset.")
    parser.add_argument(
        "--raw-path",
        default="data_raw/creditcard.csv",
        help="Path to Kaggle creditcard.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="cc_preprocessing/processed",
        help="Directory for processed outputs.",
    )
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()
    return PreprocessConfig(
        raw_path=Path(args.raw_path),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
    )

def validate_schema(data: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise DatasetSchemaError(
            "Raw dataset schema is invalid. Missing columns: "
            f"{', '.join(missing_columns)}"
        )

    invalid_targets = sorted(set(data[TARGET_COLUMN].dropna().unique()) - {0, 1})
    if invalid_targets:
        raise DatasetSchemaError(
            f"`{TARGET_COLUMN}` must contain only 0/1 labels. Found: {invalid_targets}"
        )

def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Download Kaggle ULB Credit Card Fraud Detection "
            f"and place `creditcard.csv` at: {raw_path}"
        )
    data = pd.read_csv(raw_path)
    validate_schema(data)
    return data[REQUIRED_COLUMNS].copy()

def split_and_scale(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET_COLUMN],
    )
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[SCALE_COLUMNS] = scaler.fit_transform(train_df[SCALE_COLUMNS])
    test_df[SCALE_COLUMNS] = scaler.transform(test_df[SCALE_COLUMNS])
    return train_df, test_df, scaler

def save_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    config: PreprocessConfig,
) -> dict[str, str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.output_dir / "train.csv"
    test_path = config.output_dir / "test.csv"
    scaler_path = config.output_dir / "amount_time_scaler.joblib"
    metadata_path = config.output_dir / "metadata.json"
    sample_input_path = config.output_dir / "sample_input.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    raw_sample = test_df.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    sample_input_path.write_text(json.dumps(raw_sample, indent=2), encoding="utf-8")

    metadata = {
        "config": {
            **asdict(config),
            "raw_path": str(config.raw_path),
            "output_dir": str(config.output_dir),
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scale_columns": SCALE_COLUMNS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fraud_rate_train": float(train_df[TARGET_COLUMN].mean()),
        "fraud_rate_test": float(test_df[TARGET_COLUMN].mean()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "scaler_path": str(scaler_path),
        "metadata_path": str(metadata_path),
        "sample_input_path": str(sample_input_path),
    }

def run_preprocessing(config: PreprocessConfig) -> dict[str, str]:
    data = load_raw_dataset(config.raw_path)
    train_df, test_df, scaler = split_and_scale(
        data=data,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    return save_outputs(train_df, test_df, scaler, config)

def main() -> None:
    config = parse_args()
    outputs = run_preprocessing(config)
    print(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    main()
