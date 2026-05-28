from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.preprocess import PreparedData, load_raw_csv, prepare_telco
from src.models.config import TrainConfig


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: Path
    metadata_path: Path


def build_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


def train(prepared: PreparedData, cfg: TrainConfig, out_dir: Path) -> TrainArtifacts:
    df = prepared.df.copy()

    y = df[prepared.target].astype(int)
    X = df.drop(columns=[prepared.target])

    # Keep customerID out of the model features if present.
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pre = build_pipeline(X_train)
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = Pipeline(steps=[("preprocess", pre), ("model", base)])

    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "churn_model.joblib"
    metadata_path = out_dir / "metadata.json"

    joblib.dump(calibrated, model_path)

    metadata = {
        "target": prepared.target,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "test_size": cfg.test_size,
        "random_state": cfg.random_state,
        "note": "CalibratedClassifierCV(LogisticRegression(class_weight=balanced))",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Save split for evaluation reproducibility
    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    X_test.to_parquet(split_dir / "X_test.parquet", index=False)
    y_test.to_frame("y").to_parquet(split_dir / "y_test.parquet", index=False)

    return TrainArtifacts(model_path=model_path, metadata_path=metadata_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "telco_churn.csv"
    models_dir = project_root / "models"

    df = load_raw_csv(raw_path)
    prepared = prepare_telco(df)

    cfg = TrainConfig()
    artifacts = train(prepared, cfg, out_dir=models_dir)
    print("train OK", {"model": str(artifacts.model_path), "metadata": str(artifacts.metadata_path)})


if __name__ == "__main__":
    main()

