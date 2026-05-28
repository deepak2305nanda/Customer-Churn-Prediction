from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    target: str


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_telco(df: pd.DataFrame) -> PreparedData:
    # Normalize obvious quirks in the common Telco churn dataset.
    df = df.copy()

    # TotalCharges sometimes comes as string with blanks.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Some datasets use "customerID" vs "customerID " etc.
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.rename(columns={col: "customerID"})
            break

    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Expected target column '{target}' in dataset. Found columns: {list(df.columns)}")

    # Standardize churn labels.
    churn_raw = df[target].astype(str).str.strip()
    df[target] = churn_raw.str.lower().map({"yes": 1, "no": 0})
    if df[target].isna().any():
        bad = sorted(set(churn_raw[df[target].isna()].unique()))
        raise ValueError(f"Unexpected Churn values after mapping: {bad}")

    # Drop ID from features if present (keep it for app display).
    if "customerID" in df.columns:
        df["customerID"] = df["customerID"].astype(str)

    # Simple missing handling: numeric -> median, categorical -> "Unknown"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")

    return PreparedData(df=df, target=target)


def save_processed(prepared: PreparedData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.df.to_parquet(out_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "telco_churn.csv"
    out_path = project_root / "data" / "processed" / "telco_churn.parquet"

    df = load_raw_csv(raw_path)
    prepared = prepare_telco(df)
    save_processed(prepared, out_path)
    print("preprocess OK", {"rows": int(prepared.df.shape[0]), "cols": int(prepared.df.shape[1]), "out": str(out_path)})


if __name__ == "__main__":
    main()

