from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_profit_curve(profit_curve_csv: Path, best_threshold_json: Path, out_path: Path) -> None:
    df = pd.read_csv(profit_curve_csv)
    best = json.loads(best_threshold_json.read_text())
    best_t = float(best["threshold"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["threshold"], df["expected_profit_total"], linewidth=2)
    ax.axvline(best_t, linestyle="--", linewidth=1.5)
    ax.set_title("Expected profit vs targeting threshold")
    ax.set_xlabel("Threshold (target if p_churn ≥ threshold)")
    ax.set_ylabel("Expected profit (USD)")
    ax.grid(True, alpha=0.25)
    ax.text(
        best_t,
        float(df["expected_profit_total"].max()) * 0.95,
        f"best={best_t:.2f}",
        ha="left",
        va="top",
    )
    _save(fig, out_path)


def plot_pr_curve(pr_curve_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(pr_curve_csv)
    df = df.dropna(subset=["precision", "recall"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["recall"], df["precision"], linewidth=2)
    ax.set_title("Precision–Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, out_path)


def plot_score_histogram(models_dir: Path, out_path: Path) -> None:
    """
    Uses saved test split and model to plot churn score distribution.
    (This helps visually debug calibration and threshold choice.)
    """
    import joblib  # local import to keep script lightweight

    model = joblib.load(models_dir / "churn_model.joblib")
    X_test = pd.read_parquet(models_dir / "splits" / "X_test.parquet")
    proba = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(proba, bins=30, alpha=0.9)
    ax.set_title("Churn score distribution (test split)")
    ax.set_xlabel("Predicted p(churn)")
    ax.set_ylabel("Customers")
    ax.grid(True, alpha=0.25, axis="y")
    _save(fig, out_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reports = project_root / "reports"
    models = project_root / "models"
    figs = reports / "figures"

    plot_profit_curve(
        profit_curve_csv=reports / "profit_curve.csv",
        best_threshold_json=reports / "best_threshold.json",
        out_path=figs / "profit_curve.png",
    )
    plot_pr_curve(pr_curve_csv=reports / "pr_curve.csv", out_path=figs / "pr_curve.png")
    plot_score_histogram(models_dir=models, out_path=figs / "score_hist.png")

    print("figures OK", {"out_dir": str(figs)})


if __name__ == "__main__":
    main()

