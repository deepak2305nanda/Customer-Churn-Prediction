from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from src.models.config import ProfitConfig
from src.profit.targeting import ProfitParams, best_threshold, profit_curve


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    _ensure_dir(reports_dir)

    model = joblib.load(models_dir / "churn_model.joblib")

    X_test = pd.read_parquet(models_dir / "splits" / "X_test.parquet")
    y_test = pd.read_parquet(models_dir / "splits" / "y_test.parquet")["y"].astype(int).to_numpy()

    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
        "churn_rate_test": float(np.mean(y_test)),
    }

    # Profit curve with default business assumptions.
    pcfg = ProfitConfig()
    params = ProfitParams(
        clv_usd=pcfg.clv_usd,
        offer_cost_usd=pcfg.offer_cost_usd,
        contact_cost_usd=pcfg.contact_cost_usd,
        p_save_given_target=pcfg.p_save_given_target,
    )

    curve = profit_curve(proba, params=params)
    best = best_threshold(proba, params=params)

    reports_dir.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([asdict(x) for x in curve]).to_csv(reports_dir / "profit_curve.csv", index=False)
    reports_dir.joinpath("best_threshold.json").write_text(json.dumps(asdict(best), indent=2))

    # Save PR curve points for plotting in notebook/app if desired.
    prec, rec, thr = precision_recall_curve(y_test, proba)
    pr_df = pd.DataFrame({"precision": prec, "recall": rec, "threshold": np.append(thr, np.nan)})
    pr_df.to_csv(reports_dir / "pr_curve.csv", index=False)

    print("evaluate OK", {"metrics": metrics, "best_threshold": asdict(best)})


if __name__ == "__main__":
    main()

