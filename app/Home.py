from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profit.targeting import ProfitParams, best_threshold, expected_profit_per_customer, implied_targeting_rule  # noqa: E402


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "churn_model.joblib"
    return joblib.load(model_path)


def main() -> None:
    st.set_page_config(page_title="Churn ROI Simulator", layout="wide")
    st.title("Customer Churn Prediction + Retention ROI Simulator")

    st.markdown(
        "Upload Telco-style customer data, score churn probabilities, and choose a targeting threshold that **maximizes expected profit**."
    )

    with st.sidebar:
        st.header("Business assumptions")
        clv = st.number_input("Customer lifetime value (CLV) USD", min_value=0.0, value=1200.0, step=50.0)
        offer_cost = st.number_input("Offer cost USD", min_value=0.0, value=50.0, step=5.0)
        contact_cost = st.number_input("Contact cost USD", min_value=0.0, value=2.0, step=1.0)
        p_save = st.slider("P(save | targeted)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        params = ProfitParams(
            clv_usd=float(clv),
            offer_cost_usd=float(offer_cost),
            contact_cost_usd=float(contact_cost),
            p_save_given_target=float(p_save),
        )

    model = load_model()

    st.subheader("1) Upload data")
    uploaded = st.file_uploader("CSV with same columns as training data", type=["csv"])

    sample_path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
    if sample_path.exists():
        st.caption("Tip: you can test using the downloaded sample dataset.")

    if uploaded is None:
        st.info("Upload a CSV to begin. If you haven't trained yet, run: `python -m src.data.download` and `python -m src.models.train`.")
        return

    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]

    # Drop target if included.
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    customer_ids = df["customerID"] if "customerID" in df.columns else pd.Series([None] * len(df))
    X = df.drop(columns=["customerID"]) if "customerID" in df.columns else df

    proba = model.predict_proba(X)[:, 1]

    st.subheader("2) Optimize targeting threshold for profit")
    best = best_threshold(proba, params)
    theoretical = implied_targeting_rule(params)
    st.write(
        {
            "best_threshold": best.threshold,
            "theoretical_threshold": round(theoretical, 4),
            "n_targeted": best.n_targeted,
            "expected_profit_total": round(best.expected_profit_total, 2),
            "expected_profit_per_customer": round(best.expected_profit_per_customer, 4),
        }
    )

    st.subheader("3) Score and export a targeting list")
    threshold = st.slider("Use threshold", min_value=0.01, max_value=0.99, value=float(best.threshold), step=0.01)
    targeted = proba >= threshold
    profits = expected_profit_per_customer(proba, targeted, params)

    out = pd.DataFrame(
        {
            "customerID": customer_ids,
            "p_churn": proba,
            "target": targeted.astype(int),
            "expected_profit_usd": profits,
        }
    ).sort_values(["target", "p_churn"], ascending=[False, False])

    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "Download targeting CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="targeting_list.csv",
        mime="text/csv",
    )

    st.caption("Note: This is a simple expected-value simulator; in real life you’d estimate causal uplift with randomized experiments.")


if __name__ == "__main__":
    main()

