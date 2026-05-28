# Customer Churn Prediction + Retention ROI Simulator

End-to-end data science project: predict churn, **calibrate probabilities**, and choose **who to target** using profit-based thresholding. Includes a Streamlit app to simulate retention offers and expected ROI.

## Business framing

Retention teams usually can’t contact everyone. This project turns churn predictions into an action:

- Predict \(P(\text{churn})\) for each customer
- Model the economics of an intervention (contact + discount)
- Target customers only when the **expected value is positive**, and pick a threshold that maximizes profit

## What’s inside

- **Reproducible pipeline**: dataset download → cleaning/feature encoding → train + calibration → evaluation + profit curves
- **Cost-sensitive decisioning**: choose targeting threshold that maximizes expected profit
- **Streamlit app**: upload a CSV (same schema) → score churn → simulate offer ROI and targeting

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Download data and train model:

```bash
python -m src.data.download
python -m src.models.train
python -m src.models.evaluate
```

Run the app:

```bash
streamlit run app/Home.py
```

## How the ROI logic works

We use a simple expected value rule per customer:

\[
\text{EV(target)} = p(\text{churn}) \cdot (p(\text{save}|\text{target}) \cdot \text{CLV}) - (\text{contact} + \text{offer})
\]

Target if EV \(\ge 0\). The app also searches thresholds to maximize total expected profit across customers.

## Key outputs

- `reports/metrics.json`: model metrics (ROC-AUC, PR-AUC, Brier, etc.)
- `reports/profit_curve.csv`: profit vs threshold summary
- `models/`: saved pipeline + metadata

