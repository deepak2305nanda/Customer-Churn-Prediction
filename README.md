# Customer Churn Prediction + Retention ROI Simulator

End-to-end data science project: predict churn, **calibrate probabilities**, and choose **who to target** using profit-based thresholding. Includes a Streamlit app to simulate retention offers and expected ROI.

## Business problem

Telecom churn is expensive: acquiring new customers costs more than retaining existing ones. The retention team usually has **limited capacity/budget**, so the business question isn’t just “who might churn?” — it’s:

- **Who should we contact (and offer a discount) to maximize profit?**
- **What threshold / targeting policy should we use?**

## How the problem was identified (in practice)

Typical signals that trigger this project in a real company:

- **Rising churn rate** (monthly/quarterly) and revenue leakage
- **Low ROI retention campaigns** (contacting too many low-risk customers)
- **No consistent targeting rule** (different teams using different heuristics)

In a production setting, you’d define churn with the business (e.g., “no active service after 30 days”), align on the **retention action** (call/email/discount), and agree on a **success metric** (expected profit, churn reduction at fixed budget, etc.).

## Business framing

Retention teams usually can’t contact everyone. This project turns churn predictions into an action:

- Predict \(P(\text{churn})\) for each customer
- Model the economics of an intervention (contact + discount)
- Target customers only when the **expected value is positive**, and pick a threshold that maximizes profit

## Solution formulation

This repo implements the end-to-end workflow:

- **Data**: Telco Customer Churn (IBM sample dataset)
- **Model**: Logistic Regression with `class_weight="balanced"` for imbalance handling
- **Calibration**: `CalibratedClassifierCV(method="sigmoid")` to make probabilities usable for ROI decisions
- **Decisioning**: choose a targeting threshold that maximizes expected profit; export a targeting list
- **Deployment**: Streamlit app to score customers and run “what-if” ROI simulations

## What’s inside

- **Reproducible pipeline**: dataset download → cleaning/feature encoding → train + calibration → evaluation + profit curves
- **Cost-sensitive decisioning**: choose targeting threshold that maximizes expected profit
- **Streamlit app**: upload a CSV (same schema) → score churn → simulate offer ROI and targeting

## Results (example run)

From the included evaluation run:

- **ROC-AUC**: 0.842  
- **PR-AUC**: 0.633  
- **Brier score** (calibration-sensitive): 0.138  
- **Test churn rate**: 26.5%

With default economics (editable in the app), the best profit threshold found was:

- **Best threshold**: 0.17  
- **Targeted customers**: 752 (out of the test set)  
- **Expected profit (test set)**: \$63,190.64  

See saved figures in `reports/figures/`:
- `profit_curve.png` (profit vs threshold)
- `pr_curve.png` (precision–recall)
- `score_hist.png` (score distribution)

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

## Key decisions made

- **Use calibrated probabilities**: profit-based targeting assumes \(p(\text{churn})\) is meaningful; calibration improves decision stability.
- **Optimize for business outcome** (expected profit), not just AUC: a model can have good AUC but still be a bad campaign decision-maker if probabilities/thresholding are poor.
- **Simple, explainable baseline** first (logistic regression): easier to communicate to stakeholders; can be upgraded later (GBMs, uplift models).

## What I learned / what this demonstrates

- Turning ML into an operational decision requires:
  - **explicit economics assumptions** (CLV, offer cost, success rate)
  - **calibration** and **thresholding**
  - an interface for business users to explore scenarios (the app)
- A “churn model” becomes much more portfolio-relevant when it answers:
  - **who to act on**
  - **why**
  - **what the expected impact is**

## Suggested analysis steps (for your notebook / report)

If you want to extend the project with deeper analysis, add notebooks like:

- **EDA**: churn rate by contract type, tenure buckets, monthly charges
- **Feature importance**: coefficients (logistic regression), permutation importance
- **Calibration plots**: reliability curve, Brier decomposition
- **Sensitivity**: vary CLV/offer cost/\(p(\text{save}|\text{target})\) to see how the decision threshold shifts
- **Next step (advanced)**: uplift modeling with randomized campaign data (causal targeting)

## Key outputs

- `reports/metrics.json`: model metrics (ROC-AUC, PR-AUC, Brier, etc.)
- `reports/profit_curve.csv`: profit vs threshold summary
- `models/`: saved pipeline + metadata

