from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ProfitParams:
    clv_usd: float
    offer_cost_usd: float
    contact_cost_usd: float
    p_save_given_target: float


@dataclass(frozen=True)
class ProfitAtThreshold:
    threshold: float
    n_targeted: int
    expected_profit_total: float
    expected_profit_per_customer: float


def expected_profit_per_customer(p_churn: np.ndarray, is_targeted: np.ndarray, params: ProfitParams) -> np.ndarray:
    """
    Simple expected value model:
    - If targeted: pay contact + offer; if customer would churn, we 'save' them with probability p_save.
    - If not targeted: profit baseline is 0 (measure incremental value of a campaign).
    """
    p_churn = np.asarray(p_churn, dtype=float)
    is_targeted = np.asarray(is_targeted, dtype=bool)

    cost = params.contact_cost_usd + params.offer_cost_usd
    benefit = params.p_save_given_target * params.clv_usd

    profit_targeted = p_churn * benefit - cost
    profit = np.where(is_targeted, profit_targeted, 0.0)
    return profit


def implied_targeting_rule(params: ProfitParams) -> float:
    """
    If you assume your predicted churn probability is well-calibrated, the EV of targeting is:
      EV = p_churn * (p_save * CLV) - (contact + offer)
    Target when EV >= 0, which implies a theoretical threshold:
      p_churn >= (contact + offer) / (p_save * CLV)
    """
    denom = params.p_save_given_target * params.clv_usd
    if denom <= 0:
        return 1.0
    return float((params.contact_cost_usd + params.offer_cost_usd) / denom)


def profit_curve(p_churn: np.ndarray, params: ProfitParams, thresholds: np.ndarray | None = None) -> list[ProfitAtThreshold]:
    p_churn = np.asarray(p_churn, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    out: list[ProfitAtThreshold] = []
    n = int(p_churn.shape[0])
    for t in thresholds:
        targeted = p_churn >= float(t)
        profits = expected_profit_per_customer(p_churn, targeted, params)
        total = float(np.sum(profits))
        out.append(
            ProfitAtThreshold(
                threshold=float(t),
                n_targeted=int(np.sum(targeted)),
                expected_profit_total=total,
                expected_profit_per_customer=(total / n) if n else 0.0,
            )
        )
    return out


def best_threshold(p_churn: np.ndarray, params: ProfitParams) -> ProfitAtThreshold:
    curve = profit_curve(p_churn, params)
    return max(curve, key=lambda x: x.expected_profit_total)

