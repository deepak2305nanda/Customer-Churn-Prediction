import numpy as np

from src.profit.targeting import ProfitParams, best_threshold, expected_profit_per_customer


def test_expected_profit_shapes() -> None:
    p = np.array([0.1, 0.9])
    targeted = np.array([False, True])
    params = ProfitParams(clv_usd=1000.0, offer_cost_usd=50.0, contact_cost_usd=2.0, p_save_given_target=0.2)
    prof = expected_profit_per_customer(p, targeted, params)
    assert prof.shape == (2,)


def test_best_threshold_returns_valid() -> None:
    p = np.linspace(0.01, 0.99, 50)
    params = ProfitParams(clv_usd=1000.0, offer_cost_usd=50.0, contact_cost_usd=2.0, p_save_given_target=0.2)
    best = best_threshold(p, params)
    assert 0.0 < best.threshold < 1.0
    assert 0 <= best.n_targeted <= len(p)

