from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ProfitConfig:
    # Interpretable defaults; app lets you change these.
    clv_usd: float = 1200.0  # value of retaining a customer
    offer_cost_usd: float = 50.0  # discount / incentive cost
    contact_cost_usd: float = 2.0  # call/email cost
    p_save_given_target: float = 0.25  # success rate if we target

