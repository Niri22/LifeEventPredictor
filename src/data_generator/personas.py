"""
Persona trajectory injectors.

Each persona class overlays behavioral shifts onto a user's transaction stream
after their signal_onset_date, simulating the spending patterns that the Signal
Engine should learn to detect.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np

from src.utils.mcc_codes import MCC_CATEGORIES


class PersonaInjector(ABC):
    """Base class for persona-specific transaction overlays."""

    @abstractmethod
    def inject(
        self,
        txns: list[dict],
        profile: dict,
        onset: datetime,
        end_date: datetime,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Return the modified transaction list with persona-specific overlays."""
        ...


class AspiringAffluentPersona(PersonaInjector):
    """
    'The Leapfrogger': After onset, the user aggressively saves toward RRSP.
    - Increase RRSP transfer amounts by 40-80%
    - Decrease discretionary credit card spending by 20-40%
    - Slight increase in financial services MCC (advisor visits, etc.)
    """

    def inject(self, txns, profile, onset, end_date, rng):
        months_active = max(1, (end_date - onset).days // 30)
        monthly_income = profile["annual_income"] / 12
        extra_rrsp_base = monthly_income * rng.uniform(0.08, 0.15)

        new_txns = []
        for month_offset in range(months_active):
            date = onset + timedelta(days=month_offset * 30 + 3)
            if date > end_date:
                break

            ramp = min(1.0, (month_offset + 1) / max(months_active * 0.6, 1))
            extra = round(extra_rrsp_base * ramp, 2)

            new_txns.append({
                "txn_id": str(uuid.uuid4()),
                "user_id": profile["user_id"],
                "timestamp": date,
                "account_type": "investment_rrsp",
                "amount": extra,
                "merchant": "Wealthsimple RRSP Top-Up",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 0.0,  # recalculated by orchestrator
                "channel": "internal_transfer",
            })

            if rng.random() < 0.4:
                new_txns.append({
                    "txn_id": str(uuid.uuid4()),
                    "user_id": profile["user_id"],
                    "timestamp": date + timedelta(days=int(rng.integers(1, 10))),
                    "account_type": "chequing",
                    "amount": -round(rng.uniform(150, 500), 2),
                    "merchant": rng.choice(["Financial Advisor", "Tax Consultant", "WS Premium Inquiry"]),
                    "mcc": 6012,
                    "mcc_category": "financial_services",
                    "balance_after": 0.0,
                    "channel": "online",
                })

        # Reduce discretionary spending by marking some credit card txns for removal
        reduced = _reduce_credit_spend(txns, onset, rng, reduction_pct=rng.uniform(0.2, 0.4))
        return reduced + new_txns


class StickyFamilyLeaderPersona(PersonaInjector):
    """
    'The Liquidity Watchdog': After onset, large transfers to illiquid Summit-like portfolio
    while credit card travel/dining spend spikes, creating a dangerous illiquidity gap.
    """

    def inject(self, txns, profile, onset, end_date, rng):
        months_active = max(1, (end_date - onset).days // 30)
        monthly_income = profile["annual_income"] / 12

        new_txns = []
        for month_offset in range(months_active):
            date = onset + timedelta(days=month_offset * 30 + 5)
            if date > end_date:
                break

            ramp = min(1.0, (month_offset + 1) / max(months_active * 0.5, 1))

            # Large transfer to illiquid investment (Summit-like)
            summit_amount = round(monthly_income * rng.uniform(0.15, 0.30) * ramp, 2)
            new_txns.append({
                "txn_id": str(uuid.uuid4()),
                "user_id": profile["user_id"],
                "timestamp": date,
                "account_type": "investment_non_reg",
                "amount": summit_amount,
                "merchant": "Wealthsimple Summit PE Fund",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 0.0,
                "channel": "internal_transfer",
            })

            # Spike credit card travel and dining
            n_luxury_txns = int(rng.integers(3, 8))
            for _ in range(n_luxury_txns):
                cat = rng.choice(["travel", "dining", "luxury"], p=[0.4, 0.35, 0.25])
                info = MCC_CATEGORIES[cat]
                amount = round(rng.uniform(*info["amount_range"]) * ramp, 2)
                new_txns.append({
                    "txn_id": str(uuid.uuid4()),
                    "user_id": profile["user_id"],
                    "timestamp": date + timedelta(days=int(rng.integers(0, 25))),
                    "account_type": "credit_card",
                    "amount": -amount,
                    "merchant": rng.choice(info["merchants"]),
                    "mcc": int(rng.choice(info["codes"])),
                    "mcc_category": cat,
                    "balance_after": 0.0,
                    "channel": "pos" if rng.random() < 0.5 else "online",
                })

        return txns + new_txns


class GenerationNerdPersona(PersonaInjector):
    """
    'The Analyst-in-Pocket': After onset, high-frequency trading activity,
    lump-sum deposits, and professional/conference spending patterns.
    """

    def inject(self, txns, profile, onset, end_date, rng):
        months_active = max(1, (end_date - onset).days // 30)

        new_txns = []
        for month_offset in range(months_active):
            base_date = onset + timedelta(days=month_offset * 30)
            if base_date > end_date:
                break

            # High-frequency trades (5-15 per month in non-reg)
            n_trades = int(rng.integers(5, 16))
            for i in range(n_trades):
                day_offset = int(rng.integers(0, 28))
                date = base_date + timedelta(days=day_offset)
                if date > end_date:
                    break
                amount = round(rng.uniform(500, 25000) * rng.choice([-1, 1]), 2)
                new_txns.append({
                    "txn_id": str(uuid.uuid4()),
                    "user_id": profile["user_id"],
                    "timestamp": date,
                    "account_type": "investment_non_reg",
                    "amount": amount,
                    "merchant": rng.choice(["Wealthsimple Trade", "Direct Index Rebalance", "TLH Sell", "TLH Buy"]),
                    "mcc": 6012,
                    "mcc_category": "financial_services",
                    "balance_after": 0.0,
                    "channel": "online",
                })

            # Occasional lump-sum deposit
            if rng.random() < 0.3:
                lump = round(rng.uniform(10000, 100000), 2)
                new_txns.append({
                    "txn_id": str(uuid.uuid4()),
                    "user_id": profile["user_id"],
                    "timestamp": base_date + timedelta(days=int(rng.integers(1, 15))),
                    "account_type": "investment_non_reg",
                    "amount": lump,
                    "merchant": "Wealthsimple Deposit",
                    "mcc": 6012,
                    "mcc_category": "financial_services",
                    "balance_after": 0.0,
                    "channel": "ach",
                })

            # Professional/SaaS/conference spending
            n_pro = int(rng.integers(2, 6))
            for _ in range(n_pro):
                cat = rng.choice(["conferences_saas", "professional_services", "education"])
                info = MCC_CATEGORIES[cat]
                new_txns.append({
                    "txn_id": str(uuid.uuid4()),
                    "user_id": profile["user_id"],
                    "timestamp": base_date + timedelta(days=int(rng.integers(0, 28))),
                    "account_type": "credit_card",
                    "amount": -round(rng.uniform(*info["amount_range"]), 2),
                    "merchant": rng.choice(info["merchants"]),
                    "mcc": int(rng.choice(info["codes"])),
                    "mcc_category": cat,
                    "balance_after": 0.0,
                    "channel": "online",
                })

        return txns + new_txns


PERSONA_INJECTORS: dict[str, PersonaInjector] = {
    "aspiring_affluent": AspiringAffluentPersona(),
    "sticky_family_leader": StickyFamilyLeaderPersona(),
    "generation_nerd": GenerationNerdPersona(),
}


def _reduce_credit_spend(
    txns: list[dict], onset: datetime, rng: np.random.Generator, reduction_pct: float
) -> list[dict]:
    """Remove a fraction of post-onset credit card discretionary transactions."""
    result = []
    for t in txns:
        if (
            t["account_type"] == "credit_card"
            and t["timestamp"] >= onset
            and t["amount"] < 0
            and rng.random() < reduction_pct
        ):
            continue
        result.append(t)
    return result
