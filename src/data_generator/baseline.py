"""Generate baseline (noise) transactions for a user across all account types."""

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.mcc_codes import MCC_CATEGORIES, SPEND_WEIGHTS_BY_BRACKET


def _pick_mcc_and_merchant(
    category: str, rng: np.random.Generator
) -> tuple[int, str, str]:
    info = MCC_CATEGORIES[category]
    mcc = rng.choice(info["codes"])
    merchant = rng.choice(info["merchants"])
    return int(mcc), merchant, category


def _sample_amount(category: str, rng: np.random.Generator) -> float:
    lo, hi = MCC_CATEGORIES[category]["amount_range"]
    mu = np.log((lo + hi) / 2)
    sigma = 0.5
    return round(float(np.clip(rng.lognormal(mu, sigma), lo, hi * 1.2)), 2)


def generate_baseline_transactions(
    profile: dict,
    start_date: datetime,
    end_date: datetime,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate daily transactions for a single user across account types."""
    user_id = profile["user_id"]
    bracket = profile["income_bracket"]
    monthly_income = profile["annual_income"] / 12
    initial_aua = profile["initial_aua"]

    spend_weights = SPEND_WEIGHTS_BY_BRACKET.get(bracket, SPEND_WEIGHTS_BY_BRACKET["mid"])
    categories = list(spend_weights.keys())
    probs = np.array(list(spend_weights.values()))
    probs = probs / probs.sum()

    # Monthly savings rate: 10-30% of income goes to investments
    savings_rate = rng.uniform(0.10, 0.30)
    monthly_investment = monthly_income * savings_rate

    # Split investment across account types
    rrsp_frac = rng.uniform(0.3, 0.5)
    tfsa_frac = rng.uniform(0.2, 0.4)
    resp_frac = 0.0 if profile["age"] < 28 else rng.uniform(0.0, 0.15)
    non_reg_frac = max(0, 1 - rrsp_frac - tfsa_frac - resp_frac)

    txns = []
    balances = {
        "chequing": float(rng.uniform(2000, 15000)),
        "credit_card": 0.0,
        "investment_rrsp": initial_aua * rrsp_frac,
        "investment_tfsa": initial_aua * tfsa_frac,
        "investment_resp": initial_aua * resp_frac,
        "investment_non_reg": initial_aua * non_reg_frac,
    }

    current = start_date
    day_count = 0

    while current <= end_date:
        # Monthly payroll deposit (1st of month)
        if current.day == 1:
            balances["chequing"] += monthly_income
            txns.append(_make_txn(
                user_id, current, "chequing", monthly_income,
                "Employer Payroll", 6012, "financial_services", balances["chequing"], "ach",
            ))

            # Monthly investment transfers (2nd of month)
            for acct, frac in [
                ("investment_rrsp", rrsp_frac),
                ("investment_tfsa", tfsa_frac),
                ("investment_resp", resp_frac),
                ("investment_non_reg", non_reg_frac),
            ]:
                amt = round(monthly_investment * frac, 2)
                if amt > 0:
                    balances["chequing"] -= amt
                    balances[acct] += amt
                    txns.append(_make_txn(
                        user_id, current + timedelta(days=1), acct, amt,
                        "Wealthsimple Transfer", 6012, "financial_services",
                        balances[acct], "internal_transfer",
                    ))

            # Recurring bills (rent on 1st, utilities ~5th)
            rent = round(rng.uniform(1200, 3500) if bracket != "low" else rng.uniform(800, 1500), 2)
            balances["chequing"] -= rent
            txns.append(_make_txn(
                user_id, current, "chequing", -rent,
                "Landlord Payment", 6513, "rent_mortgage", balances["chequing"], "etransfer",
            ))

        # Daily discretionary spending
        n_txns = int(rng.poisson(3))
        for _ in range(n_txns):
            # 70% chequing debit, 30% credit card
            acct = "credit_card" if rng.random() < 0.30 else "chequing"
            cat = rng.choice(categories, p=probs)
            mcc, merchant, mcc_cat = _pick_mcc_and_merchant(cat, rng)
            amount = _sample_amount(cat, rng)

            balances[acct] -= amount if acct == "chequing" else 0
            if acct == "credit_card":
                balances[acct] -= amount  # credit balance goes negative

            channel = "pos" if rng.random() < 0.6 else "online"
            txns.append(_make_txn(
                user_id, current + timedelta(hours=int(rng.integers(8, 22)),
                                              minutes=int(rng.integers(0, 60))),
                acct, -amount, merchant, mcc, mcc_cat, balances[acct], channel,
            ))

        # Credit card payment (mid-month)
        if current.day == 15 and balances["credit_card"] < 0:
            payment = abs(balances["credit_card"])
            balances["chequing"] -= payment
            balances["credit_card"] = 0.0
            txns.append(_make_txn(
                user_id, current, "chequing", -payment,
                "CC Payment", 6012, "financial_services", balances["chequing"], "ach",
            ))

        current += timedelta(days=1)
        day_count += 1

    return txns


def _make_txn(
    user_id: str, ts: datetime, account_type: str, amount: float,
    merchant: str, mcc: int, mcc_category: str, balance_after: float, channel: str,
) -> dict:
    return {
        "txn_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": ts,
        "account_type": account_type,
        "amount": round(amount, 2),
        "merchant": merchant,
        "mcc": mcc,
        "mcc_category": mcc_category,
        "balance_after": round(balance_after, 2),
        "channel": channel,
    }
