"""
Orchestrator: generates complete synthetic dataset.

Flow: profiles → opening-balance txns → per-user baseline txns → persona overlays
      → balance recompute → parquet
"""

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data_generator.baseline import generate_baseline_transactions
from src.data_generator.personas import PERSONA_INJECTORS
from src.data_generator.profiles import generate_profiles
from src.utils.io import DATA_RAW, load_config, write_parquet


def _make_opening_txns(profile: dict, start_date: datetime, rng: np.random.Generator) -> list[dict]:
    """Create day-zero transactions that seed each investment account with its share of initial_aua."""
    aua = profile["initial_aua"]
    if aua <= 0:
        return []

    rrsp_frac = rng.uniform(0.3, 0.5)
    tfsa_frac = rng.uniform(0.2, 0.4)
    resp_frac = 0.0 if profile["age"] < 28 else rng.uniform(0.0, 0.15)
    non_reg_frac = max(0, 1 - rrsp_frac - tfsa_frac - resp_frac)

    ts = start_date - timedelta(hours=1)  # just before the window starts
    txns = []
    for acct, frac in [
        ("investment_rrsp", rrsp_frac),
        ("investment_tfsa", tfsa_frac),
        ("investment_resp", resp_frac),
        ("investment_non_reg", non_reg_frac),
    ]:
        amt = round(aua * frac, 2)
        if amt > 0:
            txns.append({
                "txn_id": str(uuid.uuid4()),
                "user_id": profile["user_id"],
                "timestamp": ts,
                "account_type": acct,
                "amount": amt,
                "merchant": "Opening Balance",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 0.0,
                "channel": "ach",
            })

    # Chequing opening balance
    chequing_bal = round(float(rng.uniform(2000, 15000)), 2)
    txns.append({
        "txn_id": str(uuid.uuid4()),
        "user_id": profile["user_id"],
        "timestamp": ts,
        "account_type": "chequing",
        "amount": chequing_bal,
        "merchant": "Opening Balance",
        "mcc": 6012,
        "mcc_category": "financial_services",
        "balance_after": 0.0,
        "channel": "ach",
    })

    return txns


def _recompute_balances(txns: list[dict]) -> list[dict]:
    """Recompute running balance_after per user per account_type."""
    txns.sort(key=lambda t: t["timestamp"])

    balances: dict[tuple[str, str], float] = {}
    for t in txns:
        key = (t["user_id"], t["account_type"])
        if key not in balances:
            balances[key] = 0.0
        balances[key] += t["amount"]
        t["balance_after"] = round(balances[key], 2)

    return txns


def generate_dataset(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full synthetic dataset. Returns (profiles_df, transactions_df)."""
    if config is None:
        config = load_config()

    gen_cfg = config["data_generation"]
    seed = gen_cfg["seed"]
    rng = np.random.default_rng(seed)

    start_date = datetime.fromisoformat(gen_cfg["start_date"])
    end_date = start_date + timedelta(days=gen_cfg["months"] * 30)

    print(f"Generating {gen_cfg['num_users']} user profiles...")
    profiles_df = generate_profiles(config)

    all_txns = []
    total = len(profiles_df)

    for idx, row in profiles_df.iterrows():
        profile = row.to_dict()
        user_rng = np.random.default_rng(seed + idx + 1)

        opening = _make_opening_txns(profile, start_date, user_rng)
        txns = opening + generate_baseline_transactions(profile, start_date, end_date, user_rng)

        persona = profile["persona"]
        if persona in PERSONA_INJECTORS and profile.get("signal_onset_date") is not None:
            injector = PERSONA_INJECTORS[persona]
            txns = injector.inject(txns, profile, profile["signal_onset_date"], end_date, user_rng)

        all_txns.extend(txns)

        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{total} users ({len(all_txns):,} txns so far)")

    print(f"Recomputing balances for {len(all_txns):,} transactions...")
    all_txns = _recompute_balances(all_txns)

    txns_df = pd.DataFrame(all_txns)
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
    txns_df = txns_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    print(f"Writing parquet files to {DATA_RAW}...")
    write_parquet(profiles_df, DATA_RAW / "user_profiles.parquet")
    write_parquet(txns_df, DATA_RAW / "transactions.parquet")

    print(f"Done: {len(profiles_df)} profiles, {len(txns_df):,} transactions")
    return profiles_df, txns_df


if __name__ == "__main__":
    generate_dataset()
