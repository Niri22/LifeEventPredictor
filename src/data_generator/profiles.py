"""Generate synthetic user profiles with wealth-tier attributes."""

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.utils.io import load_config

fake = Faker("en_CA")


def _income_bracket(income: float) -> str:
    if income < 65_000:
        return "low"
    elif income < 120_000:
        return "mid"
    elif income < 250_000:
        return "high"
    return "ultra"


def _sample_rrsp_room(age: int, income: float, rng: np.random.Generator) -> float:
    """Estimate unused RRSP room: roughly 18% of income * working years, minus random utilization."""
    years_working = max(age - 22, 1)
    total_room = 0.18 * income * years_working
    utilization = rng.uniform(0.05, 0.70)
    return round(max(total_room * (1 - utilization), 0), 2)


def generate_profiles(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    gen_cfg = config["data_generation"]
    rng = np.random.default_rng(gen_cfg["seed"])
    n = gen_cfg["num_users"]
    start = datetime.fromisoformat(gen_cfg["start_date"])
    months = gen_cfg["months"]
    onset_buffer = gen_cfg["signal_onset_buffer_months"]

    personas = list(gen_cfg["persona_weights"].keys())
    weights = list(gen_cfg["persona_weights"].values())

    records = []
    for _ in range(n):
        user_id = str(uuid.uuid4())
        age = int(rng.integers(22, 65))
        persona = rng.choice(personas, p=weights)

        # Income based on persona tier expectations
        aua_range = gen_cfg["aua_ranges"].get(persona, gen_cfg["aua_ranges"]["baseline"])
        if persona == "generation_nerd":
            income = rng.uniform(150_000, 500_000)
        elif persona == "sticky_family_leader":
            income = rng.uniform(100_000, 300_000)
        elif persona == "aspiring_affluent":
            income = rng.uniform(65_000, 180_000)
        else:
            income = rng.uniform(40_000, 120_000)

        initial_aua = rng.uniform(aua_range[0], aua_range[1])
        rrsp_room = _sample_rrsp_room(age, income, rng)

        # Signal onset: random date after the buffer period
        if persona != "baseline":
            onset_offset_days = int(rng.integers(
                onset_buffer * 30,
                max((months - 3) * 30, onset_buffer * 30 + 30),
            ))
            signal_onset_date = start + timedelta(days=onset_offset_days)
        else:
            signal_onset_date = None

        created_at = start - timedelta(days=int(rng.integers(180, 1800)))

        records.append({
            "user_id": user_id,
            "created_at": created_at,
            "age": age,
            "annual_income": round(income, 2),
            "income_bracket": _income_bracket(income),
            "province": fake.province_abbr(),
            "rrsp_room": round(rrsp_room, 2),
            "initial_aua": round(initial_aua, 2),
            "persona": persona,
            "signal_onset_date": signal_onset_date,
            "direct_deposit_at_ws": bool(rng.random() < 0.4),  # 40% have DD at WS for PoC
        })

    return pd.DataFrame(records)
