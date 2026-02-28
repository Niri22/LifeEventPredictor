"""Append-only Parquet logs for experiment assignments/outcomes; idempotent pathway metrics."""

from pathlib import Path

import pandas as pd

from src.utils.io import DATA_EXPERIMENTS, read_parquet, write_parquet

# Filenames
ASSIGNMENTS_FILE = "experiment_assignments.parquet"
OUTCOMES_FILE = "experiment_outcomes.parquet"
PATHWAY_METRICS_FILE = "pathway_metrics.parquet"

# Column order for assignments
ASSIGNMENT_COLUMNS = [
    "assignment_id",
    "hypothesis_id",
    "user_id",
    "persona_tier",
    "signal",
    "product_code",
    "governance_tier_at_assignment",
    "calibrated_confidence_at_assignment",
    "assignment",
    "created_at",
    "experiment_version",
]

# Column order for outcomes
OUTCOME_COLUMNS = [
    "outcome_id",
    "assignment_id",
    "hypothesis_id",
    "user_id",
    "observation_window_days",
    "converted",
    "delta_aua",
    "delta_liquidity_months",
    "retained",
    "complaint",
    "observed_at",
]


def _ensure_dir():
    DATA_EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    """Return DataFrame from path if exists, else empty with columns."""
    _ensure_dir()
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = read_parquet(path)
    # Ensure column order
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]


def read_assignments() -> pd.DataFrame:
    path = DATA_EXPERIMENTS / ASSIGNMENTS_FILE
    return _read_or_empty(path, ASSIGNMENT_COLUMNS)


def read_outcomes() -> pd.DataFrame:
    path = DATA_EXPERIMENTS / OUTCOMES_FILE
    return _read_or_empty(path, OUTCOME_COLUMNS)


def append_assignments(rows: list[dict]) -> None:
    """Append assignment records (atomic: read, concat, write)."""
    _ensure_dir()
    path = DATA_EXPERIMENTS / ASSIGNMENTS_FILE
    existing = read_assignments()
    new_df = pd.DataFrame(rows)
    for c in ASSIGNMENT_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[ASSIGNMENT_COLUMNS]
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    write_parquet(combined, path)


def append_outcomes(rows: list[dict]) -> None:
    """Append outcome records (atomic: read, concat, write)."""
    _ensure_dir()
    path = DATA_EXPERIMENTS / OUTCOMES_FILE
    existing = read_outcomes()
    new_df = pd.DataFrame(rows)
    for c in OUTCOME_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[OUTCOME_COLUMNS]
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    write_parquet(combined, path)


def read_pathway_metrics() -> pd.DataFrame:
    path = DATA_EXPERIMENTS / PATHWAY_METRICS_FILE
    _ensure_dir()
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def write_pathway_metrics(df: pd.DataFrame) -> None:
    _ensure_dir()
    write_parquet(df, DATA_EXPERIMENTS / PATHWAY_METRICS_FILE)
