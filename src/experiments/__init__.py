"""Experimentation layer: assign treatment/control, simulate outcomes, pathway metrics, reweighting."""

from src.experiments.assign import assign_to_experiment
from src.experiments.storage import (
    append_assignments,
    append_outcomes,
    read_assignments,
    read_outcomes,
    read_pathway_metrics,
    write_pathway_metrics,
)
from src.experiments.metrics import compute_pathway_metrics
from src.experiments.outcome_simulator import simulate_outcome
from src.experiments.reweight import apply_uplift_reweighting

__all__ = [
    "assign_to_experiment",
    "append_assignments",
    "append_outcomes",
    "read_assignments",
    "read_outcomes",
    "read_pathway_metrics",
    "write_pathway_metrics",
    "compute_pathway_metrics",
    "simulate_outcome",
    "apply_uplift_reweighting",
]
