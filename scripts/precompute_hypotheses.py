"""
Pre-compute hypotheses once with default macro and write to data/processed/hypotheses.json.
Run from project root: python -m scripts.precompute_hypotheses

Use this for prototype mode so the UI only reads static results and does not run model inference on load.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import after path is set
from src.api.macro_agent import MacroSnapshot
from src.utils.io import DATA_PROCESSED, load_config

# Default macro for pre-compute (BoC 4.25%, VIX 18)
DEFAULT_BOC = 4.25
DEFAULT_VIX = 18.0

HYPOTHESES_JSON = DATA_PROCESSED / "hypotheses.json"


def _to_json_serializable(obj):
    """Convert nested dict/list and numpy types to JSON-serializable Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    print("Loading data and model...")
    from ui.lib import load_data, load_model, generate_hypotheses

    profiles, _txns, features = load_data()
    model = load_model()
    macro = MacroSnapshot(boc_prime_rate=DEFAULT_BOC, vix=DEFAULT_VIX)

    print("Running generate_hypotheses (one-time inference)...")
    hypotheses = generate_hypotheses(features, profiles, model, macro)
    print(f"Generated {len(hypotheses)} hypotheses.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    serializable = _to_json_serializable(hypotheses)
    with open(HYPOTHESES_JSON, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Wrote {HYPOTHESES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
