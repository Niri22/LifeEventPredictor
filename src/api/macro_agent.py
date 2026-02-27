"""
Macro-Context Agent -- external economic signals.

Provides (mocked) Bank of Canada prime rate and market volatility (VIX/TSX).
Adjusts recommendation confidence based on macro conditions to ensure
recommendations are contextually appropriate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Default mock values reflecting a realistic 2026 Canadian macro environment
_DEFAULT_BOC_RATE = 4.25
_DEFAULT_VIX = 18.5
_DEFAULT_TSX_VOL = 14.2


@dataclass
class MacroSnapshot:
    """Point-in-time macro-economic data."""
    boc_prime_rate: float = _DEFAULT_BOC_RATE
    vix: float = _DEFAULT_VIX
    tsx_volatility: float = _DEFAULT_TSX_VOL
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def rates_high(self) -> bool:
        return self.boc_prime_rate > 5.0

    @property
    def market_volatile(self) -> bool:
        return self.vix > 25.0

    def to_dict(self) -> dict:
        return {
            "boc_prime_rate": self.boc_prime_rate,
            "vix": self.vix,
            "tsx_volatility": self.tsx_volatility,
            "timestamp": self.timestamp,
            "rates_high": self.rates_high,
            "market_volatile": self.market_volatile,
        }


def fetch_macro_snapshot(randomize: bool = False) -> MacroSnapshot:
    """
    Return current macro data. In production this would call BoC / market APIs.
    With randomize=True, adds small jitter for demo realism.
    """
    if randomize:
        return MacroSnapshot(
            boc_prime_rate=round(_DEFAULT_BOC_RATE + random.uniform(-0.5, 0.5), 2),
            vix=round(_DEFAULT_VIX + random.uniform(-5, 8), 1),
            tsx_volatility=round(_DEFAULT_TSX_VOL + random.uniform(-3, 5), 1),
        )
    return MacroSnapshot()


def adjust_confidence_for_macro(
    confidence: float,
    persona_tier: str,
    product_code: str,
    macro: MacroSnapshot,
    tax_refund_offset_ratio: float = 1.0,
) -> tuple[float, list[str]]:
    """
    Adjust a hypothesis confidence score based on macro conditions.

    Returns (adjusted_confidence, list_of_macro_reasons).

    Rules:
    - RRSP Loan (Retirement Accelerator): If BoC rate > 5%, penalize unless
      tax refund offset > 1.2x interest cost.
    - Summit Portfolio: If VIX > 25, reduce confidence (volatile market
      makes illiquid PE riskier).
    """
    adj = confidence
    reasons: list[str] = []

    # RRSP Loan macro adjustment
    if product_code == "RRSP_LOAN" and macro.rates_high:
        if tax_refund_offset_ratio < 1.2:
            penalty = 0.10
            adj = max(adj - penalty, 0.0)
            reasons.append(
                f"BoC prime rate {macro.boc_prime_rate}% is elevated; "
                f"tax refund offset ({tax_refund_offset_ratio:.2f}x) "
                f"does not exceed 1.2x interest cost. Confidence reduced by {penalty:.0%}."
            )
        else:
            reasons.append(
                f"BoC prime rate {macro.boc_prime_rate}% is elevated, but tax refund offset "
                f"({tax_refund_offset_ratio:.2f}x) exceeds 1.2x threshold -- no penalty."
            )

    # Summit Portfolio macro adjustment
    if product_code == "SUMMIT_PORTFOLIO" and macro.market_volatile:
        penalty = min(0.15, (macro.vix - 25) * 0.01)
        adj = max(adj - penalty, 0.0)
        reasons.append(
            f"VIX at {macro.vix} indicates elevated volatility; "
            f"illiquid PE exposure is riskier. Confidence reduced by {penalty:.2%}."
        )

    # AI Research -- slight boost during volatile markets (more need for analysis)
    if product_code == "AI_RESEARCH_DIRECT_INDEX" and macro.market_volatile:
        boost = 0.03
        adj = min(adj + boost, 1.0)
        reasons.append(
            f"Elevated market volatility (VIX {macro.vix}) increases demand for "
            f"AI research and tax-loss harvesting. Confidence boosted by {boost:.0%}."
        )

    if not reasons:
        reasons.append("No macro adjustments applied -- conditions within normal range.")

    return round(adj, 4), reasons
