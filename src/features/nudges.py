"""
Persona-Based Nudge Engine.

Generates tailored rationale strings for the curator UI based on the
detected persona and signal type. Each nudge cites both a "Behavioral Reason"
(from transaction patterns) and is structured to pair with a "Macro Reason"
(from the macro agent).
"""

from __future__ import annotations

PERSONA_NUDGE_TEMPLATES: dict[str, dict[str, str]] = {
    "aspiring_affluent": {
        "leapfrog_ready": (
            "**Status Upgrade:** This client is {gap_pct} away from Premium status. "
            "A Retirement Accelerator (RRSP Loan) of ${suggested_amount:,.0f} would fill "
            "{rrsp_fill_pct} of unused RRSP room and immediately unlock fee reductions "
            "and Private Market access. Their savings rate of {savings_rate:.0%} suggests "
            "strong repayment capacity."
        ),
    },
    "sticky_family_leader": {
        "liquidity_warning": (
            "**Liquidity Safeguard:** This client's illiquidity ratio is {illiquidity_ratio:.0%} -- "
            "credit card spend is {credit_vs_invest:.1f}x their investment transfers. "
            "Recommend pairing Summit Portfolio access with the WS Credit Card's cash-back "
            "buffer to prevent cash-lock. Institutional PE diversification adds "
            "non-correlated returns to their portfolio."
        ),
    },
    "generation_nerd": {
        "harvest_opportunity": (
            "**Projected Alpha & Tax Efficiency:** With {txn_count} monthly trades "
            "and AUA of ${aua:,.0f}, this client is actively managing their portfolio. "
            "Direct Indexing can capture tax-loss harvesting across {num_positions}+ positions. "
            "AI Research summaries on their top holdings save ~2hrs/week of analysis time."
        ),
    },
}

# Shorter fallback nudge
_DEFAULT_NUDGE = (
    "Behavioral signals suggest this client may benefit from {product_name}. "
    "Review the audit log for contributing features."
)


def generate_nudge(
    persona_tier: str,
    signal: str,
    features: dict,
    product_name: str = "",
) -> str:
    """
    Generate a persona-tailored rationale string.

    Args:
        persona_tier: e.g. "aspiring_affluent"
        signal: e.g. "leapfrog_ready"
        features: dict of feature values (from the hypothesis audit log / raw features)
        product_name: display name of the recommended product

    Returns:
        Formatted nudge string ready for the UI.
    """
    templates = PERSONA_NUDGE_TEMPLATES.get(persona_tier, {})
    template = templates.get(signal)

    if template is None:
        return _DEFAULT_NUDGE.format(product_name=product_name or "the recommended product")

    aua = features.get("aua_current", 0)

    format_vars = {
        "gap_pct": f"{max(0, (100_000 - aua) / 100_000):.0%}" if persona_tier == "aspiring_affluent" else "N/A",
        "suggested_amount": features.get("suggested_amount", 25000),
        "rrsp_fill_pct": f"{features.get('rrsp_utilization', 0):.0%}",
        "savings_rate": features.get("savings_rate", 0),
        "illiquidity_ratio": features.get("illiquidity_ratio", 0),
        "credit_vs_invest": features.get("credit_spend_vs_invest", 0),
        "txn_count": int(features.get("txn_count_30d", 0)),
        "aua": aua,
        "num_positions": max(int(features.get("txn_count_30d", 10) / 3), 5),
        "product_name": product_name,
    }

    try:
        return template.format(**format_vars)
    except (KeyError, ValueError):
        return _DEFAULT_NUDGE.format(product_name=product_name or "the recommended product")


def generate_composite_reason(
    behavioral_nudge: str,
    macro_reasons: list[str],
    feedback_reason: str | None = None,
) -> str:
    """
    Combine behavioral nudge, macro reasons, and feedback loop reason
    into a single traceable rationale block.
    """
    parts = [f"**Behavioral Reason:**\n{behavioral_nudge}"]

    if macro_reasons:
        macro_text = "\n".join(f"- {r}" for r in macro_reasons)
        parts.append(f"**Macro Reason:**\n{macro_text}")

    if feedback_reason:
        parts.append(f"**Active Learning:**\n{feedback_reason}")

    return "\n\n".join(parts)
