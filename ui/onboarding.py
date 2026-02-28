"""
Wealthsimple Pulse onboarding flow — System Overview modal tour.

Explains Control Center, Decision Console, Growth Engine, personas, and governance tiers.
Triggered on first visit to Control Center or via "Start Tour" in the sidebar.
"""

import streamlit as st

ONBOARDING_STEPS = [
    {
        "title": "Welcome",
        "body": """
Pulse continuously monitors each persona for upgrade, risk, and opportunity signals — routing interventions through governance before execution.

You, as the Financial Curator, make the final approve or reject decision for each intervention. Human-in-the-loop preserves suitability and the Privacy-to-Value balance.
        """,
    },
    {
        "title": "Three Surfaces",
        "body": """
**Control Center** (home): System health KPIs, strategic levers, and Top Actions Required.

**Decision Console**: Case-level review. Tiered queue (Red / Amber / Green); traceability and Approve / Reject / Escalate.

**Growth Engine**: Experiment uplift and model reliability — which pathways are winning, treatment vs control, model precision.
        """,
    },
    {
        "title": "Personas & Signals",
        "body": """
Pulse continuously monitors each persona for upgrade, risk, and opportunity signals — routing interventions through governance before execution.

**System flow:** Persona → Detected Signal → Governance → Product → Measured Impact

---

**Momentum Builder** ($50k–$100k AUA)

**Who they are**  
Pre-Premium clients with strong growth velocity.

**Pain point**  
High ambition, but friction crossing milestone tiers.

**Detected Signal**  
<span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;">Leapfrog Readiness</span>

**System Response**  
Stage RRSP Loan to accelerate Premium conversion.

**Example**  
$82k AUA + high savings velocity → loan-sized bridge proposed.

---

**Full-Stack Client** ($100k–$500k AUA)

**Who they are**  
Premium clients with multiple goals; core of the business.

**Pain point**  
Multi-account friction; risk of over-allocating to illiquid exposure.

**Detected Signal**  
<span style="background:#fff8e1;color:#f57c00;padding:2px 8px;border-radius:4px;">Liquidity Watchdog</span>

**System Response**  
Monitor allocation; suggest rebalance + visibility tools.

**Example**  
High transfers + rising credit spend → Suggest rebalance + WS Credit Card.

---

**Legacy Architect** ($500k+ AUA)

**Who they are**  
High-net-worth clients focused on long-term, multi-generational wealth.

**Pain point**  
Sophisticated but time-poor; want institutional-grade exposure without high friction.

**Detected Signal**  
<span style="background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:4px;">Analyst-in-Pocket</span>

**System Response**  
Tax-aware optimization and institutional-grade exposure.

**Example**  
Direct index + elevated volatility → Research summary or tax-loss harvest move.

---

These personas define how Pulse prioritizes growth while preserving suitability constraints.
        """,
    },
    {
        "title": "Governance Tiers",
        "body": """
Every intervention is assigned a tier before it reaches you:

**Green** (confidence > 0.9): Low friction — silent approval or low-friction notification eligible.

**Amber** (0.7–0.9): Batch review — cohort and approve in bulk where appropriate.

**Red** (< 0.7 or high-risk product): 1-to-1 human review required (e.g. illiquid allocation > 20% of AUA).
        """,
    },
    {
        "title": "You're Ready",
        "body": """
**Control Center** — what needs attention. **Decision Console** — review and decide. **Growth Engine** — experiment and model health.

Click **Get Started** to begin.
        """,
    },
]


@st.dialog("Wealthsimple Pulse — System Overview", width="large")
def show_onboarding_dialog():
    """Render the current onboarding step and Back/Next/Get Started buttons."""
    step = st.session_state.get("onboarding_step", 0)
    n_steps = len(ONBOARDING_STEPS)
    step = max(0, min(step, n_steps - 1))
    st.session_state["onboarding_step"] = step

    data = ONBOARDING_STEPS[step]
    st.markdown(f"### {data['title']}")
    st.caption("AI-native growth and governance across client personas.")
    st.markdown(data["body"].strip(), unsafe_allow_html=True)
    st.divider()

    col_back, col_spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if step > 0:
            if st.button("← Back", use_container_width=True, key="onboard_back"):
                st.session_state["onboarding_step"] = step - 1
                st.rerun()
    with col_next:
        if step < n_steps - 1:
            if st.button("Next →", type="primary", use_container_width=True, key="onboard_next"):
                st.session_state["onboarding_step"] = step + 1
                st.rerun()
        else:
            if st.button("Get Started", type="primary", use_container_width=True, key="onboard_done"):
                st.session_state["onboarding_completed"] = True
                if "onboarding_step" in st.session_state:
                    del st.session_state["onboarding_step"]
                st.rerun()


def should_show_onboarding() -> bool:
    """True if we should auto-show the onboarding dialog (first visit)."""
    return not st.session_state.get("onboarding_completed", False)


def start_tour():
    """Reset onboarding state and open the dialog (for 'Start Tour' button)."""
    st.session_state["onboarding_completed"] = False
    st.session_state["onboarding_step"] = 0
    show_onboarding_dialog()
