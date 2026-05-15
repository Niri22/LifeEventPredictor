"""
Wealthsimple Pulse onboarding flow — System Overview modal tour.

Explains Control Center, Decision Console, Growth Engine, personas, and governance tiers.
Triggered on first visit to Control Center or via "Start Tour" in the sidebar.
"""

import streamlit as st
import plotly.graph_objects as go

ONBOARDING_STEPS = [
    {
        "title": "Welcome",
        "body": """
Pulse continuously monitors each persona for upgrade, risk, and opportunity signals — routing interventions through governance before execution.

You, as the Financial Curator, make the final approve or reject decision for each intervention. Human-in-the-loop preserves suitability and the Privacy-to-Value balance.
        """,
    },
    {
        "title": "Problem Framing",
        "slide": True,
        "body": "",  # Rendered by _render_problem_framing_slide()
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

<span style="color:#2e7d32;font-weight:600;">Green</span> (confidence > 0.9): Low friction — silent approval or low-friction notification eligible.

<span style="color:#f57c00;font-weight:600;">Amber</span> (0.7–0.9): Batch review — cohort and approve in bulk where appropriate.

<span style="color:#c62828;font-weight:600;">Red</span> (< 0.7 or high-risk product): 1-to-1 human review required (e.g. illiquid allocation > 20% of AUA).
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


def _make_scaling_gap_chart():
    """Synthetic time series: transactions (steep up), product complexity (up), advisor capacity (flat)."""
    import random
    random.seed(42)
    n = 24  # months
    t = list(range(1, n + 1))
    transactions = [20 + 2.2 * i + 0.15 * i**2 + random.gauss(0, 2) for i in range(n)]
    product_complexity = [30 + 1.8 * i + random.gauss(0, 1.5) for i in range(n)]
    advisor_capacity = [55 + 0.3 * i + random.gauss(0, 2) for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=transactions, name="Transactions per client", line=dict(color="#1e40af", width=2)))
    fig.add_trace(go.Scatter(x=t, y=product_complexity, name="Product complexity", line=dict(color="#b45309", width=2)))
    fig.add_trace(go.Scatter(x=t, y=advisor_capacity, name="Advisor capacity", line=dict(color="#4b5563", width=2)))
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Months", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title="Index (normalized)", gridcolor="#e5e7eb", zeroline=False),
        hovermode="x unified",
        height=280,
    )
    return fig


def _render_problem_framing_slide():
    """Full-width slide: The Scaling Gap — chart left, flow diagram right, takeaway bottom."""
    st.markdown(
        """
        <div class="onboard-slide">
            <h2 class="onboard-slide-title">The Scaling Gap</h2>
            <p class="onboard-slide-subtitle">Client data is growing faster than manual review can keep up.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns([3, 2])
    with col_left:
        fig = _make_scaling_gap_chart()
        st.plotly_chart(fig, use_container_width=True, key="onboard_scaling_chart")
    with col_right:
        st.markdown(
            """
            <div class="onboard-flow">
                <div class="onboard-flow-step onboard-flow-step--muted">
                    <span class="onboard-flow-num">1</span>
                    <span class="onboard-flow-label">Client data</span>
                </div>
                <div class="onboard-flow-connector" aria-hidden="true">
                    <span class="onboard-flow-connector-line"></span>
                </div>
                <div class="onboard-flow-step onboard-flow-bottleneck">
                    <span class="onboard-flow-num">2</span>
                    <span class="onboard-flow-label">Manual review bottleneck</span>
                </div>
                <div class="onboard-flow-connector" aria-hidden="true">
                    <span class="onboard-flow-connector-line"></span>
                </div>
                <div class="onboard-flow-step onboard-flow-step--muted">
                    <span class="onboard-flow-num">3</span>
                    <span class="onboard-flow-label">Missed risk &amp; missed growth</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="onboard-slide-takeaway">You can\'t scale by adding more manual scanning.</p>',
        unsafe_allow_html=True,
    )


def _inject_onboarding_styles(presenter_mode: bool):
    """Inject CSS for Problem Framing slide and optional presenter mode."""
    slide_css = """
        .onboard-slide { margin-bottom: 1.5rem; background: #fafafa; padding: 1.25rem 1.5rem; border-radius: 12px; }
        .onboard-slide-title { font-size: 1.75rem; font-weight: 700; color: #111; margin: 0 0 0.35rem 0; }
        .onboard-slide-subtitle { font-size: 1rem; color: #4b5563; margin: 0 0 0; line-height: 1.5; }
        .onboard-slide-takeaway { font-size: 1rem; font-weight: 600; color: #374151; margin: 1.25rem 0 0 0; padding-top: 1rem; border-top: 1px solid #e5e7eb; }
        .onboard-flow {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .onboard-flow-step {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.95rem;
            width: fit-content;
            max-width: 100%;
            box-sizing: border-box;
        }
        .onboard-flow-step--muted {
            color: #9ca3af;
        }
        .onboard-flow-step--muted .onboard-flow-num {
            background: #f3f4f6;
            color: #9ca3af;
        }
        .onboard-flow-step--muted .onboard-flow-label {
            color: #9ca3af;
            font-weight: 500;
        }
        .onboard-flow-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.75rem;
            height: 1.75rem;
            flex-shrink: 0;
            border-radius: 50%;
            background: #e5e7eb;
            color: #6b7280;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .onboard-flow-label {
            flex: 0 1 auto;
            line-height: 1.35;
        }
        .onboard-flow-connector {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            width: 1.75rem;
            height: 0.75rem;
            flex-shrink: 0;
        }
        .onboard-flow-connector-line {
            display: block;
            width: 2px;
            height: 100%;
            margin-left: calc(0.875rem - 1px);
            background: linear-gradient(180deg, #d1d5db 0%, #9ca3af 100%);
            border-radius: 1px;
        }
        .onboard-flow-bottleneck {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
        }
        .onboard-flow-bottleneck .onboard-flow-num { background: #f59e0b; color: #fff; }
        .onboard-flow-bottleneck .onboard-flow-label { font-weight: 700; color: #92400e; }
    """
    presenter_css = """
        section[data-testid="stSidebar"] { display: none !important; }
        [data-testid="stAppViewContainer"] { padding-left: 1rem !important; max-width: 100% !important; }
        [data-testid="block-container"] { padding: 1rem 2rem !important; max-width: 100% !important; }
    """ if presenter_mode else ""
    st.markdown(
        f"<style>{slide_css}{presenter_css}</style>",
        unsafe_allow_html=True,
    )


def show_onboarding_dialog():
    """Render onboarding in-page (no dialog) so Back/Next trigger full reruns and primary button styling works."""
    step = st.session_state.get("onboarding_step", 0)
    n_steps = len(ONBOARDING_STEPS)
    step = max(0, min(step, n_steps - 1))
    st.session_state["onboarding_step"] = step

    # Presenter mode: hide sidebar and reduce clutter
    presenter = st.session_state.get("onboarding_presenter_mode", False)
    presenter_mode = st.toggle("Presenter mode", value=presenter, key="onboard_presenter_toggle")
    if presenter_mode != presenter:
        st.session_state["onboarding_presenter_mode"] = presenter_mode
        st.rerun()
    _inject_onboarding_styles(st.session_state.get("onboarding_presenter_mode", False))

    st.markdown("---")
    st.markdown("## Wealthsimple Pulse — System Overview")
    st.caption("AI-native growth and governance across client personas.")
    st.markdown("---")

    data = ONBOARDING_STEPS[step]
    if data.get("slide"):
        st.markdown(f"### {data['title']}")
        _render_problem_framing_slide()
    else:
        st.markdown(f"### {data['title']}")
        st.markdown(data["body"].strip(), unsafe_allow_html=True)
    st.divider()

    col_back, col_spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if step > 0:
            if st.button("← Back", use_container_width=True, key="onboard_back"):
                st.session_state["onboarding_step"] = max(0, step - 1)
                st.rerun()
    with col_next:
        if step < n_steps - 1:
            if st.button("Next →", type="primary", use_container_width=True, key="onboard_next"):
                st.session_state["onboarding_step"] = min(step + 1, n_steps - 1)
                st.rerun()
        else:
            if st.button("Get Started", type="primary", use_container_width=True, key="onboard_done"):
                st.session_state["onboarding_completed"] = True
                if "onboarding_step" in st.session_state:
                    del st.session_state["onboarding_step"]
                # Next Control Center run shows workspace prep overlay before load_data().
                st.session_state["workspace_prep_state"] = "splash"
                st.rerun()
    st.markdown("---")


def should_show_onboarding() -> bool:
    """True if we should auto-show the onboarding (first visit)."""
    return not st.session_state.get("onboarding_completed", False)


def start_tour():
    """Reset onboarding state and show tour in-page (for 'Tour' button in sidebar)."""
    st.session_state["onboarding_completed"] = False
    st.session_state["onboarding_step"] = 0
    st.rerun()
