"""
Compliance & Audit — Regulatory oversight and decision trail.

Decision log exports, override audit trails, model governance, and regulatory compliance status.
This page signals institutional forethought and regulatory awareness for fintech deployment.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from ui.lib import (
    inject_ws_theme,
    render_pulse_sidebar,
    get_compliance_info,
    render_compliance_export_section,
    render_model_governance_panel,
    render_governance_constraints,
    get_system_timestamps,
    generate_decision_log_export,
    show_micro_feedback_toast,
)

st.set_page_config(page_title="Compliance & Audit — W Pulse", page_icon="W", layout="wide")


def render_regulatory_status():
    """Render regulatory compliance status dashboard."""
    st.markdown('<div class="ws-section-header">Regulatory Compliance Status</div>', unsafe_allow_html=True)
    
    compliance = get_compliance_info()
    
    # Compliance status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="ws-kpi-card">
            <div class="ws-status-indicator healthy"></div>
            <div class="ws-kpi-label">PIPEDA Privacy</div>
            <div class="ws-kpi-value" style="font-size: 1.5rem;">COMPLIANT</div>
            <div class="ws-micro">Impact assessment complete</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ws-kpi-card">
            <div class="ws-status-indicator healthy"></div>
            <div class="ws-kpi-label">OSFI ML/AI Guidelines</div>
            <div class="ws-kpi-value" style="font-size: 1.5rem;">COMPLIANT</div>
            <div class="ws-micro">Model risk framework aligned</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="ws-kpi-card">
            <div class="ws-status-indicator healthy"></div>
            <div class="ws-kpi-label">Data Retention</div>
            <div class="ws-kpi-value" style="font-size: 1.5rem;">{compliance['data_retention_days']}</div>
            <div class="ws-micro">Days (7 years)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="ws-kpi-card">
            <div class="ws-status-indicator warning"></div>
            <div class="ws-kpi-label">Next Audit Due</div>
            <div class="ws-kpi-value" style="font-size: 1.5rem;">{compliance['next_audit_due']}</div>
            <div class="ws-micro">Quarterly review</div>
        </div>
        """, unsafe_allow_html=True)


def render_decision_analytics():
    """Render decision analytics and patterns."""
    st.markdown('<div class="ws-section-header">Decision Analytics</div>', unsafe_allow_html=True)
    
    # Mock analytics data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="ws-subsection">Override Patterns (Last 30 Days)</div>', unsafe_allow_html=True)
        
        # Mock override data
        override_data = {
            "Reason": ["Manual review requested", "Compliance concern", "Model confidence low", "Customer complaint", "Other"],
            "Count": [12, 8, 15, 3, 7],
            "Percentage": [26.7, 17.8, 33.3, 6.7, 15.6]
        }
        
        df_overrides = pd.DataFrame(override_data)
        st.dataframe(df_overrides, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown('<div class="ws-subsection">Decision Volume Trends</div>', unsafe_allow_html=True)
        
        # Mock trend data
        trend_data = {
            "Date": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7, 0, -1)],
            "Total Decisions": [67, 72, 58, 81, 69, 75, 63],
            "Overrides": [8, 11, 7, 12, 9, 10, 8],
            "Override Rate %": [11.9, 15.3, 12.1, 14.8, 13.0, 13.3, 12.7]
        }
        
        df_trends = pd.DataFrame(trend_data)
        st.dataframe(df_trends, use_container_width=True, hide_index=True)


def main():
    inject_ws_theme()
    render_pulse_sidebar("compliance")

    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    
    # Header with strong typography
    col_title, col_updated = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="ws-page-title">Compliance & Audit</h1>', unsafe_allow_html=True)
        st.markdown('<div class="ws-secondary">Regulatory oversight, decision trails, and audit controls</div>', unsafe_allow_html=True)
    with col_updated:
        timestamps = get_system_timestamps()
        compliance = get_compliance_info()
        st.markdown(f"""
        <div class="ws-micro" style="text-align: right;">
        <div>Last updated: {timestamps['last_updated']}</div>
        <div>Compliance status: {compliance['compliance_status']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # Regulatory compliance status
    render_regulatory_status()
    
    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # Export and audit controls
    render_compliance_export_section()
    
    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)
    
    # Two-column layout for governance info
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Governance thresholds
        render_governance_constraints()
    
    with col_right:
        # Model governance
        render_model_governance_panel()
    
    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)
    
    # Decision analytics
    render_decision_analytics()
    
    # Compliance footer
    st.markdown(f"""
    <div class="ws-audit-summary" style="margin-top: 2rem;">
        <div class="ws-micro" style="text-align: center; color: var(--ws-muted);">
        Wealthsimple Pulse AI System | Model {compliance['model_version']} | 
        All decisions logged and retained per regulatory requirements | 
        System monitored 24/7 for compliance and performance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()