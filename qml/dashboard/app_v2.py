"""
QML Trading Dashboard v2 - Arctic Pro Edition

A professional trading dashboard with:
- Arctic Pro theme (dark mode)
- Command bar with 6 KPIs (always visible)
- 7 tabbed navigation: Dashboard, Pattern Lab, Backtest, Analytics, Experiments, ML Training, Settings

Launch command:
    streamlit run qml/dashboard/app_v2.py

Author: QML Trading System
"""

import sys
from pathlib import Path

# Add dashboard directory to path for local imports
dashboard_dir = Path(__file__).parent
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))

# Add project root to path for src imports
project_root = dashboard_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="QML Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import theme and components (relative to dashboard dir)
from theme import get_css, ARCTIC_PRO, TYPOGRAPHY
from components.command_bar import render_command_bar, KPIData, get_kpis_from_database

# Import page renderers
from pages import (
    render_dashboard_page,
    render_pattern_lab_page,
    render_backtest_page,
    render_analytics_page,
    render_experiments_page,
    render_ml_training_page,
    render_settings_page,
    render_live_scanner_page,
    render_forward_test_page,
)


def main():
    """Main entry point for the dashboard."""

    # Apply Arctic Pro theme
    st.markdown(get_css(), unsafe_allow_html=True)

    # Hide default Streamlit elements
    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # === COMMAND BAR ===
    # Always visible at top with 6 KPIs
    # Priority: session_state (from recent backtest) > database > demo values

    if 'latest_metrics' in st.session_state and st.session_state.latest_metrics:
        # Use metrics from recent backtest run in this session
        metrics = st.session_state.latest_metrics
        kpis = KPIData(
            win_rate=metrics.get('win_rate', 0),
            sharpe=metrics.get('sharpe_ratio', 0),
            profit_factor=metrics.get('profit_factor', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            expectancy=metrics.get('expectancy', 0),
            kelly=metrics.get('kelly_criterion', 0),
        )
    else:
        # Try to load from database
        kpis = get_kpis_from_database()

        # If still empty (no data), show placeholder message
        if kpis.win_rate == 0 and kpis.sharpe == 0:
            st.info("💡 Run a backtest to see your metrics here")

    render_command_bar(kpis)

    # === TAB NAVIGATION ===
    tabs = st.tabs([
        "📊 Dashboard",
        "🔬 Pattern Lab",
        "📡 Live Scanner",
        "📈 Backtest",
        "📉 Analytics",
        "🎯 Forward Test",
        "🧪 Experiments",
        "🧠 ML Training",
        "⚙️ Settings",
    ])

    # Render content for each tab
    with tabs[0]:
        render_dashboard_page()

    with tabs[1]:
        render_pattern_lab_page()

    with tabs[2]:
        render_live_scanner_page()

    with tabs[3]:
        render_backtest_page()

    with tabs[4]:
        render_analytics_page()

    with tabs[5]:
        render_forward_test_page()

    with tabs[6]:
        render_experiments_page()

    with tabs[7]:
        render_ml_training_page()

    with tabs[8]:
        render_settings_page()


if __name__ == "__main__":
    main()
