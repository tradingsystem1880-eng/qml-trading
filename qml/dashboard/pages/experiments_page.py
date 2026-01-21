"""
Experiments Page - A/B Testing and Parameter Grid Search Dashboard
==================================================================
Phase 6 implementation for systematic parameter optimization.

Features:
- Grid search progress tracking
- Parameter configuration
- Results comparison with BH correction
- Statistical significance filtering
- Visualizations (Sharpe distribution, Win Rate vs Sharpe)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from theme import ARCTIC_PRO, TYPOGRAPHY

# Try to import experiments module
try:
    from src.experiments import (
        ParameterSet,
        GridSearchConfig,
        ParameterGridManager,
        ExperimentRunner,
        ExperimentResult,
        get_significant_discoveries,
        add_p_values_to_experiments,
        rank_experiments,
    )
    from src.data.sqlite_manager import SQLiteManager, get_db
    EXPERIMENTS_AVAILABLE = True
except ImportError as e:
    EXPERIMENTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


def render_experiments_page() -> None:
    """Render the experiments comparison page."""

    # Page header
    html = '<div class="panel">'
    html += '<div class="panel-header">Experiment Lab</div>'
    html += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Systematic A/B testing with statistical significance filtering (BH FDR correction).</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    if not EXPERIMENTS_AVAILABLE:
        _render_import_error()
        return

    # Initialize database
    try:
        db = get_db()
        grid_manager = ParameterGridManager(db)
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Grid Progress",
        "Run Experiments",
        "Results Analysis",
        "Top Performers"
    ])

    with tab1:
        _render_grid_progress(grid_manager)

    with tab2:
        _render_run_experiments(db, grid_manager)

    with tab3:
        _render_results_analysis(db)

    with tab4:
        _render_top_performers(db)


def _render_import_error() -> None:
    """Render import error message."""
    error_html = '<div class="panel">'
    error_html += f'<div style="color: {ARCTIC_PRO["danger"]}; padding: 1rem;">'
    error_html += '<strong>Experiments module not available</strong><br>'
    error_html += f'<span style="color: {ARCTIC_PRO["text_muted"]};">Import error: {IMPORT_ERROR}</span>'
    error_html += '</div>'
    error_html += '</div>'
    st.markdown(error_html, unsafe_allow_html=True)


def _render_grid_progress(grid_manager: ParameterGridManager) -> None:
    """Render grid search progress section."""

    # Grid configurations
    config_options = {
        "Full Grid (~210K)": GridSearchConfig(),
        "Small Grid (~100)": GridSearchConfig.small(),
        "Minimal (1)": GridSearchConfig.minimal(),
    }

    selected_config = st.selectbox(
        "Select Grid Configuration",
        options=list(config_options.keys()),
        key="grid_config_select"
    )
    config = config_options[selected_config]

    # Calculate stats
    total = config.total_combinations()
    progress = grid_manager.get_fast_progress(config)

    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card("Total Combinations", f"{total:,}", "")

    with col2:
        _render_metric_card("Tested", f"{progress['tested']:,}", "")

    with col3:
        _render_metric_card("Remaining", f"{progress['remaining']:,}", "")

    with col4:
        _render_metric_card("Progress", f"{progress['progress_pct']:.1f}%", "")

    # Progress bar
    progress_pct = min(progress['progress_pct'], 100)
    progress_html = '<div class="panel">'
    progress_html += '<div class="panel-header">Grid Search Progress</div>'
    progress_html += '<div class="progress-bar" style="height: 12px; margin-top: 1rem;">'
    progress_html += f'<div class="progress-fill accent" style="width: {progress_pct}%;"></div>'
    progress_html += '</div>'
    progress_html += f'<p style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; margin-top: 0.5rem; text-align: center;">'
    progress_html += f'{progress["tested"]:,} of {total:,} parameter combinations tested'
    progress_html += '</p>'
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

    # Grid configuration details
    with st.expander("View Grid Configuration"):
        config_dict = config.to_dict()
        for param, values in config_dict.items():
            st.write(f"**{param}**: {values}")


def _render_run_experiments(db: SQLiteManager, grid_manager: ParameterGridManager) -> None:
    """Render experiment runner section."""

    run_panel = '<div class="panel">'
    run_panel += '<div class="panel-header">Run New Experiments</div>'
    run_panel += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Configure and launch parameter grid search.</p>'
    run_panel += '</div>'
    st.markdown(run_panel, unsafe_allow_html=True)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            key="exp_symbol"
        )
        timeframe = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            index=1,
            key="exp_timeframe"
        )

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=date(2024, 1, 1),
            key="exp_start_date"
        )
        end_date = st.date_input(
            "End Date",
            value=date(2024, 12, 31),
            key="exp_end_date"
        )

    # Advanced options
    with st.expander("Advanced Options"):
        max_experiments = st.number_input(
            "Max Experiments to Run",
            min_value=1,
            max_value=10000,
            value=100,
            step=10,
            key="max_experiments"
        )

        run_validation = st.checkbox(
            "Run Phase 4 Validation (slower)",
            value=False,
            key="run_validation"
        )

        skip_tested = st.checkbox(
            "Skip Already Tested Parameters",
            value=True,
            key="skip_tested"
        )

    # Run button
    st.markdown("---")

    if st.button("Start Grid Search", type="primary", use_container_width=True, key="run_grid"):
        st.info("Grid search functionality requires backtest integration. Configure your backtest function in the ExperimentRunner.")

        # Show example code
        with st.expander("Integration Example"):
            st.code("""
from src.experiments import ExperimentRunner, GridSearchConfig
from src.data.sqlite_manager import get_db

def my_backtest(params, symbol, timeframe, start_date, end_date):
    # Your backtest logic here
    return {
        'total_trades': 50,
        'win_rate': 0.55,
        'sharpe_ratio': 1.2,
        'profit_factor': 1.8,
        'max_drawdown': 0.15,
        'total_return': 0.25,
    }

db = get_db()
runner = ExperimentRunner(db, backtest_func=my_backtest)

results = runner.run_grid(
    config=GridSearchConfig.small(),
    symbol='BTCUSDT',
    timeframe='4h',
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    max_experiments=100,
)
""", language="python")


def _render_results_analysis(db: SQLiteManager) -> None:
    """Render results analysis section with BH correction."""

    analysis_panel = '<div class="panel">'
    analysis_panel += '<div class="panel-header">Results Analysis</div>'
    analysis_panel += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Compare experiments with Benjamini-Hochberg FDR correction.</p>'
    analysis_panel += '</div>'
    st.markdown(analysis_panel, unsafe_allow_html=True)

    # Load experiments from database
    try:
        experiments = db.get_all_experiments(limit=500)
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")
        return

    if not experiments:
        st.info("No experiments found. Run some backtests first.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_trades = st.number_input(
            "Min Trades",
            min_value=0,
            max_value=200,
            value=30,
            key="analysis_min_trades"
        )

    with col2:
        alpha = st.selectbox(
            "FDR Alpha",
            [0.01, 0.05, 0.10],
            index=1,
            key="fdr_alpha"
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["sharpe_ratio", "win_rate", "profit_factor", "total_return"],
            key="sort_by"
        )

    significant_only = st.checkbox(
        "Show Significant Results Only (after BH correction)",
        value=False,
        key="significant_only"
    )

    # Filter experiments
    filtered = [e for e in experiments if (e.get('total_trades') or 0) >= min_trades]

    if not filtered:
        st.warning(f"No experiments with at least {min_trades} trades.")
        return

    # Add p-values and apply BH correction
    filtered_with_p = add_p_values_to_experiments(filtered.copy())
    discovery_results = get_significant_discoveries(
        filtered_with_p,
        alpha=alpha,
        min_trades=min_trades,
    )

    # Show summary
    summary = discovery_results['summary']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card("Experiments", str(summary['total_experiments']), "")

    with col2:
        _render_metric_card("Tested for Sig.", str(summary['tested_for_significance']), "")

    with col3:
        sig_count = summary['significant_discoveries']
        _render_metric_card("Significant", str(sig_count), "success" if sig_count > 0 else "")

    with col4:
        _render_metric_card("Discovery Rate", f"{summary['discovery_rate']:.1f}%", "")

    # Display data
    if significant_only:
        display_data = discovery_results['significant']
    else:
        display_data = filtered_with_p

    if not display_data:
        st.info("No results match the current filters.")
        return

    # Sort
    display_data = sorted(
        display_data,
        key=lambda x: x.get(sort_by, 0) or 0,
        reverse=True
    )

    # Create DataFrame for display
    df_data = []
    for exp in display_data[:100]:  # Limit to 100 for performance
        row = {
            'Symbol': exp.get('symbol', 'N/A'),
            'Trades': exp.get('total_trades', 0),
            'Win Rate': f"{(exp.get('win_rate') or 0) * 100:.1f}%",
            'Sharpe': f"{exp.get('sharpe_ratio', 0):.2f}",
            'Profit Factor': f"{exp.get('profit_factor', 0):.2f}",
            'Max DD': f"{(exp.get('max_drawdown') or 0) * 100:.1f}%",
            'P-Value': f"{exp.get('p_value', 1):.4f}",
            'Significant': '✓' if exp.get('is_significant') else '',
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualizations
    st.markdown("---")
    _render_visualizations(filtered_with_p, discovery_results)


def _render_visualizations(experiments: List[Dict], discovery_results: Dict) -> None:
    """Render analysis visualizations."""

    col1, col2 = st.columns(2)

    with col1:
        # Sharpe distribution histogram
        sharpes = [e.get('sharpe_ratio', 0) for e in experiments if e.get('sharpe_ratio') is not None]

        if sharpes:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sharpes,
                nbinsx=30,
                marker_color=ARCTIC_PRO['accent'],
                opacity=0.7,
            ))

            # Add significance threshold line (Sharpe = 0)
            fig.add_vline(x=0, line_dash="dash", line_color=ARCTIC_PRO['text_muted'])

            fig.update_layout(
                title="Sharpe Ratio Distribution",
                xaxis_title="Sharpe Ratio",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor=ARCTIC_PRO['bg_card'],
                plot_bgcolor=ARCTIC_PRO['bg_secondary'],
                font=dict(color=ARCTIC_PRO['text_secondary']),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Win Rate vs Sharpe scatter
        win_rates = []
        sharpe_vals = []
        is_significant = []

        for e in experiments:
            wr = e.get('win_rate')
            sr = e.get('sharpe_ratio')
            if wr is not None and sr is not None:
                win_rates.append(wr * 100)
                sharpe_vals.append(sr)
                is_significant.append(e.get('is_significant', False))

        if win_rates and sharpe_vals:
            colors = [ARCTIC_PRO['success'] if sig else ARCTIC_PRO['accent'] for sig in is_significant]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=win_rates,
                y=sharpe_vals,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.7,
                ),
                hovertemplate='Win Rate: %{x:.1f}%<br>Sharpe: %{y:.2f}<extra></extra>',
            ))

            fig.update_layout(
                title="Win Rate vs Sharpe Ratio",
                xaxis_title="Win Rate (%)",
                yaxis_title="Sharpe Ratio",
                template="plotly_dark",
                paper_bgcolor=ARCTIC_PRO['bg_card'],
                plot_bgcolor=ARCTIC_PRO['bg_secondary'],
                font=dict(color=ARCTIC_PRO['text_secondary']),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)


def _render_top_performers(db: SQLiteManager) -> None:
    """Render top performing experiments."""

    top_panel = '<div class="panel">'
    top_panel += '<div class="panel-header">Top Performers</div>'
    top_panel += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Best parameter configurations ranked by Sharpe ratio (statistically significant only).</p>'
    top_panel += '</div>'
    st.markdown(top_panel, unsafe_allow_html=True)

    # Load experiments
    try:
        experiments = db.get_all_experiments(limit=500)
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")
        return

    if not experiments:
        st.info("No experiments found.")
        return

    # Apply BH correction and get significant results
    experiments_with_p = add_p_values_to_experiments(experiments.copy())
    discovery = get_significant_discoveries(experiments_with_p, alpha=0.05, min_trades=30)

    significant = discovery['significant']

    if not significant:
        st.warning("No statistically significant results found (after BH correction at α=0.05 with min 30 trades).")

        # Show best non-significant results as fallback
        st.markdown("### Best Results (Not Statistically Significant)")
        ranked = rank_experiments(experiments_with_p, min_trades=30)[:10]

        for i, exp in enumerate(ranked, 1):
            _render_experiment_card(i, exp, is_significant=False)
        return

    # Show significant top performers
    st.markdown(f"### {len(significant)} Statistically Significant Results")

    for i, exp in enumerate(significant[:10], 1):
        _render_experiment_card(i, exp, is_significant=True)


def _render_experiment_card(rank: int, exp: Dict, is_significant: bool) -> None:
    """Render a single experiment result card."""

    sig_badge = '<span class="status-badge success">Significant</span>' if is_significant else '<span class="status-badge neutral">Not Significant</span>'

    sharpe = exp.get('sharpe_ratio', 0) or 0
    win_rate = (exp.get('win_rate', 0) or 0) * 100
    pf = exp.get('profit_factor', 0) or 0
    trades = exp.get('total_trades', 0) or 0
    p_val = exp.get('p_value', 1)
    adj_p_val = exp.get('adjusted_p_value', 1)

    sharpe_class = 'positive' if sharpe > 0 else 'negative' if sharpe < 0 else ''

    card_html = f'<div class="panel" style="margin-bottom: 0.5rem;">'
    card_html += f'<div style="display: flex; justify-content: space-between; align-items: center;">'
    card_html += f'<div style="display: flex; align-items: center; gap: 1rem;">'
    card_html += f'<span style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_lg"]}; font-weight: bold;">#{rank}</span>'
    card_html += f'<div>'
    card_html += f'<div style="color: {ARCTIC_PRO["text_primary"]}; font-weight: 600;">'
    card_html += f'{exp.get("symbol", "N/A")} • {exp.get("timeframe", "N/A")}'
    card_html += f'</div>'
    card_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]};">'
    card_html += f'{trades} trades'
    card_html += f'</div>'
    card_html += f'</div>'
    card_html += f'</div>'

    card_html += f'<div style="display: flex; gap: 2rem; align-items: center;">'

    # Metrics
    card_html += f'<div style="text-align: center;">'
    card_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_xs"]}; text-transform: uppercase;">Sharpe</div>'
    card_html += f'<div class="metric-value {sharpe_class}" style="font-size: {TYPOGRAPHY["size_lg"]};">{sharpe:.2f}</div>'
    card_html += f'</div>'

    card_html += f'<div style="text-align: center;">'
    card_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_xs"]}; text-transform: uppercase;">Win Rate</div>'
    card_html += f'<div style="color: {ARCTIC_PRO["text_primary"]}; font-size: {TYPOGRAPHY["size_lg"]}; font-family: {TYPOGRAPHY["font_mono"]};">{win_rate:.1f}%</div>'
    card_html += f'</div>'

    card_html += f'<div style="text-align: center;">'
    card_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_xs"]}; text-transform: uppercase;">P-Value</div>'
    card_html += f'<div style="color: {ARCTIC_PRO["text_primary"]}; font-size: {TYPOGRAPHY["size_lg"]}; font-family: {TYPOGRAPHY["font_mono"]};">{adj_p_val:.4f}</div>'
    card_html += f'</div>'

    card_html += sig_badge
    card_html += f'</div>'

    card_html += f'</div>'
    card_html += f'</div>'

    st.markdown(card_html, unsafe_allow_html=True)


def _render_metric_card(label: str, value: str, style: str = "") -> None:
    """Render a simple metric card."""
    value_class = style if style in ['positive', 'negative', 'success', 'danger'] else ''

    html = '<div class="metric-card">'
    html += f'<div class="metric-label">{label}</div>'
    html += f'<div class="metric-value {value_class}">{value}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
