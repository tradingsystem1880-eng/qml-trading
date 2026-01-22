"""
Premium Command Bar Component - Always visible top bar with 6 KPIs

Features:
- Gradient backgrounds on cards
- Colored accent borders (left border showing status)
- Mini sparkline charts inside metric cards
- Better typography hierarchy
- Subtle glow/shadow effects
- Progress indicators for metrics like Max DD

KPIs displayed:
- Win Rate (%)
- Sharpe Ratio
- Profit Factor
- Max Drawdown (%)
- Expectancy ($)
- Kelly (%)
"""

import streamlit as st
from typing import Optional, List
from dataclasses import dataclass, field

from theme import ARCTIC_PRO, TYPOGRAPHY, SPACING, RADIUS, format_value, get_value_class, generate_sparkline_svg


@dataclass
class KPIData:
    """Data container for command bar KPIs."""
    win_rate: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    expectancy: float = 0.0
    kelly: float = 0.0
    # Historical data for sparklines
    win_rate_history: List[float] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    equity_history: List[float] = field(default_factory=list)


def render_kpi_card(
    label: str,
    value: str,
    value_class: str = '',
    trend: str = None,
    trend_direction: str = None,
    sparkline_data: List[float] = None,
    sparkline_color: str = None,
    subtext: str = None,
    progress: float = None,
    progress_class: str = 'accent'
) -> str:
    """Render a single premium KPI card as HTML.

    Args:
        label: KPI label text
        value: Formatted value string
        value_class: CSS class for value (positive/negative)
        trend: Trend indicator text (e.g., "+2.3%")
        trend_direction: 'up' or 'down'
        sparkline_data: List of values for mini sparkline chart
        sparkline_color: Color for sparkline (defaults to accent)
        subtext: Additional context text below value
        progress: Progress bar fill percentage (0-100)
        progress_class: CSS class for progress bar (success/danger/accent)
    """
    card_class = f"kpi-card {value_class}" if value_class else "kpi-card"

    html = f'<div class="{card_class}">'

    # Header with label and trend
    html += '<div class="kpi-header">'
    html += f'<span class="kpi-label">{label}</span>'
    if trend and trend_direction:
        html += f'<span class="kpi-trend {trend_direction}">{trend}</span>'
    html += '</div>'

    # Main value
    html += f'<div class="kpi-value {value_class}">{value}</div>'

    # Subtext
    if subtext:
        html += f'<div class="kpi-subtext">{subtext}</div>'

    # Progress bar
    if progress is not None:
        html += f'<div class="progress-bar" style="margin-top: 8px;">'
        html += f'<div class="progress-fill {progress_class}" style="width: {min(100, max(0, progress))}%;"></div>'
        html += '</div>'

    # Sparkline
    if sparkline_data and len(sparkline_data) > 1:
        color = sparkline_color or ARCTIC_PRO['accent']
        if value_class == 'positive':
            color = ARCTIC_PRO['success']
        elif value_class == 'negative':
            color = ARCTIC_PRO['danger']

        svg = generate_sparkline_svg(sparkline_data, width=90, height=24, color=color)
        html += f'<div class="kpi-sparkline">{svg}</div>'

    html += '</div>'
    return html


def render_command_bar(kpis: Optional[KPIData] = None) -> None:
    """Render the premium command bar with 6 KPI cards.

    Args:
        kpis: KPIData object with metric values. If None, shows placeholder values.
    """
    if kpis is None:
        kpis = KPIData()

    # Generate mock sparkline data if not provided
    if not kpis.win_rate_history:
        kpis.win_rate_history = [52, 55, 53, 58, 54, 56, 59, 57, 60, 58]
    if not kpis.sharpe_history:
        kpis.sharpe_history = [1.2, 1.4, 1.3, 1.5, 1.6, 1.4, 1.7, 1.5, 1.8, 1.6]
    if not kpis.equity_history:
        kpis.equity_history = [10000, 10200, 10150, 10400, 10350, 10600, 10550, 10800, 10750, 11000]

    # Build KPI definitions
    kpi_definitions = [
        {
            'label': 'Win Rate',
            'value': format_value(kpis.win_rate, 'percent', 1),
            'class': get_value_class(kpis.win_rate - 50),
            'trend': '+2.3%' if kpis.win_rate > 0 else None,
            'trend_dir': 'up' if kpis.win_rate > 50 else 'down' if kpis.win_rate < 50 else None,
            'sparkline': kpis.win_rate_history,
            'subtext': f'{int(kpis.win_rate * 0.42)} / {42} trades' if kpis.win_rate > 0 else 'No trades yet',
        },
        {
            'label': 'Sharpe Ratio',
            'value': format_value(kpis.sharpe, 'number', 2),
            'class': get_value_class(kpis.sharpe),
            'trend': '+0.12' if kpis.sharpe > 0 else None,
            'trend_dir': 'up' if kpis.sharpe > 0 else 'down' if kpis.sharpe < 0 else None,
            'sparkline': kpis.sharpe_history,
            'subtext': 'Risk-adjusted return',
        },
        {
            'label': 'Profit Factor',
            'value': format_value(kpis.profit_factor, 'ratio', 2),
            'class': get_value_class(kpis.profit_factor - 1),
            'trend': None,
            'trend_dir': None,
            'sparkline': None,
            'subtext': 'Gross profit / loss',
        },
        {
            'label': 'Max Drawdown',
            'value': format_value(kpis.max_drawdown, 'percent', 1),
            'class': 'negative' if kpis.max_drawdown < 0 else '',
            'trend': None,
            'trend_dir': None,
            'sparkline': None,
            'subtext': 'Risk limit: 20%',
            'progress': abs(kpis.max_drawdown) / 20 * 100 if kpis.max_drawdown != 0 else 0,
            'progress_class': 'danger' if abs(kpis.max_drawdown) > 15 else 'accent',
        },
        {
            'label': 'Expectancy',
            'value': format_value(kpis.expectancy, 'currency', 0),
            'class': get_value_class(kpis.expectancy),
            'trend': None,
            'trend_dir': None,
            'sparkline': kpis.equity_history,
            'subtext': 'Expected $ per trade',
        },
        {
            'label': 'Kelly %',
            'value': format_value(kpis.kelly, 'percent', 1),
            'class': get_value_class(kpis.kelly),
            'trend': None,
            'trend_dir': None,
            'sparkline': None,
            'subtext': 'Optimal position size',
            'progress': min(100, kpis.kelly * 2) if kpis.kelly > 0 else 0,
            'progress_class': 'success' if kpis.kelly > 10 else 'accent',
        },
    ]

    # Build command bar HTML
    html = '<div class="command-bar">'
    html += '<div class="kpi-grid">'

    for kpi in kpi_definitions:
        html += render_kpi_card(
            label=kpi['label'],
            value=kpi['value'],
            value_class=kpi['class'],
            trend=kpi.get('trend'),
            trend_direction=kpi.get('trend_dir'),
            sparkline_data=kpi.get('sparkline'),
            subtext=kpi.get('subtext'),
            progress=kpi.get('progress'),
            progress_class=kpi.get('progress_class', 'accent'),
        )

    html += '</div></div>'

    st.markdown(html, unsafe_allow_html=True)


def get_kpis_from_results(results: dict) -> KPIData:
    """Extract KPI data from backtest results dictionary.

    Args:
        results: Dictionary containing backtest metrics

    Returns:
        KPIData object with populated values
    """
    return KPIData(
        win_rate=results.get('win_rate', 0.0),
        sharpe=results.get('sharpe_ratio', 0.0),
        profit_factor=results.get('profit_factor', 0.0),
        max_drawdown=results.get('max_drawdown', 0.0),
        expectancy=results.get('expectancy', 0.0),
        kelly=results.get('kelly_criterion', 0.0),
        equity_history=results.get('equity_curve', []),
    )


def get_kpis_from_database() -> KPIData:
    """Pull KPI data from the database via DataService.

    Attempts to load metrics from the latest experiment.
    Falls back to empty KPIs if no data exists.

    Returns:
        KPIData object with values from database
    """
    try:
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from qml.dashboard.data.data_service import get_data_service

        service = get_data_service()
        metrics = service.get_command_bar_metrics()

        return KPIData(
            win_rate=metrics.get('win_rate', 0.0),
            sharpe=metrics.get('sharpe_ratio', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            expectancy=metrics.get('expectancy', 0.0),
            kelly=metrics.get('kelly_criterion', 0.0),
        )
    except Exception as e:
        # Fall back to empty KPIs on error
        print(f"[command_bar] Error loading from database: {e}")
        return KPIData()
