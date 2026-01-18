"""
QML Dashboard Components Package
================================

Reusable UI components for the trading dashboard.

Components:
- cards: Metric cards, pattern cards, stat boxes
- backtest: Live backtest runner integration
- visualizations: Sparklines, gauges, data grids
"""

from .cards import (
    metric_card,
    pattern_card,
    stat_box,
    verdict_banner,
)
from .backtest import BacktestRunner
from .visualizations import (
    render_sparkline,
    circular_gauge,
    horizontal_bar,
    data_grid_2x2,
    ticker_tape,
    stats_table,
)

__all__ = [
    "metric_card",
    "pattern_card",
    "stat_box",
    "verdict_banner",
    "BacktestRunner",
    "render_sparkline",
    "circular_gauge",
    "horizontal_bar",
    "data_grid_2x2",
    "ticker_tape",
    "stats_table",
]
