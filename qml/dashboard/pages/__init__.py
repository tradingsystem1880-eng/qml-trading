"""
QML Dashboard Pages Module

Each page corresponds to a tab in the main navigation.
"""

from .dashboard_page import render_dashboard_page
from .pattern_lab_page import render_pattern_lab_page
from .backtest_page import render_backtest_page
from .analytics_page import render_analytics_page
from .experiments_page import render_experiments_page
from .ml_training_page import render_ml_training_page
from .settings_page import render_settings_page
from .live_scanner_page import render_live_scanner_page
from .forward_test_page import render_forward_test_page

__all__ = [
    'render_dashboard_page',
    'render_pattern_lab_page',
    'render_backtest_page',
    'render_analytics_page',
    'render_experiments_page',
    'render_ml_training_page',
    'render_settings_page',
    'render_live_scanner_page',
    'render_forward_test_page',
]
