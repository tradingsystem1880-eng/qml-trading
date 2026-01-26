"""
QML Trading System
==================
Professional-grade algo trading system for QML pattern detection.

Core modules:
- qml.core: Data loading, indicators, configuration
- qml.dashboard: Streamlit UI (the "brain")

The dashboard directly imports from src/ for detection, backtest, and validation.

Usage:
    # Run the dashboard
    streamlit run qml/dashboard/app_v2.py
"""

__version__ = "2.0.0"
__author__ = "QML System"

# Core utilities (config, data loading, indicators)
from qml.core.config import QMLConfig
from qml.core.data import DataLoader
from qml.core.indicators import Indicators

__all__ = [
    "QMLConfig",
    "DataLoader",
    "Indicators",
]
