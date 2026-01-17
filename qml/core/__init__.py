"""
QML Core Package
================
Core functionality for the QML Trading System.

Contains:
- engine.py: Main QML engine (the central hub)
- config.py: Configuration management
- data.py: Data loading and management
- indicators.py: Technical indicators (powered by ta library)
"""

from qml.core.config import QMLConfig, default_config
from qml.core.engine import QMLEngine
from qml.core.data import DataLoader
from qml.core.indicators import Indicators

__all__ = [
    "QMLConfig",
    "default_config",
    "QMLEngine",
    "DataLoader",
    "Indicators",
]
