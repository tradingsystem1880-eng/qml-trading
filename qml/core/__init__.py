"""
QML Core Package
================
Core functionality for the QML Trading System.

Contains:
- config.py: Configuration management
- data.py: Data loading and management
- indicators.py: Technical indicators (powered by ta library)

Note: The engine has been removed. Dashboard imports directly from src/.
"""

from qml.core.config import QMLConfig, default_config
from qml.core.data import DataLoader
from qml.core.indicators import Indicators

__all__ = [
    "QMLConfig",
    "default_config",
    "DataLoader",
    "Indicators",
]
