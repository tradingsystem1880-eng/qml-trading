"""
QML Trading System
==================
Professional-grade algo trading system for QML pattern detection.

Core modules:
- qml.core: Data loading, indicators, configuration
- qml.strategy: Pattern detection and signal generation
- qml.backtest: Backtesting engine
- qml.validation: Walk-forward, Monte Carlo, permutation tests
- qml.dashboard: Streamlit UI (the "brain")

Architecture follows Freqtrade/NautilusTrader patterns.

Usage:
    from qml import QMLEngine
    
    engine = QMLEngine()
    patterns = engine.detect_patterns("BTC/USDT", "4h")
    results = engine.backtest(patterns)
"""

__version__ = "2.0.0"
__author__ = "QML System"

# Core engine
from qml.core.engine import QMLEngine
from qml.core.config import QMLConfig

# Easy imports for common operations
from qml.core.data import DataLoader
from qml.core.indicators import Indicators
from qml.strategy.detector import PatternDetector
from qml.backtest.engine import BacktestEngine
from qml.validation.validator import Validator

__all__ = [
    "QMLEngine",
    "QMLConfig",
    "DataLoader",
    "Indicators",
    "PatternDetector",
    "BacktestEngine",
    "Validator",
]
