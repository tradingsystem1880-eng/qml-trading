"""
Core Module
===========
Fundamental data structures and abstractions for the QML Trading System.

This module contains:
- models: Candle, Signal, Trade, SwingPoint dataclasses
- config: Configuration loading and management (TODO)
- exceptions: Custom exceptions (TODO)
"""

from src.core.models import (
    Candle,
    CandleList,
    Signal,
    SignalType,
    Trade,
    TradeResult,
    Side,
    SwingPoint,
    SwingType,
    PatternDirection,
)

__all__ = [
    'Candle',
    'CandleList',
    'Signal',
    'SignalType',
    'Trade',
    'TradeResult',
    'Side',
    'SwingPoint',
    'SwingType',
    'PatternDirection',
]
