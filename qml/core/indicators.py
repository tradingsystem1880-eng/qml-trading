"""
Technical Indicators
====================
Wrapper around ta library for all indicators.

Uses battle-tested ta library (3.7k+ stars).
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from loguru import logger


class Indicators:
    """
    Technical indicators using ta library.
    
    All indicators are static methods for easy use.
    
    Example:
        from qml.core.indicators import Indicators
        
        atr = Indicators.atr(high, low, close, period=14)
        rsi = Indicators.rsi(close, period=14)
    """
    
    @staticmethod
    def atr(
        high: NDArray,
        low: NDArray,
        close: NDArray,
        period: int = 14
    ) -> NDArray:
        """Calculate Average True Range."""
        from src.utils.indicators import calculate_atr
        return calculate_atr(high, low, close, period)
    
    @staticmethod
    def rsi(close: NDArray, period: int = 14) -> NDArray:
        """Calculate Relative Strength Index."""
        from src.utils.indicators import calculate_rsi
        return calculate_rsi(close, period)
    
    @staticmethod
    def macd(
        close: NDArray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """Calculate MACD."""
        from src.utils.indicators import calculate_macd
        return calculate_macd(close, fast, slow, signal)
    
    @staticmethod
    def bollinger_bands(
        close: NDArray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """Calculate Bollinger Bands."""
        from src.utils.indicators import calculate_bollinger_bands
        return calculate_bollinger_bands(close, period, std_dev)
    
    @staticmethod
    def ema(data: NDArray, period: int) -> NDArray:
        """Calculate EMA."""
        from src.utils.indicators import calculate_ema
        return calculate_ema(data, period)
    
    @staticmethod
    def sma(data: NDArray, period: int) -> NDArray:
        """Calculate SMA."""
        from src.utils.indicators import calculate_sma
        return calculate_sma(data, period)
    
    @staticmethod
    def adx(
        high: NDArray,
        low: NDArray,
        close: NDArray,
        period: int = 14
    ) -> NDArray:
        """Calculate ADX (trend strength)."""
        from src.utils.indicators import calculate_adx
        return calculate_adx(high, low, close, period)
