"""
Swing Point Detection for QML Trading System
=============================================
Identifies significant swing highs and lows using ATR-adaptive thresholds.
Designed to filter noise while capturing meaningful price pivots.

Key improvements over naive implementations:
1. ATR-adaptive significance thresholds (timeframe-specific)
2. Gap handling for crypto markets
3. Confirmation-based validation
4. Efficient vectorized calculations
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from loguru import logger

from config.settings import settings
from src.data.models import SwingPoint, SwingType
from src.utils.indicators import calculate_atr


@dataclass
class SwingConfig:
    """Configuration for swing point detection."""
    
    # Minimum bars on each side of a swing point
    lookback_bars: int = 5
    lookforward_bars: int = 3
    
    # ATR multiplier for significance (timeframe-specific)
    atr_multiplier: float = 1.0
    
    # Minimum significance as percentage of price
    min_significance_pct: float = 0.3
    
    # ATR calculation period
    atr_period: int = 14
    
    # Confirmation settings
    require_confirmation: bool = True
    confirmation_bars: int = 2


class SwingDetector:
    """
    Detects significant swing highs and lows in price data.
    
    A swing high is a local maximum that exceeds surrounding prices
    by a statistically significant amount (ATR-normalized).
    
    A swing low is a local minimum below surrounding prices
    by a statistically significant amount.
    
    The detector uses adaptive thresholds based on:
    - ATR (Average True Range) for volatility adjustment
    - Timeframe-specific multipliers
    - Confirmation via subsequent price action
    """
    
    def __init__(self, config: Optional[SwingConfig] = None, timeframe: str = "4h"):
        """
        Initialize swing detector.
        
        Args:
            config: Detection configuration
            timeframe: Timeframe for parameter adjustment
        """
        self.config = config or SwingConfig()
        self.timeframe = timeframe
        
        # Apply timeframe-specific ATR multiplier from settings
        self.config.atr_multiplier = settings.detection.get_swing_atr_multiplier(timeframe)
    
    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> List[SwingPoint]:
        """
        Detect all swing points in the price data.
        
        Args:
            df: DataFrame with columns [time, high, low, close, volume]
            symbol: Trading pair symbol
            
        Returns:
            List of SwingPoint objects sorted by time
        """
        if len(df) < self.config.lookback_bars + self.config.lookforward_bars + self.config.atr_period:
            logger.warning(f"Insufficient data for swing detection: {len(df)} bars")
            return []
        
        # Extract arrays
        time = df["time"].values
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        
        # Calculate ATR
        atr = calculate_atr(high, low, close, self.config.atr_period)
        
        # Detect swing highs and lows
        swing_highs = self._detect_swing_highs(time, high, low, close, atr, symbol)
        swing_lows = self._detect_swing_lows(time, high, low, close, atr, symbol)
        
        # Combine and sort
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x.time)
        
        logger.debug(
            f"Detected {len(swing_highs)} swing highs and {len(swing_lows)} swing lows "
            f"for {symbol} {self.timeframe}"
        )
        
        return all_swings
    
    def _detect_swing_highs(
        self,
        time: NDArray,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        atr: NDArray[np.float64],
        symbol: str
    ) -> List[SwingPoint]:
        """
        Detect swing high points.
        
        A valid swing high must:
        1. Be higher than surrounding bars
        2. Exceed surrounding prices by at least ATR * multiplier
        3. Be confirmed by subsequent lower prices (optional)
        """
        swing_highs = []
        n = len(high)
        lb = self.config.lookback_bars
        lf = self.config.lookforward_bars
        
        for i in range(lb, n - lf):
            # Skip if ATR is not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            current_high = high[i]
            
            # Get surrounding highs
            left_highs = high[i - lb:i]
            right_highs = high[i + 1:i + lf + 1]
            
            # Check if current is a local maximum
            if current_high <= np.max(left_highs) or current_high <= np.max(right_highs):
                continue
            
            # Calculate significance (how much it exceeds surrounding prices)
            left_max = np.max(left_highs)
            right_max = np.max(right_highs)
            significance = min(current_high - left_max, current_high - right_max)
            
            # Normalize by ATR
            atr_significance = significance / atr[i]
            
            # Check minimum significance threshold
            min_threshold = max(
                self.config.atr_multiplier,
                self.config.min_significance_pct / 100 * current_high / atr[i]
            )
            
            if atr_significance < min_threshold:
                continue
            
            # Confirmation check: subsequent bars should close below the high
            confirmed = True
            if self.config.require_confirmation and i + self.config.confirmation_bars < n:
                confirm_closes = close[i + 1:i + 1 + self.config.confirmation_bars]
                confirmed = np.all(confirm_closes < current_high)
            
            # Create swing point
            swing_point = SwingPoint(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=self.timeframe,
                swing_type=SwingType.HIGH,
                price=float(current_high),
                significance=float(atr_significance),
                atr_at_point=float(atr[i]),
                confirmed=confirmed,
                bar_index=i
            )
            
            swing_highs.append(swing_point)
        
        return swing_highs
    
    def _detect_swing_lows(
        self,
        time: NDArray,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        atr: NDArray[np.float64],
        symbol: str
    ) -> List[SwingPoint]:
        """
        Detect swing low points.
        
        A valid swing low must:
        1. Be lower than surrounding bars
        2. Exceed surrounding prices by at least ATR * multiplier
        3. Be confirmed by subsequent higher prices (optional)
        """
        swing_lows = []
        n = len(low)
        lb = self.config.lookback_bars
        lf = self.config.lookforward_bars
        
        for i in range(lb, n - lf):
            # Skip if ATR is not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            current_low = low[i]
            
            # Get surrounding lows
            left_lows = low[i - lb:i]
            right_lows = low[i + 1:i + lf + 1]
            
            # Check if current is a local minimum
            if current_low >= np.min(left_lows) or current_low >= np.min(right_lows):
                continue
            
            # Calculate significance (how much it exceeds surrounding prices)
            left_min = np.min(left_lows)
            right_min = np.min(right_lows)
            significance = min(left_min - current_low, right_min - current_low)
            
            # Normalize by ATR
            atr_significance = significance / atr[i]
            
            # Check minimum significance threshold
            min_threshold = max(
                self.config.atr_multiplier,
                self.config.min_significance_pct / 100 * current_low / atr[i]
            )
            
            if atr_significance < min_threshold:
                continue
            
            # Confirmation check: subsequent bars should close above the low
            confirmed = True
            if self.config.require_confirmation and i + self.config.confirmation_bars < n:
                confirm_closes = close[i + 1:i + 1 + self.config.confirmation_bars]
                confirmed = np.all(confirm_closes > current_low)
            
            # Create swing point
            swing_point = SwingPoint(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=self.timeframe,
                swing_type=SwingType.LOW,
                price=float(current_low),
                significance=float(atr_significance),
                atr_at_point=float(atr[i]),
                confirmed=confirmed,
                bar_index=i
            )
            
            swing_lows.append(swing_point)
        
        return swing_lows
    
    def filter_significant_swings(
        self,
        swings: List[SwingPoint],
        min_significance: float = 1.0
    ) -> List[SwingPoint]:
        """
        Filter swings by minimum significance.
        
        Args:
            swings: List of swing points
            min_significance: Minimum ATR-normalized significance
            
        Returns:
            Filtered list of swing points
        """
        return [s for s in swings if s.significance >= min_significance]
    
    def get_recent_swings(
        self,
        swings: List[SwingPoint],
        count: int = 6,
        swing_type: Optional[SwingType] = None
    ) -> List[SwingPoint]:
        """
        Get most recent swing points.
        
        Args:
            swings: List of all swing points
            count: Number of recent swings to return
            swing_type: Filter by HIGH or LOW (None for both)
            
        Returns:
            List of recent swing points
        """
        if swing_type:
            filtered = [s for s in swings if s.swing_type == swing_type]
        else:
            filtered = swings
        
        # Sort by time descending and take last N
        sorted_swings = sorted(filtered, key=lambda x: x.time, reverse=True)
        return sorted_swings[:count]


def detect_swings(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: Optional[SwingConfig] = None
) -> List[SwingPoint]:
    """
    Convenience function to detect swing points.
    
    Args:
        df: OHLCV DataFrame
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        config: Optional configuration
        
    Returns:
        List of swing points
    """
    detector = SwingDetector(config=config, timeframe=timeframe)
    return detector.detect(df, symbol)

