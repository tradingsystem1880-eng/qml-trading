"""
Historical Swing Detector
=========================
Idempotent batch detector for backtesting and historical analysis.

Key properties:
- Uses scipy.signal.argrelextrema for deterministic local extrema detection
- Calculates own ATR using pandas_ta (no dependency on df columns)
- Z-score based significance filtering for cross-timeframe consistency
- Stores atr_at_formation for each swing point
- Idempotent: same input always produces identical output

This detector coexists with v2_atr.py:
- HistoricalSwingDetector: batch backtesting, idempotent, uses lookforward
- v2_atr: live streaming, causal, state machine
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from src.detection.config import SwingDetectionConfig, DetectionConfig


@dataclass
class HistoricalSwingPoint:
    """
    Swing point detected by the historical detector.

    Contains all information needed for pattern detection and scoring.
    """
    bar_index: int
    price: float
    timestamp: pd.Timestamp
    swing_type: str  # 'HIGH' or 'LOW'

    # ATR context
    atr_at_formation: float  # ATR at the time of detection

    # Significance metrics
    significance_atr: float  # ATR-normalized significance
    significance_zscore: float  # Z-score for cross-timeframe comparison

    # Metadata
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    confirmed: bool = True  # Historical swings are always confirmed

    @property
    def is_high(self) -> bool:
        """Check if this is a swing high."""
        return self.swing_type == 'HIGH'

    @property
    def is_low(self) -> bool:
        """Check if this is a swing low."""
        return self.swing_type == 'LOW'


class HistoricalSwingDetector:
    """
    Idempotent batch swing point detector.

    Uses scipy.signal.argrelextrema for deterministic detection of local
    extrema, with z-score based significance filtering.

    This detector is designed for:
    - Backtesting (where lookahead is acceptable)
    - Historical analysis
    - Strategy research

    For live trading, use the streaming ATR detector (v2_atr.py).
    """

    def __init__(
        self,
        config: Optional[SwingDetectionConfig] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ):
        """
        Initialize the historical detector.

        Args:
            config: Swing detection configuration
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        self.config = config or SwingDetectionConfig()
        self.symbol = symbol
        self.timeframe = timeframe

    def detect(self, df: pd.DataFrame) -> List[HistoricalSwingPoint]:
        """
        Detect all swing points in historical data.

        This is the main entry point. It's idempotent: calling with the
        same DataFrame will always produce identical results.

        Args:
            df: OHLCV DataFrame with columns [time, open, high, low, close, volume]

        Returns:
            List of HistoricalSwingPoint objects sorted by bar index
        """
        if len(df) < self.config.atr_period + self.config.lookback + self.config.lookforward:
            return []

        # Calculate ATR independently (don't rely on df having it)
        atr = self._calculate_atr(df)

        # Detect swing highs and lows using argrelextrema
        swing_highs = self._detect_swing_highs(df, atr)
        swing_lows = self._detect_swing_lows(df, atr)

        # Combine and sort by bar index
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x.bar_index)

        return all_swings

    def _calculate_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> np.ndarray:
        """
        Calculate ATR using Wilder's smoothing method.

        This ensures consistent ATR calculation regardless of whether
        pandas_ta is available.

        Args:
            df: OHLCV DataFrame
            period: ATR period (defaults to config value)

        Returns:
            numpy array of ATR values
        """
        period = period or self.config.atr_period

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range calculation
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        # First value can't use previous close
        high_close[0] = high_low[0]
        low_close[0] = high_low[0]

        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # Wilder's smoothing (EMA with alpha = 1/period)
        atr = np.zeros_like(tr)
        atr[period - 1] = np.mean(tr[:period])  # Initial SMA

        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return atr

    def _detect_swing_highs(
        self,
        df: pd.DataFrame,
        atr: np.ndarray
    ) -> List[HistoricalSwingPoint]:
        """
        Detect swing high points using scipy argrelextrema.

        A swing high is a local maximum where:
        1. It's higher than surrounding bars (order = lookback + lookforward)
        2. The price move exceeds the minimum ATR threshold
        3. The z-score is above the minimum significance level

        Args:
            df: OHLCV DataFrame
            atr: Pre-calculated ATR array

        Returns:
            List of swing high points
        """
        highs = df['high'].values
        time = df['time'].values if 'time' in df.columns else df.index.values

        # Use argrelextrema for deterministic detection
        order = max(self.config.lookback, self.config.lookforward)
        local_max_indices = argrelextrema(highs, np.greater_equal, order=order)[0]

        swing_highs = []
        all_significances = []

        # First pass: collect all potential swing highs and their significances
        for i in local_max_indices:
            if i < self.config.atr_period or i >= len(df) - self.config.lookforward:
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            # Calculate significance: how much it exceeds surrounding lows
            window_start = max(0, i - self.config.lookback)
            window_end = min(len(df), i + self.config.lookforward + 1)
            surrounding_lows = df['low'].values[window_start:window_end]
            min_low = np.min(surrounding_lows)

            swing_size = highs[i] - min_low
            atr_significance = swing_size / current_atr

            # Check minimum threshold
            min_threshold = max(
                self.config.atr_multiplier,
                highs[i] * self.config.min_threshold_pct / current_atr
            )

            if atr_significance >= min_threshold:
                all_significances.append(atr_significance)

        # Calculate z-score parameters (for all detected swings)
        if not all_significances:
            return []

        mean_sig = np.mean(all_significances)
        std_sig = np.std(all_significances) if len(all_significances) > 1 else 1.0
        std_sig = max(std_sig, 0.001)  # Prevent division by zero

        # Second pass: create swing points with z-scores
        sig_idx = 0
        for i in local_max_indices:
            if i < self.config.atr_period or i >= len(df) - self.config.lookforward:
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            window_start = max(0, i - self.config.lookback)
            window_end = min(len(df), i + self.config.lookforward + 1)
            surrounding_lows = df['low'].values[window_start:window_end]
            min_low = np.min(surrounding_lows)

            swing_size = highs[i] - min_low
            atr_significance = swing_size / current_atr

            min_threshold = max(
                self.config.atr_multiplier,
                highs[i] * self.config.min_threshold_pct / current_atr
            )

            if atr_significance >= min_threshold:
                zscore = (atr_significance - mean_sig) / std_sig

                # Only include if z-score meets minimum
                if zscore >= self.config.min_zscore or len(all_significances) < 5:
                    swing_highs.append(HistoricalSwingPoint(
                        bar_index=i,
                        price=float(highs[i]),
                        timestamp=pd.Timestamp(time[i]),
                        swing_type='HIGH',
                        atr_at_formation=float(current_atr),
                        significance_atr=float(atr_significance),
                        significance_zscore=float(zscore),
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                    ))

        return swing_highs

    def _detect_swing_lows(
        self,
        df: pd.DataFrame,
        atr: np.ndarray
    ) -> List[HistoricalSwingPoint]:
        """
        Detect swing low points using scipy argrelextrema.

        A swing low is a local minimum where:
        1. It's lower than surrounding bars (order = lookback + lookforward)
        2. The price move exceeds the minimum ATR threshold
        3. The z-score is above the minimum significance level

        Args:
            df: OHLCV DataFrame
            atr: Pre-calculated ATR array

        Returns:
            List of swing low points
        """
        lows = df['low'].values
        time = df['time'].values if 'time' in df.columns else df.index.values

        # Use argrelextrema for deterministic detection
        order = max(self.config.lookback, self.config.lookforward)
        local_min_indices = argrelextrema(lows, np.less_equal, order=order)[0]

        swing_lows = []
        all_significances = []

        # First pass: collect all potential swing lows and their significances
        for i in local_min_indices:
            if i < self.config.atr_period or i >= len(df) - self.config.lookforward:
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            # Calculate significance: how much it drops below surrounding highs
            window_start = max(0, i - self.config.lookback)
            window_end = min(len(df), i + self.config.lookforward + 1)
            surrounding_highs = df['high'].values[window_start:window_end]
            max_high = np.max(surrounding_highs)

            swing_size = max_high - lows[i]
            atr_significance = swing_size / current_atr

            # Check minimum threshold
            min_threshold = max(
                self.config.atr_multiplier,
                lows[i] * self.config.min_threshold_pct / current_atr
            )

            if atr_significance >= min_threshold:
                all_significances.append(atr_significance)

        # Calculate z-score parameters
        if not all_significances:
            return []

        mean_sig = np.mean(all_significances)
        std_sig = np.std(all_significances) if len(all_significances) > 1 else 1.0
        std_sig = max(std_sig, 0.001)

        # Second pass: create swing points with z-scores
        for i in local_min_indices:
            if i < self.config.atr_period or i >= len(df) - self.config.lookforward:
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            window_start = max(0, i - self.config.lookback)
            window_end = min(len(df), i + self.config.lookforward + 1)
            surrounding_highs = df['high'].values[window_start:window_end]
            max_high = np.max(surrounding_highs)

            swing_size = max_high - lows[i]
            atr_significance = swing_size / current_atr

            min_threshold = max(
                self.config.atr_multiplier,
                lows[i] * self.config.min_threshold_pct / current_atr
            )

            if atr_significance >= min_threshold:
                zscore = (atr_significance - mean_sig) / std_sig

                # Only include if z-score meets minimum
                if zscore >= self.config.min_zscore or len(all_significances) < 5:
                    swing_lows.append(HistoricalSwingPoint(
                        bar_index=i,
                        price=float(lows[i]),
                        timestamp=pd.Timestamp(time[i]),
                        swing_type='LOW',
                        atr_at_formation=float(current_atr),
                        significance_atr=float(atr_significance),
                        significance_zscore=float(zscore),
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                    ))

        return swing_lows

    def get_swing_stats(self, swings: List[HistoricalSwingPoint]) -> dict:
        """
        Get summary statistics for detected swings.

        Useful for debugging and analysis.

        Args:
            swings: List of detected swing points

        Returns:
            Dictionary with swing statistics
        """
        if not swings:
            return {
                'total': 0,
                'highs': 0,
                'lows': 0,
                'mean_significance': 0,
                'mean_zscore': 0,
            }

        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if s.is_low]

        return {
            'total': len(swings),
            'highs': len(highs),
            'lows': len(lows),
            'mean_significance': np.mean([s.significance_atr for s in swings]),
            'mean_zscore': np.mean([s.significance_zscore for s in swings]),
            'min_zscore': min(s.significance_zscore for s in swings),
            'max_zscore': max(s.significance_zscore for s in swings),
        }
