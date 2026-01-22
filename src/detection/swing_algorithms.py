"""
Swing Point Detection with multiple algorithm options.

Algorithms:
1. ROLLING - Simple rolling window high/low (baseline)
2. SAVGOL - Savitzky-Golay smoothing before detection
3. FRACTAL - Williams Fractal pattern (5-bar)
4. WAVELET - Multi-scale wavelet analysis (optional PyWavelets)

All algorithms use ATR-based minimum swing size filtering.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelextrema
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Optional wavelet support
try:
    import pywt
    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False


class SwingAlgorithm(Enum):
    """Available swing detection algorithms."""
    ROLLING = "rolling"
    SAVGOL = "savgol"
    FRACTAL = "fractal"
    WAVELET = "wavelet"


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    index: int
    price: float
    timestamp: pd.Timestamp
    swing_type: str  # 'HIGH' or 'LOW'
    strength: float  # Confidence/strength (0-1)


@dataclass
class SwingConfig:
    """Configuration for swing detection."""
    algorithm: SwingAlgorithm = SwingAlgorithm.ROLLING
    lookback: int = 5
    smoothing_window: int = 5
    min_swing_atr: float = 0.3
    wavelet_scales: List[int] = field(default_factory=lambda: [3, 5, 8])


class MultiAlgorithmSwingDetector:
    """
    Multi-algorithm swing point detector.

    Supports four detection algorithms with ATR-based filtering.
    Uses Wilder's smoothing for ATR calculation (DeepSeek fix).
    """

    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect swing points using configured algorithm.

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close' columns

        Returns:
            List of SwingPoint objects sorted by index
        """
        if len(df) < self.config.lookback * 2 + 1:
            return []

        df = self._add_atr(df)

        if self.config.algorithm == SwingAlgorithm.ROLLING:
            return self._detect_rolling(df)
        elif self.config.algorithm == SwingAlgorithm.SAVGOL:
            return self._detect_savgol(df)
        elif self.config.algorithm == SwingAlgorithm.FRACTAL:
            return self._detect_fractal(df)
        elif self.config.algorithm == SwingAlgorithm.WAVELET:
            return self._detect_wavelet(df)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add ATR column using Wilder's smoothing (DeepSeek fix).

        Uses EMA with alpha = 1/period instead of SMA.
        """
        df = df.copy()

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Wilder's smoothing
        df['atr'] = tr.ewm(alpha=1.0/period, adjust=False).mean()

        # Fallback if all NaN (shouldn't happen with ewm)
        if df['atr'].isna().all():
            df['atr'] = tr.rolling(period).mean()

        return df

    def _get_timestamp(self, df: pd.DataFrame, idx: int) -> pd.Timestamp:
        """Get timestamp at index, handling different index types."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index[idx]
        elif 'time' in df.columns:
            return pd.Timestamp(df.iloc[idx]['time'])
        elif 'timestamp' in df.columns:
            return pd.Timestamp(df.iloc[idx]['timestamp'])
        else:
            return pd.Timestamp.now()

    def _detect_rolling(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Simple rolling window high/low detection.

        A swing high is the highest point in a window of 2*lookback+1 bars.
        A swing low is the lowest point in a window of 2*lookback+1 bars.
        """
        swings = []
        lookback = self.config.lookback
        min_atr = self.config.min_swing_atr

        highs = df['high'].values
        lows = df['low'].values
        atr = df['atr'].values

        for i in range(lookback, len(df) - lookback):
            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            window_start = i - lookback
            window_end = i + lookback + 1

            # Swing high check
            if highs[i] == highs[window_start:window_end].max():
                # Minimum swing size check
                prev_low = lows[max(0, i-lookback):i].min()
                swing_size = highs[i] - prev_low

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(highs[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='HIGH',
                        strength=1.0
                    ))

            # Swing low check
            if lows[i] == lows[window_start:window_end].min():
                # Minimum swing size check
                prev_high = highs[max(0, i-lookback):i].max()
                swing_size = prev_high - lows[i]

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(lows[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='LOW',
                        strength=1.0
                    ))

        return sorted(swings, key=lambda x: x.index)

    def _detect_savgol(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Savitzky-Golay smoothed detection.

        Applies smoothing filter before finding local extrema.
        Reduces noise-induced false swings.
        """
        swings = []
        window = self.config.smoothing_window
        lookback = self.config.lookback
        min_atr = self.config.min_swing_atr

        # Window must be odd
        if window % 2 == 0:
            window += 1

        if len(df) < window:
            return []

        # Smooth the data
        smoothed_high = savgol_filter(df['high'].values, window, 2)
        smoothed_low = savgol_filter(df['low'].values, window, 2)

        # Find local extrema on smoothed data
        high_indices = argrelextrema(smoothed_high, np.greater, order=lookback)[0]
        low_indices = argrelextrema(smoothed_low, np.less, order=lookback)[0]

        atr = df['atr'].values

        # Process swing highs
        for i in high_indices:
            if i < lookback or i >= len(df):
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            actual_high = df['high'].iloc[i]
            prev_low = df['low'].iloc[max(0, i-lookback):i].min()
            swing_size = actual_high - prev_low

            if swing_size >= min_atr * current_atr:
                swings.append(SwingPoint(
                    index=i,
                    price=float(actual_high),
                    timestamp=self._get_timestamp(df, i),
                    swing_type='HIGH',
                    strength=0.9  # Slightly lower than rolling
                ))

        # Process swing lows
        for i in low_indices:
            if i < lookback or i >= len(df):
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            actual_low = df['low'].iloc[i]
            prev_high = df['high'].iloc[max(0, i-lookback):i].max()
            swing_size = prev_high - actual_low

            if swing_size >= min_atr * current_atr:
                swings.append(SwingPoint(
                    index=i,
                    price=float(actual_low),
                    timestamp=self._get_timestamp(df, i),
                    swing_type='LOW',
                    strength=0.9
                ))

        return sorted(swings, key=lambda x: x.index)

    def _detect_fractal(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Williams Fractal detection (5-bar pattern).

        Fractal high: Bar with highest high surrounded by 2 lower highs on each side.
        Fractal low: Bar with lowest low surrounded by 2 higher lows on each side.
        """
        swings = []
        n = 2  # Williams uses 2 bars on each side
        min_atr = self.config.min_swing_atr

        highs = df['high'].values
        lows = df['low'].values
        atr = df['atr'].values

        for i in range(n, len(df) - n):
            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            # Fractal high: H[i] > H[i-1], H[i-2], H[i+1], H[i+2]
            is_fractal_high = (
                highs[i] > highs[i-1] and
                highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and
                highs[i] > highs[i+2]
            )

            if is_fractal_high:
                prev_low = lows[max(0, i-5):i].min()
                swing_size = highs[i] - prev_low

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(highs[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='HIGH',
                        strength=0.85
                    ))

            # Fractal low: L[i] < L[i-1], L[i-2], L[i+1], L[i+2]
            is_fractal_low = (
                lows[i] < lows[i-1] and
                lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and
                lows[i] < lows[i+2]
            )

            if is_fractal_low:
                prev_high = highs[max(0, i-5):i].max()
                swing_size = prev_high - lows[i]

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(lows[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='LOW',
                        strength=0.85
                    ))

        return sorted(swings, key=lambda x: x.index)

    def _detect_wavelet(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Multi-scale wavelet detection.

        Uses continuous wavelet transform to find swings confirmed
        across multiple scales. Requires PyWavelets.
        """
        if not HAS_WAVELET:
            raise ImportError(
                "PyWavelets required for wavelet detection. "
                "Install with: pip install PyWavelets"
            )

        swings = []
        scales = self.config.wavelet_scales
        min_atr = self.config.min_swing_atr

        prices = df['close'].values
        atr = df['atr'].values

        # Find extrema at each scale
        scale_swings = {s: {'highs': set(), 'lows': set()} for s in scales}

        for scale in scales:
            coef, _ = pywt.cwt(prices, [scale], 'morl')
            smoothed = coef[0]

            high_idx = argrelextrema(smoothed, np.greater, order=scale)[0]
            low_idx = argrelextrema(smoothed, np.less, order=scale)[0]

            scale_swings[scale]['highs'].update(high_idx)
            scale_swings[scale]['lows'].update(low_idx)

        # Require confirmation from multiple scales
        min_scales = len(scales) // 2 + 1

        # Collect all candidates
        all_high_candidates = set()
        all_low_candidates = set()
        for scale in scales:
            all_high_candidates.update(scale_swings[scale]['highs'])
            all_low_candidates.update(scale_swings[scale]['lows'])

        # Process high candidates
        for i in all_high_candidates:
            if i >= len(df):
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            # Count scale confirmations (with tolerance)
            confirmations = sum(
                1 for s in scales
                if i in scale_swings[s]['highs'] or
                any(abs(i - j) <= 2 for j in scale_swings[s]['highs'])
            )

            if confirmations >= min_scales:
                strength = confirmations / len(scales)

                prev_low = df['low'].iloc[max(0, i-10):i].min() if i > 0 else df['low'].iloc[i]
                swing_size = df['high'].iloc[i] - prev_low

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(df['high'].iloc[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='HIGH',
                        strength=strength
                    ))

        # Process low candidates
        for i in all_low_candidates:
            if i >= len(df):
                continue

            current_atr = atr[i]
            if current_atr <= 0 or np.isnan(current_atr):
                continue

            confirmations = sum(
                1 for s in scales
                if i in scale_swings[s]['lows'] or
                any(abs(i - j) <= 2 for j in scale_swings[s]['lows'])
            )

            if confirmations >= min_scales:
                strength = confirmations / len(scales)

                prev_high = df['high'].iloc[max(0, i-10):i].max() if i > 0 else df['high'].iloc[i]
                swing_size = prev_high - df['low'].iloc[i]

                if swing_size >= min_atr * current_atr:
                    swings.append(SwingPoint(
                        index=i,
                        price=float(df['low'].iloc[i]),
                        timestamp=self._get_timestamp(df, i),
                        swing_type='LOW',
                        strength=strength
                    ))

        return sorted(swings, key=lambda x: x.index)
