"""
Hierarchical Swing Detection
============================
3-layer swing point detection designed to reduce noise and improve pattern quality.

Layer 1: GEOMETRY
  - Find local extrema with minimum bar separation
  - Prevents detecting micro-swings

Layer 2: SIGNIFICANCE
  - Filter by ATR-normalized move size
  - Require forward confirmation (price retraces after swing)

Layer 3: CONTEXT
  - ADX-based regime detection
  - Higher thresholds in trending markets

This detector coexists with historical_detector.py:
- HierarchicalSwingDetector: Noise-resistant, ML-optimizable parameters
- HistoricalSwingDetector: Simpler argrelextrema-based detection
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.detection.historical_detector import HistoricalSwingPoint


@dataclass
class HierarchicalSwingConfig:
    """
    Configuration for 3-layer hierarchical swing detection.
    All parameters are ML-optimizable.
    """
    # Layer 1: Geometry
    min_bar_separation: int = 5  # Minimum bars between swings
    lookback: int = 5  # Bars to look back for comparison
    lookforward: int = 5  # Bars to look forward for confirmation

    # Layer 2: Significance
    min_move_atr: float = 1.0  # Minimum move size in ATR units
    forward_confirm_bars: int = 3  # Bars to check for retracement
    forward_confirm_pct: float = 0.3  # Minimum retracement (30%)

    # Layer 3: Context
    adx_threshold: float = 20.0  # ADX > this = trending market
    adx_period: int = 14  # ADX calculation period
    trending_multiplier: float = 1.5  # Increase thresholds in trends

    # ATR settings
    atr_period: int = 14

    def __post_init__(self):
        """Validate configuration."""
        if self.min_bar_separation < 1:
            raise ValueError("min_bar_separation must be >= 1")
        if self.min_move_atr <= 0:
            raise ValueError("min_move_atr must be > 0")
        if not (0 < self.forward_confirm_pct <= 1):
            raise ValueError("forward_confirm_pct must be between 0 and 1")


@dataclass
class SwingCandidate:
    """Internal representation of a swing point candidate."""
    bar_index: int
    price: float
    swing_type: str  # 'HIGH' or 'LOW'
    move_size_atr: float
    has_confirmation: bool
    regime: str  # 'TRENDING' or 'RANGING'
    adx_value: float


class HierarchicalSwingDetector:
    """
    3-layer hierarchical swing point detector.

    Designed to reduce noise and produce higher-quality swing points
    for QML pattern detection. All parameters are ML-optimizable.
    """

    def __init__(
        self,
        config: Optional[HierarchicalSwingConfig] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ):
        """
        Initialize the hierarchical detector.

        Args:
            config: Detection configuration
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        self.config = config or HierarchicalSwingConfig()
        self.symbol = symbol
        self.timeframe = timeframe

    def detect(self, df: pd.DataFrame) -> List[HistoricalSwingPoint]:
        """
        Detect swing points using 3-layer hierarchical filtering.

        Args:
            df: OHLCV DataFrame with columns [time, open, high, low, close, volume]

        Returns:
            List of HistoricalSwingPoint objects sorted by bar index
        """
        min_len = max(
            self.config.atr_period,
            self.config.adx_period,
            self.config.lookback + self.config.lookforward
        ) + 10

        if len(df) < min_len:
            return []

        # Pre-calculate indicators
        atr = self._calculate_atr(df)
        adx = self._calculate_adx(df)

        # Layer 1: Geometry - find local extrema
        raw_highs = self._find_local_highs(df, atr)
        raw_lows = self._find_local_lows(df, atr)

        # Layer 2: Significance - filter by move size and confirmation
        significant_highs = self._filter_by_significance(raw_highs, df, atr, 'HIGH')
        significant_lows = self._filter_by_significance(raw_lows, df, atr, 'LOW')

        # Layer 3: Context - apply regime-based filtering
        final_highs = self._filter_by_context(significant_highs, adx)
        final_lows = self._filter_by_context(significant_lows, adx)

        # Convert to HistoricalSwingPoint format
        time = df['time'].values if 'time' in df.columns else df.index.values
        swing_points = []

        for candidate in final_highs + final_lows:
            swing_points.append(HistoricalSwingPoint(
                bar_index=candidate.bar_index,
                price=candidate.price,
                timestamp=pd.Timestamp(time[candidate.bar_index]),
                swing_type=candidate.swing_type,
                atr_at_formation=float(atr[candidate.bar_index]),
                significance_atr=candidate.move_size_atr,
                significance_zscore=self._calculate_zscore(
                    candidate.move_size_atr,
                    final_highs + final_lows
                ),
                symbol=self.symbol,
                timeframe=self.timeframe,
            ))

        # Sort by bar index
        swing_points.sort(key=lambda x: x.bar_index)

        # Apply minimum bar separation
        swing_points = self._enforce_min_separation(swing_points)

        return swing_points

    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate ATR using Wilder's smoothing."""
        period = self.config.atr_period

        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values

        # True Range
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        high_close[0] = high_low[0]
        low_close[0] = high_low[0]

        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period - 1] = np.mean(tr[:period])

        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return atr

    def _calculate_adx(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate ADX (Average Directional Index).

        ADX measures trend strength:
        - ADX < 20: Weak trend (ranging)
        - ADX 20-40: Strong trend
        - ADX > 40: Very strong trend
        """
        period = self.config.adx_period

        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values

        # Calculate +DM and -DM
        up_move = np.diff(high, prepend=high[0])
        down_move = np.diff(-low, prepend=-low[0])

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate ATR for normalization
        atr = self._calculate_atr(df)

        # Smooth DMs
        plus_dm_smooth = self._wilder_smooth(plus_dm, period)
        minus_dm_smooth = self._wilder_smooth(minus_dm, period)

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / np.maximum(atr, 1e-10)
        minus_di = 100 * minus_dm_smooth / np.maximum(atr, 1e-10)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
        adx = self._wilder_smooth(dx, period)

        return adx

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing method."""
        smoothed = np.zeros_like(data, dtype=float)
        smoothed[period - 1] = np.mean(data[:period])

        alpha = 1.0 / period
        for i in range(period, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

        return smoothed

    def _find_local_highs(
        self,
        df: pd.DataFrame,
        atr: np.ndarray
    ) -> List[SwingCandidate]:
        """
        Layer 1: Find local high points using lookback/lookforward comparison.

        A point is a local high if it's higher than all points within
        the lookback and lookforward windows.
        """
        high = df['high'].values if 'high' in df.columns else df['High'].values

        candidates = []
        lookback = self.config.lookback
        lookforward = self.config.lookforward

        for i in range(lookback, len(df) - lookforward):
            if atr[i] <= 0:
                continue

            current = high[i]

            # Check if higher than surrounding bars
            window_start = i - lookback
            window_end = i + lookforward + 1
            window = high[window_start:window_end]

            # Must be strictly higher than all others (or equal to max)
            if current >= np.max(window):
                candidates.append(SwingCandidate(
                    bar_index=i,
                    price=float(current),
                    swing_type='HIGH',
                    move_size_atr=0.0,  # Calculated in Layer 2
                    has_confirmation=False,
                    regime='UNKNOWN',
                    adx_value=0.0,
                ))

        return candidates

    def _find_local_lows(
        self,
        df: pd.DataFrame,
        atr: np.ndarray
    ) -> List[SwingCandidate]:
        """
        Layer 1: Find local low points using lookback/lookforward comparison.
        """
        low = df['low'].values if 'low' in df.columns else df['Low'].values

        candidates = []
        lookback = self.config.lookback
        lookforward = self.config.lookforward

        for i in range(lookback, len(df) - lookforward):
            if atr[i] <= 0:
                continue

            current = low[i]

            window_start = i - lookback
            window_end = i + lookforward + 1
            window = low[window_start:window_end]

            if current <= np.min(window):
                candidates.append(SwingCandidate(
                    bar_index=i,
                    price=float(current),
                    swing_type='LOW',
                    move_size_atr=0.0,
                    has_confirmation=False,
                    regime='UNKNOWN',
                    adx_value=0.0,
                ))

        return candidates

    def _filter_by_significance(
        self,
        candidates: List[SwingCandidate],
        df: pd.DataFrame,
        atr: np.ndarray,
        swing_type: str
    ) -> List[SwingCandidate]:
        """
        Layer 2: Filter by move significance and forward confirmation.

        A swing is significant if:
        1. Move from prior swing >= min_move_atr × ATR
        2. Price retraces >= forward_confirm_pct within forward_confirm_bars
        """
        if not candidates:
            return []

        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values

        significant = []

        for i, candidate in enumerate(candidates):
            idx = candidate.bar_index
            current_atr = atr[idx]

            if current_atr <= 0:
                continue

            # Calculate move size from prior extreme
            if swing_type == 'HIGH':
                # Find lowest low in lookback window
                window_start = max(0, idx - self.config.lookback * 2)
                prior_low = np.min(low[window_start:idx])
                move_size = (candidate.price - prior_low) / current_atr
            else:
                # Find highest high in lookback window
                window_start = max(0, idx - self.config.lookback * 2)
                prior_high = np.max(high[window_start:idx])
                move_size = (prior_high - candidate.price) / current_atr

            # Check minimum move size
            if move_size < self.config.min_move_atr:
                continue

            # Check forward confirmation (retracement after swing)
            has_confirmation = self._check_forward_confirmation(
                df, candidate, swing_type
            )

            candidate.move_size_atr = move_size
            candidate.has_confirmation = has_confirmation

            # Only include if confirmed or move is very significant
            if has_confirmation or move_size >= self.config.min_move_atr * 2:
                significant.append(candidate)

        return significant

    def _check_forward_confirmation(
        self,
        df: pd.DataFrame,
        candidate: SwingCandidate,
        swing_type: str
    ) -> bool:
        """
        Check if price retraces after the swing point.

        For a HIGH: Price should drop by at least forward_confirm_pct
        For a LOW: Price should rise by at least forward_confirm_pct
        """
        idx = candidate.bar_index
        confirm_end = min(
            idx + self.config.forward_confirm_bars + 1,
            len(df)
        )

        if confirm_end <= idx:
            return False

        close = df['close'].values if 'close' in df.columns else df['Close'].values
        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values

        if swing_type == 'HIGH':
            # Check if price dropped after high
            max_drop = candidate.price - np.min(low[idx:confirm_end])
            expected_drop = (candidate.price - close[idx]) * self.config.forward_confirm_pct

            # Also check close-based drop
            close_drop = candidate.price - np.min(close[idx:confirm_end])

            return max_drop >= expected_drop or close_drop >= candidate.price * 0.005
        else:
            # Check if price rose after low
            max_rise = np.max(high[idx:confirm_end]) - candidate.price
            expected_rise = (close[idx] - candidate.price) * self.config.forward_confirm_pct

            close_rise = np.max(close[idx:confirm_end]) - candidate.price

            return max_rise >= expected_rise or close_rise >= candidate.price * 0.005

    def _filter_by_context(
        self,
        candidates: List[SwingCandidate],
        adx: np.ndarray
    ) -> List[SwingCandidate]:
        """
        Layer 3: Apply context-based filtering using ADX.

        In trending markets (ADX > threshold), require larger moves.
        """
        filtered = []

        for candidate in candidates:
            idx = candidate.bar_index
            adx_value = adx[idx]

            # Determine regime
            if adx_value > self.config.adx_threshold:
                regime = 'TRENDING'
                # In trending markets, require larger moves
                min_move = self.config.min_move_atr * self.config.trending_multiplier
            else:
                regime = 'RANGING'
                min_move = self.config.min_move_atr

            candidate.regime = regime
            candidate.adx_value = float(adx_value)

            # Apply context-aware threshold
            if candidate.move_size_atr >= min_move:
                filtered.append(candidate)

        return filtered

    def _calculate_zscore(
        self,
        move_size_atr: float,
        all_candidates: List[SwingCandidate]
    ) -> float:
        """Calculate z-score for a swing point's significance."""
        if not all_candidates:
            return 0.0

        all_moves = [c.move_size_atr for c in all_candidates if c.move_size_atr > 0]

        if not all_moves or len(all_moves) < 2:
            return 0.0

        mean_move = np.mean(all_moves)
        std_move = np.std(all_moves)

        if std_move < 0.001:
            return 0.0

        return (move_size_atr - mean_move) / std_move

    def _enforce_min_separation(
        self,
        swing_points: List[HistoricalSwingPoint]
    ) -> List[HistoricalSwingPoint]:
        """
        Enforce minimum bar separation between swing points.

        When two swings are too close, keep the more significant one.
        """
        if len(swing_points) < 2:
            return swing_points

        filtered = []
        last_kept = None

        for swing in swing_points:
            if last_kept is None:
                filtered.append(swing)
                last_kept = swing
                continue

            bars_since_last = swing.bar_index - last_kept.bar_index

            if bars_since_last >= self.config.min_bar_separation:
                filtered.append(swing)
                last_kept = swing
            else:
                # Too close - keep the more significant one
                if swing.significance_atr > last_kept.significance_atr:
                    filtered[-1] = swing
                    last_kept = swing

        return filtered

    def get_swing_stats(self, swings: List[HistoricalSwingPoint]) -> dict:
        """Get summary statistics for detected swings."""
        if not swings:
            return {
                'total': 0,
                'highs': 0,
                'lows': 0,
                'mean_significance': 0.0,
                'mean_zscore': 0.0,
            }

        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if s.is_low]

        return {
            'total': len(swings),
            'highs': len(highs),
            'lows': len(lows),
            'mean_significance': float(np.mean([s.significance_atr for s in swings])),
            'mean_zscore': float(np.mean([s.significance_zscore for s in swings])),
            'min_significance': float(min(s.significance_atr for s in swings)),
            'max_significance': float(max(s.significance_atr for s in swings)),
        }
