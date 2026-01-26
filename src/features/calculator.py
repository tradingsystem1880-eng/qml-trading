"""
Feature Calculator for QML Patterns - Phase 3
==============================================
Uses pandas-ta for all technical indicators (90% rule).

Features are organized into tiers:
- Tier 1: Pattern Geometry (MUST HAVE) - 6 features (CUSTOM - QML-specific)
- Tier 2: Market Context (HIGH PRIORITY) - 5 features
- Tier 3: Volume (HIGH PRIORITY) - 3 features
- Tier 4: Pattern Quality (MEDIUM) - 2 features

Total: 16 features per pattern
"""

import numpy as np
import pandas as pd
from scipy import stats

# pandas_ta is optional - only needed for advanced feature calculation
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    ta = None
    HAS_PANDAS_TA = False
from datetime import datetime
from typing import Optional

from src.data.schemas import PatternDetection, FeatureVector, generate_id


class FeatureCalculator:
    """
    Calculates all features for a detected QML pattern.

    Uses pandas-ta for technical indicators (ATR, RSI, ADX, EMA, etc.)
    Only Tier 1 Pattern Geometry is custom (QML-specific calculations).

    Usage:
        calculator = FeatureCalculator(ohlcv_df)
        features = calculator.calculate_features(pattern)
    """

    def __init__(self, ohlcv_data: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            ohlcv_data: DataFrame with columns [time, open, high, low, close, volume]
        """
        self.data = ohlcv_data.copy()
        self._validate_data()
        self._calculate_indicators()

    def _validate_data(self) -> None:
        """Validate OHLCV data has required columns."""
        # Normalize column names (handle capitalized columns)
        col_map = {
            'Time': 'time', 'timestamp': 'time', 'datetime': 'time',
            'Open': 'open', 'OPEN': 'open',
            'High': 'high', 'HIGH': 'high',
            'Low': 'low', 'LOW': 'low',
            'Close': 'close', 'CLOSE': 'close',
            'Volume': 'volume', 'VOLUME': 'volume',
        }
        self.data = self.data.rename(columns=col_map)

        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['time']):
            self.data['time'] = pd.to_datetime(self.data['time'])

        # Sort by time and reset index
        self.data = self.data.sort_values('time').reset_index(drop=True)

    def _calculate_indicators(self) -> None:
        """
        Pre-calculate technical indicators using pandas-ta.

        Library mapping (90% rule):
        - ATR: ta.atr()
        - RSI: ta.rsi()
        - ADX: ta.adx()
        - EMA: ta.ema()
        - SMA: ta.sma()
        - Bollinger Bands: ta.bbands()
        - MACD: ta.macd()
        - OBV: ta.obv()
        - ATR Percentile: scipy.stats.percentileofscore()
        - Volume Trend: scipy.stats.linregress()
        """
        if not HAS_PANDAS_TA:
            # Fill with default values when pandas_ta not available
            self.data['atr'] = (self.data['high'] - self.data['low']).rolling(14).mean()
            self.data['rsi'] = 50.0
            self.data['adx'] = 25.0
            self.data['dmp'] = 25.0
            self.data['dmn'] = 25.0
            self.data['ema_20'] = self.data['close'].ewm(span=20).mean()
            self.data['ema_50'] = self.data['close'].ewm(span=50).mean()
            self.data['ema_200'] = self.data['close'].ewm(span=200).mean()
            self.data['volume_sma'] = self.data['volume'].rolling(20).mean()
            self.data['bb_upper'] = self.data['close'].rolling(20).mean() + 2 * self.data['close'].rolling(20).std()
            self.data['bb_middle'] = self.data['close'].rolling(20).mean()
            self.data['bb_lower'] = self.data['close'].rolling(20).mean() - 2 * self.data['close'].rolling(20).std()
            self.data['macd'] = 0.0
            self.data['macd_signal'] = 0.0
            self.data['macd_hist'] = 0.0
            self.data['obv'] = 0.0
            return

        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        volume = self.data['volume']

        # ATR (14-period) - pandas-ta
        self.data['atr'] = ta.atr(high, low, close, length=14)

        # RSI (14-period) - pandas-ta
        self.data['rsi'] = ta.rsi(close, length=14)
        self.data['rsi'] = self.data['rsi'].fillna(50)

        # ADX (14-period) - pandas-ta returns DataFrame with ADX, DMP, DMN
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None:
            self.data['adx'] = adx_df['ADX_14']
            self.data['dmp'] = adx_df['DMP_14']  # +DI
            self.data['dmn'] = adx_df['DMN_14']  # -DI
        else:
            self.data['adx'] = 25.0  # Default neutral
            self.data['dmp'] = 25.0
            self.data['dmn'] = 25.0

        # EMAs - pandas-ta
        self.data['ema_20'] = ta.ema(close, length=20)
        self.data['ema_50'] = ta.ema(close, length=50)
        self.data['ema_200'] = ta.ema(close, length=200)

        # SMA for volume - pandas-ta
        self.data['volume_sma'] = ta.sma(volume, length=20)

        # Bollinger Bands - pandas-ta
        # Note: Column names vary by version (BBU_20_2.0 or BBU_20_2.0_2.0)
        bbands = ta.bbands(close, length=20, std=2.0)
        if bbands is not None:
            # Find the actual column names (handle version differences)
            bb_cols = bbands.columns.tolist()
            bbu_col = [c for c in bb_cols if c.startswith('BBU')][0]
            bbm_col = [c for c in bb_cols if c.startswith('BBM')][0]
            bbl_col = [c for c in bb_cols if c.startswith('BBL')][0]

            self.data['bb_upper'] = bbands[bbu_col]
            self.data['bb_middle'] = bbands[bbm_col]
            self.data['bb_lower'] = bbands[bbl_col]
            self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']

        # MACD - pandas-ta
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None:
            self.data['macd'] = macd_df['MACD_12_26_9']
            self.data['macd_signal'] = macd_df['MACDs_12_26_9']
            self.data['macd_hist'] = macd_df['MACDh_12_26_9']

        # OBV - pandas-ta
        self.data['obv'] = ta.obv(close, volume)

        # ATR Percentile (rolling 100) - scipy.stats.percentileofscore
        self.data['atr_percentile'] = self._calc_rolling_percentile(
            self.data['atr'], window=100
        )

    def _calc_rolling_percentile(self, series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate rolling percentile using scipy.stats.percentileofscore.

        Args:
            series: Input series
            window: Lookback window

        Returns:
            Series with percentile values (0-1 scale)
        """
        def percentile_func(x):
            if len(x) < 2 or pd.isna(x.iloc[-1]):
                return 0.5
            return stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100.0

        result = series.rolling(window, min_periods=10).apply(
            percentile_func, raw=False
        )
        return result.fillna(0.5)

    def calculate_features(self, pattern: PatternDetection) -> FeatureVector:
        """
        Calculate all features for a pattern.

        Args:
            pattern: PatternDetection object with P1-P5 points

        Returns:
            FeatureVector with all calculated features
        """
        # Get ATR at pattern time for normalization
        atr = self._get_atr_at_time(pattern.p5_time)
        if atr == 0 or np.isnan(atr):
            atr = self.data['atr'].mean()  # Fallback

        features = FeatureVector(
            pattern_id=pattern.id,
            calculation_time=datetime.now(),

            # Tier 1: Pattern Geometry (CUSTOM - QML-specific)
            head_extension_atr=self._calc_head_extension(pattern, atr),
            bos_depth_atr=self._calc_bos_depth(pattern, atr),
            shoulder_symmetry=self._calc_shoulder_symmetry(pattern, atr),
            amplitude_ratio=self._calc_amplitude_ratio(pattern),
            time_ratio=self._calc_time_ratio(pattern),
            fib_retracement_p5=self._calc_fib_retracement(pattern),

            # Tier 2: Market Context
            htf_trend_alignment=self._calc_trend_alignment(pattern),
            distance_to_sr_atr=self._calc_distance_to_sr(pattern, atr),
            volatility_percentile=self._calc_volatility_percentile(pattern),
            regime_state=self._calc_regime_state(pattern),
            rsi_divergence=self._calc_rsi_divergence(pattern),

            # Tier 3: Volume
            volume_spike_p3=self._calc_volume_spike(pattern.p3_time),
            volume_spike_p4=self._calc_volume_spike(pattern.p4_time),
            volume_trend_p1_p5=self._calc_volume_trend(pattern),

            # Tier 4: Pattern Quality
            noise_ratio=self._calc_noise_ratio(pattern),
            bos_candle_strength=self._calc_bos_strength(pattern),
        )

        return features

    # =========================================================================
    # TIER 1: Pattern Geometry (6 features) - CUSTOM (QML-specific)
    # =========================================================================

    def _calc_head_extension(self, p: PatternDetection, atr: float) -> float:
        """
        How much head (P3) extends beyond left shoulder (P1).

        For BULLISH: P3 is lower than P1 (head extends down)
        For BEARISH: P3 is higher than P1 (head extends up)
        """
        try:
            if p.direction == 'BULLISH':
                return (p.p1_price - p.p3_price) / atr
            else:
                return (p.p3_price - p.p1_price) / atr
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _calc_bos_depth(self, p: PatternDetection, atr: float) -> float:
        """
        Depth of break of structure (P2 to P4).

        For BULLISH: P4 breaks below P2 (lower low)
        For BEARISH: P4 breaks above P2 (higher high)
        """
        try:
            if p.direction == 'BULLISH':
                return (p.p2_price - p.p4_price) / atr
            else:
                return (p.p4_price - p.p2_price) / atr
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _calc_shoulder_symmetry(self, p: PatternDetection, atr: float) -> float:
        """
        How close right shoulder (P5) is to left shoulder (P1).
        Lower = more symmetric.
        """
        try:
            return abs(p.p5_price - p.p1_price) / atr
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _calc_amplitude_ratio(self, p: PatternDetection) -> float:
        """
        Ratio of first leg amplitude to second leg amplitude.
        Leg 1: P1 → P2
        Leg 2: P3 → P4
        """
        try:
            leg1 = abs(p.p1_price - p.p2_price)
            leg2 = abs(p.p3_price - p.p4_price)
            return leg1 / leg2 if leg2 != 0 else 0.0
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _calc_time_ratio(self, p: PatternDetection) -> float:
        """
        Temporal symmetry: bars P1→P3 vs P3→P5.
        Ideal is around 1.0 (symmetric timing).
        """
        try:
            bars_p1_p3 = self._bars_between(p.p1_time, p.p3_time)
            bars_p3_p5 = self._bars_between(p.p3_time, p.p5_time)
            return bars_p1_p3 / bars_p3_p5 if bars_p3_p5 != 0 else 0.0
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _calc_fib_retracement(self, p: PatternDetection) -> float:
        """
        Where P5 falls on P3→P4 Fibonacci retracement.
        0.618 or 0.786 are ideal QML retracement levels.
        """
        try:
            if p.direction == 'BULLISH':
                total_move = p.p4_price - p.p3_price
                p5_retracement = p.p4_price - p.p5_price
            else:
                total_move = p.p3_price - p.p4_price
                p5_retracement = p.p5_price - p.p4_price
            return p5_retracement / total_move if total_move != 0 else 0.0
        except (ZeroDivisionError, TypeError):
            return 0.0

    # =========================================================================
    # TIER 2: Market Context (5 features)
    # =========================================================================

    def _calc_trend_alignment(self, p: PatternDetection) -> float:
        """
        Higher timeframe trend alignment (-1 to 1).
        +1 = pattern aligned with trend
        -1 = pattern against trend

        Uses EMA 50 from pandas-ta.
        """
        try:
            idx = self._get_index_at_time(p.p5_time)
            if idx < 50:
                return 0.0

            # Compare current EMA to EMA 50 bars ago
            ema_now = self.data['ema_50'].iloc[idx]
            ema_prev = self.data['ema_50'].iloc[max(0, idx - 50)]

            # Determine trend direction
            trend = 1.0 if ema_now > ema_prev else -1.0

            # Alignment: bullish pattern in uptrend = 1, bearish in downtrend = 1
            if p.direction == 'BULLISH':
                return trend
            else:
                return -trend
        except Exception:
            return 0.0

    def _calc_distance_to_sr(self, p: PatternDetection, atr: float) -> float:
        """
        Distance to nearest support/resistance level in ATR.
        Uses recent swing highs/lows as S/R proxy.
        """
        try:
            idx = self._get_index_at_time(p.p5_time)
            lookback = min(100, idx)
            if lookback < 10:
                return 0.0

            recent_high = self.data['high'].iloc[idx - lookback:idx].max()
            recent_low = self.data['low'].iloc[idx - lookback:idx].min()

            if p.direction == 'BULLISH':
                # Distance to resistance (upside target)
                distance = recent_high - p.entry_price
            else:
                # Distance to support (downside target)
                distance = p.entry_price - recent_low

            return distance / atr if atr != 0 else 0.0
        except Exception:
            return 0.0

    def _calc_volatility_percentile(self, p: PatternDetection) -> float:
        """ATR percentile over last 100 periods (0-1). Uses scipy."""
        try:
            idx = self._get_index_at_time(p.p5_time)
            return float(self.data['atr_percentile'].iloc[idx])
        except Exception:
            return 0.5

    def _calc_regime_state(self, p: PatternDetection) -> str:
        """
        Market regime classification using ADX from pandas-ta.
        Returns: 'TRENDING', 'RANGING', or 'VOLATILE'
        """
        try:
            idx = self._get_index_at_time(p.p5_time)

            # Use ADX for trend strength
            adx = self.data['adx'].iloc[idx] if 'adx' in self.data.columns else 25
            vol_percentile = self._calc_volatility_percentile(p)

            if vol_percentile > 0.8:
                return 'VOLATILE'
            elif adx > 25:  # ADX > 25 indicates trending
                return 'TRENDING'
            else:
                return 'RANGING'
        except Exception:
            return 'UNKNOWN'

    def _calc_rsi_divergence(self, p: PatternDetection) -> float:
        """
        RSI divergence at head (P3) relative to P1.
        Uses RSI from pandas-ta.
        Returns 1.0 for divergence, 0.0 for no divergence.
        """
        try:
            idx_p1 = self._get_index_at_time(p.p1_time)
            idx_p3 = self._get_index_at_time(p.p3_time)

            rsi_p1 = self.data['rsi'].iloc[idx_p1]
            rsi_p3 = self.data['rsi'].iloc[idx_p3]

            if p.direction == 'BULLISH':
                # Bullish divergence: price makes lower low, RSI makes higher low
                price_lower = p.p3_price < p.p1_price
                rsi_higher = rsi_p3 > rsi_p1
                return 1.0 if price_lower and rsi_higher else 0.0
            else:
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_higher = p.p3_price > p.p1_price
                rsi_lower = rsi_p3 < rsi_p1
                return 1.0 if price_higher and rsi_lower else 0.0
        except Exception:
            return 0.0

    # =========================================================================
    # TIER 3: Volume (3 features)
    # =========================================================================

    def _calc_volume_spike(self, time: datetime) -> float:
        """
        Volume at time relative to 20-period SMA.
        >1 indicates above-average volume.
        Uses SMA from pandas-ta.
        """
        try:
            idx = self._get_index_at_time(time)
            vol = self.data['volume'].iloc[idx]
            vol_sma = self.data['volume_sma'].iloc[idx]
            return vol / vol_sma if vol_sma != 0 else 1.0
        except Exception:
            return 1.0

    def _calc_volume_trend(self, p: PatternDetection) -> float:
        """
        Volume trend from P1 to P5 using scipy.stats.linregress.
        Positive = increasing volume, Negative = decreasing.
        Normalized by mean volume.
        """
        try:
            idx_p1 = self._get_index_at_time(p.p1_time)
            idx_p5 = self._get_index_at_time(p.p5_time)

            volumes = self.data['volume'].iloc[idx_p1:idx_p5 + 1].values
            if len(volumes) < 2:
                return 0.0

            # Use scipy.stats.linregress for slope
            x = np.arange(len(volumes))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, volumes)
            mean_vol = volumes.mean()

            return slope / mean_vol if mean_vol != 0 else 0.0
        except Exception:
            return 0.0

    # =========================================================================
    # TIER 4: Pattern Quality (2 features)
    # =========================================================================

    def _calc_noise_ratio(self, p: PatternDetection) -> float:
        """
        Ratio of wicks to bodies within pattern.
        Lower = cleaner pattern with less noise.
        """
        try:
            idx_p1 = self._get_index_at_time(p.p1_time)
            idx_p5 = self._get_index_at_time(p.p5_time)

            subset = self.data.iloc[idx_p1:idx_p5 + 1]
            bodies = abs(subset['close'] - subset['open']).sum()
            total_range = (subset['high'] - subset['low']).sum()
            wicks = total_range - bodies

            return wicks / bodies if bodies != 0 else 1.0
        except Exception:
            return 1.0

    def _calc_bos_strength(self, p: PatternDetection) -> float:
        """
        Break of structure candle (P4) body relative to its range.
        Higher = stronger/cleaner BoS candle.
        """
        try:
            idx_p4 = self._get_index_at_time(p.p4_time)
            candle = self.data.iloc[idx_p4]

            body = abs(candle['close'] - candle['open'])
            range_ = candle['high'] - candle['low']

            return body / range_ if range_ != 0 else 0.0
        except Exception:
            return 0.0

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_index_at_time(self, time: datetime) -> int:
        """Get DataFrame index for a timestamp."""
        try:
            # Remove timezone if present for comparison
            if hasattr(time, 'tzinfo') and time.tzinfo is not None:
                time = time.replace(tzinfo=None)

            # Find nearest index
            time_col = self.data['time']
            if hasattr(time_col.iloc[0], 'tzinfo') and time_col.iloc[0].tzinfo is not None:
                time_col = time_col.dt.tz_localize(None)

            idx = (time_col - time).abs().argmin()
            return idx
        except Exception:
            return 0

    def _get_atr_at_time(self, time: datetime) -> float:
        """Get ATR value at a specific time."""
        try:
            idx = self._get_index_at_time(time)
            return float(self.data['atr'].iloc[idx])
        except Exception:
            return self.data['atr'].mean()

    def _bars_between(self, start: datetime, end: datetime) -> int:
        """Count bars between two timestamps."""
        try:
            idx_start = self._get_index_at_time(start)
            idx_end = self._get_index_at_time(end)
            return abs(idx_end - idx_start)
        except Exception:
            return 1

    def get_indicator_at_time(self, indicator: str, time: datetime) -> float:
        """
        Get any pre-calculated indicator value at a specific time.

        Args:
            indicator: Column name (e.g., 'atr', 'rsi', 'adx', 'ema_50')
            time: Timestamp

        Returns:
            Indicator value
        """
        try:
            idx = self._get_index_at_time(time)
            if indicator in self.data.columns:
                return float(self.data[indicator].iloc[idx])
            return 0.0
        except Exception:
            return 0.0

    def get_available_indicators(self) -> list:
        """Return list of all available indicator columns."""
        indicator_cols = [
            'atr', 'rsi', 'adx', 'dmp', 'dmn',
            'ema_20', 'ema_50', 'ema_200',
            'volume_sma', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'macd', 'macd_signal', 'macd_hist', 'obv', 'atr_percentile'
        ]
        return [c for c in indicator_cols if c in self.data.columns]
