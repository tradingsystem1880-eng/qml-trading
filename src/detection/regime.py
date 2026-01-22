"""
Market Regime Detection for adaptive QML filtering.

QML patterns work best in RANGING markets, poorly in strong trends.
This module provides regime classification to filter detection.

Based on DeepSeek recommendations for adaptive pattern detection.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "TRENDING"      # Strong directional move - AVOID QML
    VOLATILE = "VOLATILE"      # High vol, no direction - CAUTION
    RANGING = "RANGING"        # Ideal for QML patterns
    EXTREME = "EXTREME"        # Overbought/oversold - CAUTION


@dataclass
class RegimeResult:
    """Result of market regime analysis."""
    regime: MarketRegime
    adx: float
    volatility_percentile: float
    rsi: float
    confidence: float

    def is_favorable_for_qml(self) -> bool:
        """Check if current regime is favorable for QML patterns."""
        return self.regime == MarketRegime.RANGING


class MarketRegimeDetector:
    """
    Detect market regime for adaptive pattern filtering.

    Uses ADX, RSI, and volatility percentile to classify regime.
    """

    def __init__(
        self,
        adx_period: int = 14,
        rsi_period: int = 14,
        vol_lookback: int = 100
    ):
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.vol_lookback = vol_lookback

    def get_regime(self, df: pd.DataFrame) -> RegimeResult:
        """
        Identify current market regime.

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close' columns

        Returns:
            RegimeResult with classification and indicators
        """
        min_bars = max(self.adx_period, self.rsi_period, self.vol_lookback) + 10

        if len(df) < min_bars:
            # Insufficient data - return neutral regime
            return RegimeResult(
                regime=MarketRegime.RANGING,
                adx=0.0,
                volatility_percentile=0.5,
                rsi=50.0,
                confidence=0.5
            )

        adx = self._calculate_adx(df)
        rsi = self._calculate_rsi(df)
        vol_percentile = self._calculate_volatility_percentile(df)

        regime = self._classify_regime(adx, rsi, vol_percentile)
        confidence = self._calculate_confidence(adx, rsi, vol_percentile, regime)

        return RegimeResult(
            regime=regime,
            adx=adx,
            volatility_percentile=vol_percentile,
            rsi=rsi,
            confidence=confidence
        )

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """
        Calculate ADX (Average Directional Index).

        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend / ranging
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / self.adx_period

        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        result = float(adx.iloc[-1])
        return result if not np.isnan(result) else 0.0

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """
        Calculate RSI (Relative Strength Index).

        RSI measures momentum:
        - RSI > 70: Overbought
        - RSI < 30: Oversold
        - RSI 40-60: Neutral
        """
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing
        alpha = 1.0 / self.rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        result = float(rsi.iloc[-1])
        return result if not np.isnan(result) else 50.0

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """
        Calculate current volatility percentile vs historical.

        Uses 20-period rolling standard deviation of returns.
        """
        returns = df['close'].pct_change().dropna()

        if len(returns) < 20:
            return 0.5

        # Current volatility (last 20 periods)
        current_vol = returns.iloc[-20:].std()

        # Historical volatility distribution
        historical_vol = returns.rolling(20).std().dropna()

        if len(historical_vol) == 0:
            return 0.5

        # Percentile rank
        percentile = float((historical_vol < current_vol).sum() / len(historical_vol))
        return percentile

    def _classify_regime(
        self,
        adx: float,
        rsi: float,
        vol_percentile: float
    ) -> MarketRegime:
        """
        Classify market regime based on indicators.

        Priority order:
        1. TRENDING: Strong ADX with moderate volatility
        2. VOLATILE: High volatility with weak ADX
        3. EXTREME: RSI at extremes
        4. RANGING: Default (ideal for QML)
        """
        # Strong trend takes priority
        if adx > 25 and vol_percentile < 0.75:
            return MarketRegime.TRENDING

        # High volatility without direction
        if vol_percentile > 0.75 and adx < 25:
            return MarketRegime.VOLATILE

        # Overbought/oversold
        if rsi > 70 or rsi < 30:
            return MarketRegime.EXTREME

        # Default: ranging market (best for QML)
        return MarketRegime.RANGING

    def _calculate_confidence(
        self,
        adx: float,
        rsi: float,
        vol_percentile: float,
        regime: MarketRegime
    ) -> float:
        """
        Calculate confidence in regime classification.

        Returns value between 0 and 1.
        """
        if regime == MarketRegime.TRENDING:
            # Higher ADX = more confident
            return min(1.0, adx / 50.0)

        elif regime == MarketRegime.VOLATILE:
            # Higher vol percentile = more confident
            return vol_percentile

        elif regime == MarketRegime.EXTREME:
            # Further from 50 = more confident
            return abs(rsi - 50) / 50.0

        else:  # RANGING
            # Low ADX and neutral volatility = more confident
            adx_factor = 1.0 - min(1.0, adx / 25.0)
            vol_factor = 1.0 - abs(vol_percentile - 0.5)
            return (adx_factor + vol_factor) / 2.0
