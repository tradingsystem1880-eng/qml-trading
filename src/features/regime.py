"""
Market Regime Classifier for QML Trading System
================================================
Classifies market conditions into distinct regimes
for adaptive strategy behavior.

Regimes:
- Trending Up / Trending Down
- Ranging / Consolidating
- High Volatility / Low Volatility
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.indicators import (
    calculate_atr,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_ema,
)


class TrendRegime(str, Enum):
    """Trend regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MarketRegime(str, Enum):
    """Combined market regime."""
    TRENDING_VOLATILE = "trending_volatile"
    TRENDING_CALM = "trending_calm"
    RANGING_VOLATILE = "ranging_volatile"
    RANGING_CALM = "ranging_calm"


@dataclass
class RegimeConfig:
    """Configuration for regime classification."""
    
    # ADX thresholds for trend strength
    strong_trend_adx: float = 40.0
    weak_trend_adx: float = 20.0
    
    # Volatility percentile thresholds
    high_vol_percentile: float = 75.0
    low_vol_percentile: float = 25.0
    
    # Lookback periods
    trend_lookback: int = 20
    volatility_lookback: int = 100
    
    # Bollinger Band width threshold for ranging detection
    ranging_bb_width: float = 0.02  # 2% bandwidth indicates ranging


@dataclass
class RegimeState:
    """Current market regime state."""
    
    trend: TrendRegime = TrendRegime.NEUTRAL
    volatility: VolatilityRegime = VolatilityRegime.NORMAL
    combined: MarketRegime = MarketRegime.RANGING_CALM
    
    adx: float = 0.0
    atr_percentile: float = 50.0
    trend_direction: float = 0.0  # -1 to 1
    
    confidence: float = 0.0


class RegimeClassifier:
    """
    Classifies market regime for adaptive strategy behavior.
    
    Uses multiple indicators to determine:
    1. Trend regime (trending vs ranging)
    2. Volatility regime (high/normal/low)
    3. Combined regime for strategy adjustment
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime classifier.
        
        Args:
            config: Classification configuration
        """
        self.config = config or RegimeConfig()
    
    def classify(self, df: pd.DataFrame) -> RegimeState:
        """
        Classify current market regime.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            RegimeState with all classifications
        """
        if len(df) < self.config.volatility_lookback:
            logger.warning("Insufficient data for regime classification")
            return RegimeState()
        
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        # Calculate indicators
        adx = calculate_adx(high, low, close, 14)
        atr = calculate_atr(high, low, close, 14)
        
        # Current values (last bar)
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
        
        # Calculate trend direction
        ema_fast = calculate_ema(close, 20)
        ema_slow = calculate_ema(close, 50)
        trend_direction = self._calculate_trend_direction(close, ema_fast, ema_slow)
        
        # Calculate ATR percentile
        atr_percentile = self._calculate_atr_percentile(atr)
        
        # Classify trend
        trend_regime = self._classify_trend(current_adx, trend_direction)
        
        # Classify volatility
        volatility_regime = self._classify_volatility(atr_percentile)
        
        # Combine regimes
        combined_regime = self._combine_regimes(trend_regime, volatility_regime)
        
        # Calculate confidence
        confidence = self._calculate_confidence(current_adx, atr_percentile)
        
        return RegimeState(
            trend=trend_regime,
            volatility=volatility_regime,
            combined=combined_regime,
            adx=current_adx,
            atr_percentile=atr_percentile,
            trend_direction=trend_direction,
            confidence=confidence
        )
    
    def _calculate_trend_direction(
        self,
        close: np.ndarray,
        ema_fast: np.ndarray,
        ema_slow: np.ndarray
    ) -> float:
        """
        Calculate trend direction (-1 to 1).
        
        Returns:
            Trend direction: -1 (strong down) to 1 (strong up)
        """
        # EMA relationship
        if ema_fast[-1] > ema_slow[-1]:
            ema_signal = (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]
        else:
            ema_signal = (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]
        
        # Price momentum
        lookback = min(20, len(close) - 1)
        price_change = (close[-1] - close[-lookback - 1]) / close[-lookback - 1]
        
        # Combine signals
        direction = np.clip(ema_signal * 20 + price_change * 5, -1, 1)
        
        return float(direction)
    
    def _calculate_atr_percentile(self, atr: np.ndarray) -> float:
        """Calculate current ATR as percentile of recent history."""
        lookback = self.config.volatility_lookback
        
        if len(atr) < lookback:
            return 50.0
        
        current_atr = atr[-1]
        historical_atr = atr[-lookback:-1]
        
        # Remove NaN values
        historical_atr = historical_atr[~np.isnan(historical_atr)]
        
        if len(historical_atr) == 0:
            return 50.0
        
        percentile = (np.sum(historical_atr < current_atr) / len(historical_atr)) * 100
        
        return float(percentile)
    
    def _classify_trend(
        self,
        adx: float,
        trend_direction: float
    ) -> TrendRegime:
        """Classify trend regime based on ADX and direction."""
        
        # Check for strong trend
        if adx >= self.config.strong_trend_adx:
            if trend_direction > 0.3:
                return TrendRegime.STRONG_UPTREND
            elif trend_direction < -0.3:
                return TrendRegime.STRONG_DOWNTREND
            else:
                # Strong ADX but unclear direction
                return TrendRegime.NEUTRAL
        
        # Check for moderate trend
        if adx >= self.config.weak_trend_adx:
            if trend_direction > 0.2:
                return TrendRegime.UPTREND
            elif trend_direction < -0.2:
                return TrendRegime.DOWNTREND
            else:
                return TrendRegime.NEUTRAL
        
        # Weak ADX = ranging/neutral
        return TrendRegime.NEUTRAL
    
    def _classify_volatility(self, atr_percentile: float) -> VolatilityRegime:
        """Classify volatility regime based on ATR percentile."""
        
        if atr_percentile >= self.config.high_vol_percentile:
            return VolatilityRegime.HIGH
        elif atr_percentile <= self.config.low_vol_percentile:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL
    
    def _combine_regimes(
        self,
        trend: TrendRegime,
        volatility: VolatilityRegime
    ) -> MarketRegime:
        """Combine trend and volatility into a single regime."""
        
        is_trending = trend in [
            TrendRegime.STRONG_UPTREND,
            TrendRegime.STRONG_DOWNTREND,
            TrendRegime.UPTREND,
            TrendRegime.DOWNTREND
        ]
        
        is_volatile = volatility == VolatilityRegime.HIGH
        
        if is_trending and is_volatile:
            return MarketRegime.TRENDING_VOLATILE
        elif is_trending and not is_volatile:
            return MarketRegime.TRENDING_CALM
        elif not is_trending and is_volatile:
            return MarketRegime.RANGING_VOLATILE
        else:
            return MarketRegime.RANGING_CALM
    
    def _calculate_confidence(
        self,
        adx: float,
        atr_percentile: float
    ) -> float:
        """
        Calculate classification confidence.
        
        Higher confidence when:
        - ADX is very high or very low (clear regime)
        - ATR percentile is extreme (clear volatility regime)
        """
        # ADX confidence (how clearly trending or ranging)
        if adx > 40:
            adx_confidence = 0.9
        elif adx > 30:
            adx_confidence = 0.7
        elif adx < 15:
            adx_confidence = 0.8  # Clearly ranging
        else:
            adx_confidence = 0.5
        
        # Volatility confidence
        if atr_percentile > 80 or atr_percentile < 20:
            vol_confidence = 0.9
        elif atr_percentile > 70 or atr_percentile < 30:
            vol_confidence = 0.7
        else:
            vol_confidence = 0.5
        
        return (adx_confidence + vol_confidence) / 2
    
    def get_regime_features(self, regime_state: RegimeState) -> Dict[str, float]:
        """
        Convert regime state to feature dictionary for ML.
        
        Args:
            regime_state: Current regime state
            
        Returns:
            Dictionary of regime features
        """
        return {
            "regime_adx": regime_state.adx,
            "regime_atr_percentile": regime_state.atr_percentile,
            "regime_trend_direction": regime_state.trend_direction,
            "regime_confidence": regime_state.confidence,
            "regime_is_trending": 1.0 if regime_state.trend not in [TrendRegime.NEUTRAL] else 0.0,
            "regime_is_volatile": 1.0 if regime_state.volatility == VolatilityRegime.HIGH else 0.0,
        }


def classify_regime(df: pd.DataFrame, config: Optional[RegimeConfig] = None) -> RegimeState:
    """
    Convenience function to classify market regime.
    
    Args:
        df: OHLCV DataFrame
        config: Optional configuration
        
    Returns:
        RegimeState
    """
    classifier = RegimeClassifier(config=config)
    return classifier.classify(df)

