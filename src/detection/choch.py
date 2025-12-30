"""
Change of Character (CHoCH) Detection for QML Trading System
=============================================================
Detects the first sign of potential trend reversal when price
breaks a key structural level against the prevailing trend.

CHoCH is the critical first component of a QML pattern, signaling
that the trend may be shifting.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.models import (
    CHoCHEvent,
    MarketStructure,
    PatternType,
    SwingPoint,
    SwingType,
    TrendType,
)
from src.detection.structure import TrendState
from src.utils.indicators import calculate_atr


@dataclass
class CHoCHConfig:
    """Configuration for CHoCH detection."""
    
    # Minimum ATR multiple for break confirmation
    min_break_atr: float = 0.3
    
    # Number of bars needed to confirm break (close-based)
    confirmation_bars: int = 2
    
    # Use close price for confirmation (vs. wick)
    use_close_confirmation: bool = True
    
    # Minimum volume spike for confirmation (multiplier of average)
    volume_spike_threshold: float = 1.3
    
    # Lookback period for average volume
    volume_lookback: int = 20
    
    # Whether to require volume confirmation
    require_volume_confirmation: bool = False


class CHoCHDetector:
    """
    Detects Change of Character (CHoCH) events.
    
    CHoCH occurs when:
    - In an uptrend: Price breaks below the most recent Higher Low (HL)
    - In a downtrend: Price breaks above the most recent Lower High (LH)
    
    This is the first indication that the trend may be reversing,
    and forms the "Left Shoulder" of a potential QML pattern.
    
    Detection criteria:
    1. Established trend (HH/HL for up, LH/LL for down)
    2. Break of key structural level (HL for up, LH for down)
    3. Break exceeds minimum ATR threshold
    4. Confirmation via subsequent price action
    5. Optional volume spike confirmation
    """
    
    def __init__(self, config: Optional[CHoCHConfig] = None):
        """
        Initialize CHoCH detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or CHoCHConfig()
        # Override from settings
        self.config.min_break_atr = settings.detection.choch_break_atr_multiplier
        self.config.confirmation_bars = settings.detection.choch_confirmation_bars
    
    def detect(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structures: List[MarketStructure],
        trend_state: TrendState,
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ) -> List[CHoCHEvent]:
        """
        Detect CHoCH events in the price data.
        
        Args:
            df: OHLCV DataFrame
            swing_points: Detected swing points
            structures: Market structures
            trend_state: Current trend state
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Returns:
            List of CHoCHEvent objects
        """
        if len(df) < 20:
            logger.warning("Insufficient data for CHoCH detection")
            return []
        
        choch_events = []
        
        # Calculate ATR
        atr = calculate_atr(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            settings.detection.atr_period
        )
        
        # Calculate average volume
        volume = df["volume"].values
        avg_volume = pd.Series(volume).rolling(self.config.volume_lookback).mean().values
        
        # Get time and price arrays
        time = df["time"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        # Detect based on trend
        if trend_state.trend == TrendType.UPTREND:
            # Look for bearish CHoCH (break below HL)
            events = self._detect_bearish_choch(
                time, high, low, close, volume, atr, avg_volume,
                trend_state, symbol, timeframe
            )
            choch_events.extend(events)
        
        elif trend_state.trend == TrendType.DOWNTREND:
            # Look for bullish CHoCH (break above LH)
            events = self._detect_bullish_choch(
                time, high, low, close, volume, atr, avg_volume,
                trend_state, symbol, timeframe
            )
            choch_events.extend(events)
        
        else:
            # CONSOLIDATION: Look for both bullish and bearish CHoCH
            # In consolidation, we can have local trends within the range
            # Look for breaks of recent swing points
            
            # Check if we have local uptrend indicators (recent HH/HL)
            if trend_state.last_hl and trend_state.last_hh:
                events = self._detect_bearish_choch(
                    time, high, low, close, volume, atr, avg_volume,
                    trend_state, symbol, timeframe
                )
                choch_events.extend(events)
            
            # Check if we have local downtrend indicators (recent LH/LL)
            if trend_state.last_lh and trend_state.last_ll:
                events = self._detect_bullish_choch(
                    time, high, low, close, volume, atr, avg_volume,
                    trend_state, symbol, timeframe
                )
                choch_events.extend(events)
        
        logger.debug(f"Detected {len(choch_events)} CHoCH events for {symbol} {timeframe}")
        
        return choch_events
    
    def _detect_bearish_choch(
        self,
        time: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        atr: np.ndarray,
        avg_volume: np.ndarray,
        trend_state: TrendState,
        symbol: str,
        timeframe: str
    ) -> List[CHoCHEvent]:
        """
        Detect bearish CHoCH (break below HL in uptrend).
        
        Returns list of CHoCH events.
        """
        events = []
        
        if not trend_state.last_hl:
            return events
        
        hl_level = trend_state.last_hl.price
        hl_time = trend_state.last_hl.time
        
        n = len(close)
        
        for i in range(1, n):
            # Skip bars before the HL was formed
            if pd.Timestamp(time[i]) <= hl_time:
                continue
            
            # Skip if ATR not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            min_break = atr[i] * self.config.min_break_atr
            
            # Check for break below HL
            if self.config.use_close_confirmation:
                break_price = close[i]
            else:
                break_price = low[i]
            
            if break_price >= hl_level - min_break:
                continue
            
            # Calculate break strength
            break_distance = hl_level - break_price
            break_strength = break_distance / atr[i]
            
            # Confirmation check
            confirmed = True
            if self.config.confirmation_bars > 0 and i + self.config.confirmation_bars < n:
                confirm_closes = close[i + 1:i + 1 + self.config.confirmation_bars]
                confirmed = np.all(confirm_closes < hl_level)
            
            # Volume confirmation
            volume_confirmation = True
            if self.config.require_volume_confirmation and not np.isnan(avg_volume[i]):
                volume_confirmation = volume[i] > avg_volume[i] * self.config.volume_spike_threshold
            
            if not volume_confirmation and self.config.require_volume_confirmation:
                continue
            
            # Create CHoCH event
            event = CHoCHEvent(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=timeframe,
                choch_type=PatternType.BEARISH,
                break_level=hl_level,
                break_strength=break_strength,
                volume_confirmation=volume_confirmation,
                confirmed=confirmed,
                bar_index=i
            )
            
            events.append(event)
            
            # Only detect first valid CHoCH after HL
            break
        
        return events
    
    def _detect_bullish_choch(
        self,
        time: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        atr: np.ndarray,
        avg_volume: np.ndarray,
        trend_state: TrendState,
        symbol: str,
        timeframe: str
    ) -> List[CHoCHEvent]:
        """
        Detect bullish CHoCH (break above LH in downtrend).
        
        Returns list of CHoCH events.
        """
        events = []
        
        if not trend_state.last_lh:
            return events
        
        lh_level = trend_state.last_lh.price
        lh_time = trend_state.last_lh.time
        
        n = len(close)
        
        for i in range(1, n):
            # Skip bars before the LH was formed
            if pd.Timestamp(time[i]) <= lh_time:
                continue
            
            # Skip if ATR not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            min_break = atr[i] * self.config.min_break_atr
            
            # Check for break above LH
            if self.config.use_close_confirmation:
                break_price = close[i]
            else:
                break_price = high[i]
            
            if break_price <= lh_level + min_break:
                continue
            
            # Calculate break strength
            break_distance = break_price - lh_level
            break_strength = break_distance / atr[i]
            
            # Confirmation check
            confirmed = True
            if self.config.confirmation_bars > 0 and i + self.config.confirmation_bars < n:
                confirm_closes = close[i + 1:i + 1 + self.config.confirmation_bars]
                confirmed = np.all(confirm_closes > lh_level)
            
            # Volume confirmation
            volume_confirmation = True
            if self.config.require_volume_confirmation and not np.isnan(avg_volume[i]):
                volume_confirmation = volume[i] > avg_volume[i] * self.config.volume_spike_threshold
            
            if not volume_confirmation and self.config.require_volume_confirmation:
                continue
            
            # Create CHoCH event
            event = CHoCHEvent(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=timeframe,
                choch_type=PatternType.BULLISH,
                break_level=lh_level,
                break_strength=break_strength,
                volume_confirmation=volume_confirmation,
                confirmed=confirmed,
                bar_index=i
            )
            
            events.append(event)
            
            # Only detect first valid CHoCH after LH
            break
        
        return events
    
    def get_choch_level(
        self,
        choch_event: CHoCHEvent,
        df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Get the CHoCH level and potential entry zone.
        
        Args:
            choch_event: CHoCH event
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (choch_level, potential_entry_zone)
        """
        choch_level = choch_event.break_level
        
        # Entry zone is typically at or near the CHoCH level on retest
        if choch_event.choch_type == PatternType.BULLISH:
            # For bullish, entry zone is below the CHoCH level
            entry_zone = choch_level * 0.995  # 0.5% below
        else:
            # For bearish, entry zone is above the CHoCH level
            entry_zone = choch_level * 1.005  # 0.5% above
        
        return choch_level, entry_zone


def detect_choch(
    df: pd.DataFrame,
    swing_points: List[SwingPoint],
    structures: List[MarketStructure],
    trend_state: TrendState,
    symbol: str,
    timeframe: str,
    config: Optional[CHoCHConfig] = None
) -> List[CHoCHEvent]:
    """
    Convenience function to detect CHoCH events.
    
    Args:
        df: OHLCV DataFrame
        swing_points: Detected swing points
        structures: Market structures
        trend_state: Current trend state
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        config: Optional configuration
        
    Returns:
        List of CHoCH events
    """
    detector = CHoCHDetector(config=config)
    return detector.detect(df, swing_points, structures, trend_state, symbol, timeframe)

