"""
Break of Structure (BoS) Detection for QML Trading System
==========================================================
Detects Break of Structure events that confirm trend continuation
following a CHoCH event.

BoS is the second critical component of a QML pattern, confirming
that the new trend direction is established.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.models import (
    BoSEvent,
    CHoCHEvent,
    PatternType,
    SwingPoint,
    SwingType,
)
from src.utils.indicators import calculate_atr


@dataclass
class BoSConfig:
    """Configuration for BoS detection."""
    
    # Minimum ATR multiple for break confirmation
    min_break_atr: float = 0.5
    
    # Volume spike threshold (multiplier of average)
    volume_spike_threshold: float = 1.5
    
    # Lookback period for average volume
    volume_lookback: int = 20
    
    # Maximum bars to look for BoS after CHoCH
    max_lookforward_bars: int = 50
    
    # Whether to require volume spike
    require_volume_spike: bool = True


class BoSDetector:
    """
    Detects Break of Structure (BoS) events.
    
    BoS occurs when:
    - After bullish CHoCH: Price makes a Higher High (new high above CHoCH level)
    - After bearish CHoCH: Price makes a Lower Low (new low below CHoCH level)
    
    This confirms the reversal and establishes the new trend direction.
    The area between CHoCH and BoS typically contains the "Head" of the
    QML pattern (the deepest point of the counter-move).
    
    Detection criteria:
    1. Valid CHoCH event has occurred
    2. Price breaks beyond the CHoCH level in the direction of reversal
    3. Break exceeds minimum ATR threshold
    4. Volume spike confirms institutional participation
    """
    
    def __init__(self, config: Optional[BoSConfig] = None):
        """
        Initialize BoS detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or BoSConfig()
        # Override from settings
        self.config.min_break_atr = settings.detection.bos_break_atr_multiplier
        self.config.volume_spike_threshold = settings.detection.bos_volume_spike_threshold
    
    def detect(
        self,
        df: pd.DataFrame,
        choch_events: List[CHoCHEvent],
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ) -> List[BoSEvent]:
        """
        Detect BoS events following CHoCH.
        
        Args:
            df: OHLCV DataFrame
            choch_events: List of CHoCH events
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Returns:
            List of BoSEvent objects
        """
        if not choch_events:
            return []
        
        if len(df) < 20:
            logger.warning("Insufficient data for BoS detection")
            return []
        
        bos_events = []
        
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
        
        # Process each CHoCH event
        for choch in choch_events:
            if choch.choch_type == PatternType.BULLISH:
                bos = self._detect_bullish_bos(
                    choch, time, high, low, close, volume, atr, avg_volume,
                    symbol, timeframe
                )
            else:
                bos = self._detect_bearish_bos(
                    choch, time, high, low, close, volume, atr, avg_volume,
                    symbol, timeframe
                )
            
            if bos:
                bos_events.append(bos)
        
        logger.debug(f"Detected {len(bos_events)} BoS events for {symbol} {timeframe}")
        
        return bos_events
    
    def _detect_bullish_bos(
        self,
        choch: CHoCHEvent,
        time: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        atr: np.ndarray,
        avg_volume: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> Optional[BoSEvent]:
        """
        Detect bullish BoS (new high above CHoCH level after bullish CHoCH).
        
        For a bullish BoS, we need:
        1. Price to break above the CHoCH level (which was the LH)
        2. Ideally with a volume spike
        
        The "Head" of the pattern is the lowest point between CHoCH and BoS.
        """
        n = len(close)
        choch_idx = choch.bar_index or 0
        choch_level = choch.break_level
        
        # Search window after CHoCH
        start_idx = choch_idx + 1
        end_idx = min(choch_idx + self.config.max_lookforward_bars, n)
        
        best_bos: Optional[BoSEvent] = None
        
        for i in range(start_idx, end_idx):
            # Skip if ATR not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            min_break = atr[i] * self.config.min_break_atr
            
            # Check for break above CHoCH level
            if high[i] <= choch_level + min_break:
                continue
            
            # Calculate break strength
            break_level = high[i]
            
            # Check volume spike
            volume_spike = False
            if not np.isnan(avg_volume[i]) and avg_volume[i] > 0:
                volume_spike = volume[i] > avg_volume[i] * self.config.volume_spike_threshold
            
            # If we require volume and don't have it, continue searching
            if self.config.require_volume_spike and not volume_spike:
                continue
            
            # Create BoS event
            bos = BoSEvent(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=timeframe,
                bos_type=PatternType.BULLISH,
                break_level=break_level,
                volume_spike=volume_spike,
                choch_event=choch,
                bar_index=i
            )
            
            # Take the first valid BoS
            best_bos = bos
            break
        
        return best_bos
    
    def _detect_bearish_bos(
        self,
        choch: CHoCHEvent,
        time: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        atr: np.ndarray,
        avg_volume: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> Optional[BoSEvent]:
        """
        Detect bearish BoS (new low below CHoCH level after bearish CHoCH).
        
        For a bearish BoS, we need:
        1. Price to break below the CHoCH level (which was the HL)
        2. Ideally with a volume spike
        
        The "Head" of the pattern is the highest point between CHoCH and BoS.
        """
        n = len(close)
        choch_idx = choch.bar_index or 0
        choch_level = choch.break_level
        
        # Search window after CHoCH
        start_idx = choch_idx + 1
        end_idx = min(choch_idx + self.config.max_lookforward_bars, n)
        
        best_bos: Optional[BoSEvent] = None
        
        for i in range(start_idx, end_idx):
            # Skip if ATR not available
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            
            min_break = atr[i] * self.config.min_break_atr
            
            # Check for break below CHoCH level
            if low[i] >= choch_level - min_break:
                continue
            
            # Calculate break level
            break_level = low[i]
            
            # Check volume spike
            volume_spike = False
            if not np.isnan(avg_volume[i]) and avg_volume[i] > 0:
                volume_spike = volume[i] > avg_volume[i] * self.config.volume_spike_threshold
            
            # If we require volume and don't have it, continue searching
            if self.config.require_volume_spike and not volume_spike:
                continue
            
            # Create BoS event
            bos = BoSEvent(
                time=pd.Timestamp(time[i]),
                symbol=symbol,
                timeframe=timeframe,
                bos_type=PatternType.BEARISH,
                break_level=break_level,
                volume_spike=volume_spike,
                choch_event=choch,
                bar_index=i
            )
            
            # Take the first valid BoS
            best_bos = bos
            break
        
        return best_bos
    
    def find_head_point(
        self,
        df: pd.DataFrame,
        choch: CHoCHEvent,
        bos: BoSEvent
    ) -> Tuple[Optional[float], Optional[datetime], Optional[int]]:
        """
        Find the head point (extreme that triggered the CHoCH).
        
        In QML pattern:
        - For bearish: head is the highest high BEFORE the CHoCH break
        - For bullish: head is the lowest low BEFORE the CHoCH break
        
        Args:
            df: OHLCV DataFrame
            choch: CHoCH event
            bos: BoS event
            
        Returns:
            Tuple of (head_price, head_time, head_index) or (None, None, None)
        """
        choch_idx = choch.bar_index or 0
        
        time = df["time"].values
        high = df["high"].values
        low = df["low"].values
        
        # Look backwards from CHoCH to find the head
        # Head is the extreme point that formed before CHoCH
        lookback = min(50, choch_idx)  # Look back up to 50 bars
        
        if lookback < 3:
            return None, None, None
        
        region_start = max(0, choch_idx - lookback)
        region_end = choch_idx
        
        if bos.bos_type == PatternType.BULLISH:
            # For bullish reversal (from downtrend), head is the lowest low
            head_idx = region_start + np.argmin(low[region_start:region_end])
            head_price = float(low[head_idx])
        else:
            # For bearish reversal (from uptrend), head is the highest high
            head_idx = region_start + np.argmax(high[region_start:region_end])
            head_price = float(high[head_idx])
        
        head_time = pd.Timestamp(time[head_idx])
        
        return head_price, head_time, head_idx


def detect_bos(
    df: pd.DataFrame,
    choch_events: List[CHoCHEvent],
    symbol: str,
    timeframe: str,
    config: Optional[BoSConfig] = None
) -> List[BoSEvent]:
    """
    Convenience function to detect BoS events.
    
    Args:
        df: OHLCV DataFrame
        choch_events: List of CHoCH events
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        config: Optional configuration
        
    Returns:
        List of BoS events
    """
    detector = BoSDetector(config=config)
    return detector.detect(df, choch_events, symbol, timeframe)

