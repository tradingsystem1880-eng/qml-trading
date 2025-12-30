"""
Market Structure Analyzer for QML Trading System
=================================================
Analyzes market structure by classifying swing points into
Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL).

This module provides the foundation for trend identification and
Change of Character (CHoCH) detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.models import (
    MarketStructure,
    StructureType,
    SwingPoint,
    SwingType,
    TrendType,
)


@dataclass
class StructureConfig:
    """Configuration for structure analysis."""
    
    # Minimum swings needed for structure analysis
    min_swings: int = 4
    
    # ATR threshold for structure classification
    # Move must exceed this * ATR to be considered a new structure
    structure_atr_threshold: float = 0.15
    
    # Trend strength threshold (0-1)
    trend_strength_threshold: float = 0.6
    
    # Number of recent structures to consider for trend
    trend_lookback: int = 6


@dataclass
class TrendState:
    """Current market trend state."""
    
    trend: TrendType = TrendType.CONSOLIDATION
    strength: float = 0.0
    last_hh: Optional[SwingPoint] = None
    last_hl: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None
    last_ll: Optional[SwingPoint] = None
    structure_sequence: List[StructureType] = field(default_factory=list)


class StructureAnalyzer:
    """
    Analyzes market structure from swing points.
    
    Market Structure Classification:
    - HH (Higher High): Swing high exceeds previous swing high
    - HL (Higher Low): Swing low exceeds previous swing low
    - LH (Lower High): Swing high fails to exceed previous swing high
    - LL (Lower Low): Swing low breaks below previous swing low
    
    Trend Classification:
    - Uptrend: Sequence of HH and HL
    - Downtrend: Sequence of LH and LL
    - Consolidation: Mixed structure without clear direction
    """
    
    def __init__(self, config: Optional[StructureConfig] = None):
        """
        Initialize structure analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or StructureConfig()
    
    def analyze(
        self,
        swing_points: List[SwingPoint],
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ) -> Tuple[List[MarketStructure], TrendState]:
        """
        Analyze market structure from swing points.
        
        Args:
            swing_points: List of swing points (must be sorted by time)
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Returns:
            Tuple of (list of MarketStructure, current TrendState)
        """
        if len(swing_points) < self.config.min_swings:
            logger.warning(f"Insufficient swings for structure analysis: {len(swing_points)}")
            return [], TrendState()
        
        # Separate swing highs and lows
        highs = [s for s in swing_points if s.swing_type == SwingType.HIGH]
        lows = [s for s in swing_points if s.swing_type == SwingType.LOW]
        
        # Sort by time
        highs.sort(key=lambda x: x.time)
        lows.sort(key=lambda x: x.time)
        
        structures: List[MarketStructure] = []
        trend_state = TrendState()
        
        # Classify swing highs
        for i in range(1, len(highs)):
            current = highs[i]
            previous = highs[i - 1]
            
            # Determine structure type
            price_diff = current.price - previous.price
            atr_diff = price_diff / current.atr_at_point
            
            if atr_diff > self.config.structure_atr_threshold:
                structure_type = StructureType.HH
                trend_state.last_hh = current
            elif atr_diff < -self.config.structure_atr_threshold:
                structure_type = StructureType.LH
                trend_state.last_lh = current
            else:
                # Equal high (within threshold)
                structure_type = StructureType.LH  # Treat as LH (failure to make new high)
                trend_state.last_lh = current
            
            structure = MarketStructure(
                time=current.time,
                symbol=symbol,
                timeframe=timeframe,
                structure_type=structure_type,
                price=current.price,
                previous_price=previous.price,
                swing_point=current
            )
            structures.append(structure)
            trend_state.structure_sequence.append(structure_type)
        
        # Classify swing lows
        for i in range(1, len(lows)):
            current = lows[i]
            previous = lows[i - 1]
            
            # Determine structure type
            price_diff = current.price - previous.price
            atr_diff = price_diff / current.atr_at_point
            
            if atr_diff > self.config.structure_atr_threshold:
                structure_type = StructureType.HL
                trend_state.last_hl = current
            elif atr_diff < -self.config.structure_atr_threshold:
                structure_type = StructureType.LL
                trend_state.last_ll = current
            else:
                # Equal low (within threshold)
                structure_type = StructureType.LL  # Treat as LL (failure to hold)
                trend_state.last_ll = current
            
            structure = MarketStructure(
                time=current.time,
                symbol=symbol,
                timeframe=timeframe,
                structure_type=structure_type,
                price=current.price,
                previous_price=previous.price,
                swing_point=current
            )
            structures.append(structure)
            trend_state.structure_sequence.append(structure_type)
        
        # Sort all structures by time
        structures.sort(key=lambda x: x.time)
        
        # Determine current trend
        trend_state.trend, trend_state.strength = self._calculate_trend(
            trend_state.structure_sequence[-self.config.trend_lookback:]
        )
        
        # Update trend in structures
        for s in structures:
            s.trend = trend_state.trend
            s.trend_strength = trend_state.strength
        
        logger.debug(
            f"Structure analysis for {symbol} {timeframe}: "
            f"{len(structures)} structures, trend={trend_state.trend.value}, "
            f"strength={trend_state.strength:.2f}"
        )
        
        return structures, trend_state
    
    def _calculate_trend(
        self,
        structure_sequence: List[StructureType]
    ) -> Tuple[TrendType, float]:
        """
        Calculate trend type and strength from structure sequence.
        
        Args:
            structure_sequence: Recent structure types
            
        Returns:
            Tuple of (TrendType, strength 0-1)
        """
        if not structure_sequence:
            return TrendType.CONSOLIDATION, 0.0
        
        # Count bullish and bearish structures
        bullish_count = sum(
            1 for s in structure_sequence
            if s in [StructureType.HH, StructureType.HL]
        )
        bearish_count = sum(
            1 for s in structure_sequence
            if s in [StructureType.LH, StructureType.LL]
        )
        
        total = len(structure_sequence)
        
        if total == 0:
            return TrendType.CONSOLIDATION, 0.0
        
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        
        # Calculate strength as dominance of one direction
        strength = abs(bullish_ratio - bearish_ratio)
        
        # Determine trend
        if bullish_ratio > bearish_ratio and strength >= self.config.trend_strength_threshold:
            return TrendType.UPTREND, strength
        elif bearish_ratio > bullish_ratio and strength >= self.config.trend_strength_threshold:
            return TrendType.DOWNTREND, strength
        else:
            return TrendType.CONSOLIDATION, strength
    
    def get_key_levels(
        self,
        structures: List[MarketStructure],
        trend_state: TrendState
    ) -> Dict[str, Optional[float]]:
        """
        Get key structural levels for trading.
        
        Args:
            structures: List of market structures
            trend_state: Current trend state
            
        Returns:
            Dictionary with key levels
        """
        levels = {
            "last_hh": trend_state.last_hh.price if trend_state.last_hh else None,
            "last_hl": trend_state.last_hl.price if trend_state.last_hl else None,
            "last_lh": trend_state.last_lh.price if trend_state.last_lh else None,
            "last_ll": trend_state.last_ll.price if trend_state.last_ll else None,
        }
        
        # Get protected levels (key invalidation points)
        if trend_state.trend == TrendType.UPTREND and trend_state.last_hl:
            levels["protected_low"] = trend_state.last_hl.price
        elif trend_state.trend == TrendType.DOWNTREND and trend_state.last_lh:
            levels["protected_high"] = trend_state.last_lh.price
        
        return levels
    
    def is_structure_break(
        self,
        current_price: float,
        structures: List[MarketStructure],
        trend_state: TrendState,
        atr: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current price breaks a key structural level.
        
        Args:
            current_price: Current market price
            structures: Market structures
            trend_state: Current trend state
            atr: Current ATR value
            
        Returns:
            Tuple of (is_break, break_type or None)
        """
        min_break_distance = atr * settings.detection.choch_break_atr_multiplier
        
        # In uptrend, check for break below last HL
        if trend_state.trend == TrendType.UPTREND and trend_state.last_hl:
            if current_price < trend_state.last_hl.price - min_break_distance:
                return True, "bearish_choch"
        
        # In downtrend, check for break above last LH
        if trend_state.trend == TrendType.DOWNTREND and trend_state.last_lh:
            if current_price > trend_state.last_lh.price + min_break_distance:
                return True, "bullish_choch"
        
        return False, None


def analyze_structure(
    swing_points: List[SwingPoint],
    symbol: str,
    timeframe: str,
    config: Optional[StructureConfig] = None
) -> Tuple[List[MarketStructure], TrendState]:
    """
    Convenience function to analyze market structure.
    
    Args:
        swing_points: List of swing points
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        config: Optional configuration
        
    Returns:
        Tuple of (structures, trend_state)
    """
    analyzer = StructureAnalyzer(config=config)
    return analyzer.analyze(swing_points, symbol, timeframe)

