"""
Triple Barrier Labeling for QML Trading System
===============================================
Creates target labels for ML training using the triple-barrier method.

The triple barrier method labels outcomes based on which barrier
price touches first:
1. Upper barrier (Take Profit) -> Win
2. Lower barrier (Stop Loss) -> Loss
3. Vertical barrier (Time limit) -> Timeout
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.models import PatternType, QMLPattern, TradeOutcome


class Label(Enum):
    """Trade outcome labels."""
    WIN = 1
    LOSS = 0
    TIMEOUT = -1


@dataclass
class LabelConfig:
    """Configuration for triple barrier labeling."""
    
    # Risk-reward ratios for barriers
    take_profit_rr: float = 3.0  # 3:1 R:R
    stop_loss_multiplier: float = 1.0  # 1x risk
    
    # Time barrier (maximum holding period in bars)
    max_holding_bars: int = 50
    
    # Alternative: time barrier as percentage of pattern duration
    time_barrier_pattern_multiple: float = 3.0
    
    # Minimum number of bars before timeout
    min_bars_before_timeout: int = 10
    
    # Use pattern-derived levels vs. fixed multipliers
    use_pattern_levels: bool = True


@dataclass
class LabelResult:
    """Result of labeling a pattern."""
    
    pattern_id: int
    label: Label
    outcome: TradeOutcome
    
    entry_price: float
    exit_price: float
    return_pct: float
    
    bars_to_outcome: int
    hit_tp: bool
    hit_sl: bool
    hit_time_barrier: bool


class TripleBarrierLabeler:
    """
    Labels patterns using the triple-barrier method.
    
    For each pattern, determines the outcome by simulating
    a trade and checking which barrier is hit first:
    
    1. Upper barrier (TP): Price reaches take profit level
    2. Lower barrier (SL): Price reaches stop loss level  
    3. Vertical barrier: Maximum holding time exceeded
    
    This provides clean, unambiguous labels for ML training.
    """
    
    def __init__(self, config: Optional[LabelConfig] = None):
        """
        Initialize labeler.
        
        Args:
            config: Labeling configuration
        """
        self.config = config or LabelConfig()
    
    def label_pattern(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame
    ) -> Optional[LabelResult]:
        """
        Label a single pattern using triple barrier method.
        
        Args:
            pattern: QML pattern to label
            df: OHLCV DataFrame with forward price data
            
        Returns:
            LabelResult or None if unable to label
        """
        # Get entry bar index
        entry_idx = self._find_bar_index(df, pattern.detection_time)
        
        if entry_idx is None:
            logger.warning(f"Could not find entry bar for pattern")
            return None
        
        # Check we have enough forward data
        if entry_idx + self.config.min_bars_before_timeout >= len(df):
            logger.debug("Insufficient forward data for labeling")
            return None
        
        # Get trading levels
        if self.config.use_pattern_levels and pattern.trading_levels:
            entry_price = pattern.trading_levels.entry
            stop_loss = pattern.trading_levels.stop_loss
            take_profit = pattern.trading_levels.take_profit_3  # Use 3:1 target
        else:
            # Calculate from pattern
            entry_price, stop_loss, take_profit = self._calculate_barriers(pattern, df, entry_idx)
        
        if entry_price is None or stop_loss is None or take_profit is None:
            return None
        
        # Calculate time barrier
        if pattern.trading_levels:
            # Use pattern duration as reference
            duration = self._get_pattern_duration(pattern, df)
            time_barrier = min(
                self.config.max_holding_bars,
                int(duration * self.config.time_barrier_pattern_multiple)
            )
        else:
            time_barrier = self.config.max_holding_bars
        
        # Ensure minimum time barrier
        time_barrier = max(time_barrier, self.config.min_bars_before_timeout)
        
        # Run barrier check
        result = self._check_barriers(
            pattern=pattern,
            df=df,
            entry_idx=entry_idx,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_barrier=time_barrier
        )
        
        return result
    
    def label_patterns(
        self,
        patterns: List[QMLPattern],
        price_data: pd.DataFrame
    ) -> Tuple[List[LabelResult], pd.DataFrame]:
        """
        Label multiple patterns and return results with feature-ready DataFrame.
        
        Args:
            patterns: List of patterns to label
            price_data: OHLCV DataFrame
            
        Returns:
            Tuple of (list of LabelResults, DataFrame with labels)
        """
        results = []
        
        for pattern in patterns:
            result = self.label_pattern(pattern, price_data)
            if result:
                results.append(result)
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame([
                {
                    "pattern_id": r.pattern_id,
                    "label": r.label.value,
                    "outcome": r.outcome.value,
                    "return_pct": r.return_pct,
                    "bars_to_outcome": r.bars_to_outcome,
                    "hit_tp": r.hit_tp,
                    "hit_sl": r.hit_sl,
                }
                for r in results
            ])
        else:
            df = pd.DataFrame()
        
        logger.info(f"Labeled {len(results)} patterns: "
                   f"{sum(1 for r in results if r.label == Label.WIN)} wins, "
                   f"{sum(1 for r in results if r.label == Label.LOSS)} losses, "
                   f"{sum(1 for r in results if r.label == Label.TIMEOUT)} timeouts")
        
        return results, df
    
    def _find_bar_index(
        self,
        df: pd.DataFrame,
        target_time: datetime
    ) -> Optional[int]:
        """Find bar index for a timestamp."""
        times = pd.to_datetime(df["time"])
        target = pd.Timestamp(target_time)
        
        # Find exact or next bar after target time
        for i, t in enumerate(times):
            if t >= target:
                return i
        
        return None
    
    def _get_pattern_duration(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame
    ) -> int:
        """Get pattern duration in bars."""
        start_idx = self._find_bar_index(df, pattern.left_shoulder_time)
        end_idx = self._find_bar_index(df, pattern.detection_time)
        
        if start_idx is not None and end_idx is not None:
            return end_idx - start_idx
        
        return 10  # Default
    
    def _calculate_barriers(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame,
        entry_idx: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate barrier levels from pattern geometry."""
        
        # Entry is at detection point
        entry_price = df["close"].iloc[entry_idx]
        
        # Risk based on head distance
        if pattern.pattern_type == PatternType.BULLISH:
            risk = entry_price - pattern.head_price
            stop_loss = entry_price - (risk * self.config.stop_loss_multiplier)
            take_profit = entry_price + (risk * self.config.take_profit_rr)
        else:
            risk = pattern.head_price - entry_price
            stop_loss = entry_price + (risk * self.config.stop_loss_multiplier)
            take_profit = entry_price - (risk * self.config.take_profit_rr)
        
        return entry_price, stop_loss, take_profit
    
    def _check_barriers(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        time_barrier: int
    ) -> LabelResult:
        """
        Check which barrier is hit first.
        
        Simulates bar-by-bar price action to determine outcome.
        """
        is_long = pattern.pattern_type == PatternType.BULLISH
        
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        end_idx = min(entry_idx + time_barrier, len(df) - 1)
        
        hit_tp = False
        hit_sl = False
        exit_idx = entry_idx
        exit_price = entry_price
        
        for i in range(entry_idx + 1, end_idx + 1):
            bar_high = high[i]
            bar_low = low[i]
            
            if is_long:
                # Long trade: check for TP hit first (intrabar), then SL
                if bar_high >= take_profit:
                    hit_tp = True
                    exit_idx = i
                    exit_price = take_profit
                    break
                if bar_low <= stop_loss:
                    hit_sl = True
                    exit_idx = i
                    exit_price = stop_loss
                    break
            else:
                # Short trade: check for TP hit first, then SL
                if bar_low <= take_profit:
                    hit_tp = True
                    exit_idx = i
                    exit_price = take_profit
                    break
                if bar_high >= stop_loss:
                    hit_sl = True
                    exit_idx = i
                    exit_price = stop_loss
                    break
        
        # Determine label
        if hit_tp:
            label = Label.WIN
            outcome = TradeOutcome.WIN
        elif hit_sl:
            label = Label.LOSS
            outcome = TradeOutcome.LOSS
        else:
            # Time barrier hit
            label = Label.TIMEOUT
            exit_price = close[end_idx]
            exit_idx = end_idx
            
            # Determine outcome based on final P&L
            if is_long:
                outcome = TradeOutcome.WIN if exit_price > entry_price else TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.WIN if exit_price < entry_price else TradeOutcome.LOSS
        
        # Calculate return
        if is_long:
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100
        
        return LabelResult(
            pattern_id=pattern.id or 0,
            label=label,
            outcome=outcome,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=return_pct,
            bars_to_outcome=exit_idx - entry_idx,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            hit_time_barrier=not hit_tp and not hit_sl
        )


def create_labeler(config: Optional[LabelConfig] = None) -> TripleBarrierLabeler:
    """Factory function for labeler."""
    return TripleBarrierLabeler(config=config)

