#!/usr/bin/env python3
"""
Phase 9.6: Smart Money Concepts (SMC) Research Module
=====================================================
Research-only implementation of Order Blocks and Fair Value Gaps.

These are complementary concepts to QML patterns, NOT replacements.
This module is for RESEARCH purposes to understand if SMC concepts
can enhance QML pattern quality scoring.

References:
- Inner Circle Trader (ICT) methodology
- SMC Wikipedia/educational materials

Status: RESEARCH SKELETON - Not for production use.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class OrderBlockType(Enum):
    """Type of order block."""
    BULLISH = "bullish"  # Last bearish candle before rally
    BEARISH = "bearish"  # Last bullish candle before drop


class FVGType(Enum):
    """Type of Fair Value Gap."""
    BULLISH = "bullish"  # Gap up (support)
    BEARISH = "bearish"  # Gap down (resistance)


@dataclass
class OrderBlock:
    """
    Order Block detection result.

    An Order Block is the last opposing candle before a significant move.
    - Bullish OB: Last red candle before a strong bullish move
    - Bearish OB: Last green candle before a strong bearish move

    Theory: These represent institutional accumulation/distribution zones
    where price may return for re-accumulation.
    """
    block_type: OrderBlockType
    start_idx: int
    end_idx: int
    high: float
    low: float
    open_price: float
    close_price: float
    timestamp: datetime
    strength: float  # 0-1 based on move after block
    mitigated: bool = False  # True if price returned to zone
    mitigation_idx: Optional[int] = None


@dataclass
class FairValueGap:
    """
    Fair Value Gap (FVG) detection result.

    An FVG is an imbalance between 3 consecutive candles where
    the wicks don't overlap:
    - Bullish FVG: Candle 1 high < Candle 3 low (gap up)
    - Bearish FVG: Candle 1 low > Candle 3 high (gap down)

    Theory: Price tends to return to fill these gaps, providing
    support/resistance and potential entry points.
    """
    fvg_type: FVGType
    candle1_idx: int
    candle2_idx: int
    candle3_idx: int
    gap_high: float
    gap_low: float
    midpoint: float
    timestamp: datetime
    gap_size_pct: float  # Gap size as % of price
    filled: bool = False  # True if price returned to fill
    fill_idx: Optional[int] = None
    fill_percentage: float = 0.0  # How much of gap was filled


class OrderBlockDetector:
    """
    Detects Order Blocks in price data.

    RESEARCH IMPLEMENTATION - Not production ready.

    Algorithm:
    1. Find significant moves (> threshold ATR)
    2. Look back for last opposing candle
    3. Mark as Order Block
    4. Track mitigation (price returning to zone)
    """

    def __init__(
        self,
        min_move_atr: float = 2.0,
        lookback: int = 10,
        atr_period: int = 14,
    ):
        """
        Initialize detector.

        Args:
            min_move_atr: Minimum move size in ATR for OB qualification
            lookback: How many candles back to look for opposing candle
            atr_period: ATR calculation period
        """
        self.min_move_atr = min_move_atr
        self.lookback = lookback
        self.atr_period = atr_period

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect Order Blocks in price data.

        Args:
            df: DataFrame with columns: open, high, low, close, time/timestamp

        Returns:
            List of OrderBlock objects
        """
        if len(df) < self.atr_period + self.lookback + 5:
            return []

        atr = self._calculate_atr(df)
        order_blocks = []

        # Ensure we have a time column
        time_col = 'time' if 'time' in df.columns else 'timestamp'

        for i in range(self.lookback + 5, len(df)):
            current_atr = atr.iloc[i]
            if pd.isna(current_atr) or current_atr == 0:
                continue

            # Check for significant bullish move
            move_up = df['close'].iloc[i] - df['low'].iloc[i - 3:i].min()
            if move_up > self.min_move_atr * current_atr:
                # Look for last bearish candle
                for j in range(i - 1, max(i - self.lookback, 0), -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]:  # Bearish candle
                        ob = OrderBlock(
                            block_type=OrderBlockType.BULLISH,
                            start_idx=j,
                            end_idx=j,
                            high=df['high'].iloc[j],
                            low=df['low'].iloc[j],
                            open_price=df['open'].iloc[j],
                            close_price=df['close'].iloc[j],
                            timestamp=df[time_col].iloc[j],
                            strength=min(1.0, move_up / (self.min_move_atr * current_atr * 2)),
                        )
                        order_blocks.append(ob)
                        break

            # Check for significant bearish move
            move_down = df['high'].iloc[i - 3:i].max() - df['close'].iloc[i]
            if move_down > self.min_move_atr * current_atr:
                # Look for last bullish candle
                for j in range(i - 1, max(i - self.lookback, 0), -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]:  # Bullish candle
                        ob = OrderBlock(
                            block_type=OrderBlockType.BEARISH,
                            start_idx=j,
                            end_idx=j,
                            high=df['high'].iloc[j],
                            low=df['low'].iloc[j],
                            open_price=df['open'].iloc[j],
                            close_price=df['close'].iloc[j],
                            timestamp=df[time_col].iloc[j],
                            strength=min(1.0, move_down / (self.min_move_atr * current_atr * 2)),
                        )
                        order_blocks.append(ob)
                        break

        return order_blocks

    def check_mitigation(
        self,
        ob: OrderBlock,
        df: pd.DataFrame,
        start_idx: int,
    ) -> OrderBlock:
        """
        Check if Order Block has been mitigated (price returned to zone).

        Args:
            ob: OrderBlock to check
            df: Price DataFrame
            start_idx: Index to start checking from

        Returns:
            Updated OrderBlock with mitigation status
        """
        for i in range(start_idx, len(df)):
            if ob.block_type == OrderBlockType.BULLISH:
                # Bullish OB mitigated when price touches the zone from above
                if df['low'].iloc[i] <= ob.high:
                    ob.mitigated = True
                    ob.mitigation_idx = i
                    break
            else:
                # Bearish OB mitigated when price touches the zone from below
                if df['high'].iloc[i] >= ob.low:
                    ob.mitigated = True
                    ob.mitigation_idx = i
                    break

        return ob


class FVGDetector:
    """
    Detects Fair Value Gaps in price data.

    RESEARCH IMPLEMENTATION - Not production ready.

    Algorithm:
    1. For each set of 3 consecutive candles
    2. Check if candle 1 wick doesn't overlap with candle 3 wick
    3. Mark the gap zone
    4. Track if gap gets filled
    """

    def __init__(
        self,
        min_gap_pct: float = 0.001,  # 0.1% minimum gap
    ):
        """
        Initialize detector.

        Args:
            min_gap_pct: Minimum gap size as percentage of price
        """
        self.min_gap_pct = min_gap_pct

    def detect(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price data.

        Args:
            df: DataFrame with columns: open, high, low, close, time/timestamp

        Returns:
            List of FairValueGap objects
        """
        if len(df) < 5:
            return []

        fvgs = []
        time_col = 'time' if 'time' in df.columns else 'timestamp'

        for i in range(2, len(df)):
            c1_high = df['high'].iloc[i - 2]
            c1_low = df['low'].iloc[i - 2]
            c3_high = df['high'].iloc[i]
            c3_low = df['low'].iloc[i]
            mid_price = df['close'].iloc[i - 1]

            # Bullish FVG: Candle 1 high < Candle 3 low
            if c1_high < c3_low:
                gap_size = c3_low - c1_high
                gap_pct = gap_size / mid_price

                if gap_pct >= self.min_gap_pct:
                    fvg = FairValueGap(
                        fvg_type=FVGType.BULLISH,
                        candle1_idx=i - 2,
                        candle2_idx=i - 1,
                        candle3_idx=i,
                        gap_high=c3_low,
                        gap_low=c1_high,
                        midpoint=(c3_low + c1_high) / 2,
                        timestamp=df[time_col].iloc[i - 1],
                        gap_size_pct=gap_pct,
                    )
                    fvgs.append(fvg)

            # Bearish FVG: Candle 1 low > Candle 3 high
            if c1_low > c3_high:
                gap_size = c1_low - c3_high
                gap_pct = gap_size / mid_price

                if gap_pct >= self.min_gap_pct:
                    fvg = FairValueGap(
                        fvg_type=FVGType.BEARISH,
                        candle1_idx=i - 2,
                        candle2_idx=i - 1,
                        candle3_idx=i,
                        gap_high=c1_low,
                        gap_low=c3_high,
                        midpoint=(c1_low + c3_high) / 2,
                        timestamp=df[time_col].iloc[i - 1],
                        gap_size_pct=gap_pct,
                    )
                    fvgs.append(fvg)

        return fvgs

    def check_fill(
        self,
        fvg: FairValueGap,
        df: pd.DataFrame,
        start_idx: int,
    ) -> FairValueGap:
        """
        Check if FVG has been filled.

        Args:
            fvg: FairValueGap to check
            df: Price DataFrame
            start_idx: Index to start checking from

        Returns:
            Updated FairValueGap with fill status
        """
        gap_size = fvg.gap_high - fvg.gap_low

        for i in range(start_idx, len(df)):
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG filled when price drops into gap
                if df['low'].iloc[i] <= fvg.gap_high:
                    fill_depth = fvg.gap_high - max(df['low'].iloc[i], fvg.gap_low)
                    fvg.fill_percentage = min(1.0, fill_depth / gap_size)
                    if fvg.fill_percentage >= 0.5:  # Consider >50% as filled
                        fvg.filled = True
                        fvg.fill_idx = i
                        break
            else:
                # Bearish FVG filled when price rises into gap
                if df['high'].iloc[i] >= fvg.gap_low:
                    fill_depth = min(df['high'].iloc[i], fvg.gap_high) - fvg.gap_low
                    fvg.fill_percentage = min(1.0, fill_depth / gap_size)
                    if fvg.fill_percentage >= 0.5:
                        fvg.filled = True
                        fvg.fill_idx = i
                        break

        return fvg


class SMCAnalyzer:
    """
    Combines Order Blocks and FVGs for confluence analysis.

    RESEARCH IMPLEMENTATION - Not production ready.

    Purpose: Analyze if SMC concepts can enhance QML pattern quality.

    Research questions:
    1. Do QML patterns forming near Order Blocks have higher win rates?
    2. Do FVGs provide better entry refinement for QML signals?
    3. Can SMC confluence improve the pattern scoring system?
    """

    def __init__(self):
        """Initialize analyzer."""
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()

    def analyze_confluence(
        self,
        df: pd.DataFrame,
        pattern_idx: int,
        pattern_direction: str,  # 'BULLISH' or 'BEARISH'
    ) -> dict:
        """
        Analyze SMC confluence for a QML pattern.

        Args:
            df: Price DataFrame
            pattern_idx: Index where pattern completes (P5)
            pattern_direction: Pattern direction

        Returns:
            Dict with confluence analysis
        """
        # Detect SMC structures before pattern
        lookback_start = max(0, pattern_idx - 100)
        df_window = df.iloc[lookback_start:pattern_idx + 1].copy()
        df_window = df_window.reset_index(drop=True)

        order_blocks = self.ob_detector.detect(df_window)
        fvgs = self.fvg_detector.detect(df_window)

        # Find relevant structures
        pattern_price = df['close'].iloc[pattern_idx]
        nearby_obs = []
        nearby_fvgs = []

        for ob in order_blocks:
            if ob.block_type.value == pattern_direction.lower():
                # Check if price is near OB zone
                distance_pct = abs(pattern_price - ob.high) / pattern_price
                if distance_pct < 0.02:  # Within 2%
                    nearby_obs.append({
                        'ob': ob,
                        'distance_pct': distance_pct,
                    })

        for fvg in fvgs:
            if fvg.fvg_type.value == pattern_direction.lower():
                # Check if pattern is near FVG
                distance_pct = abs(pattern_price - fvg.midpoint) / pattern_price
                if distance_pct < 0.02:
                    nearby_fvgs.append({
                        'fvg': fvg,
                        'distance_pct': distance_pct,
                    })

        return {
            'has_ob_confluence': len(nearby_obs) > 0,
            'has_fvg_confluence': len(nearby_fvgs) > 0,
            'ob_count': len(nearby_obs),
            'fvg_count': len(nearby_fvgs),
            'confluence_score': min(1.0, (len(nearby_obs) * 0.5 + len(nearby_fvgs) * 0.3)),
            'nearby_obs': nearby_obs,
            'nearby_fvgs': nearby_fvgs,
        }

    def generate_research_report(
        self,
        df: pd.DataFrame,
        trades: List[dict],
    ) -> dict:
        """
        Generate research report on SMC confluence with trade outcomes.

        Args:
            df: Price DataFrame
            trades: List of trade dicts with 'entry_idx', 'direction', 'pnl_r'

        Returns:
            Research report dict
        """
        results = {
            'total_trades': len(trades),
            'trades_with_ob': 0,
            'trades_with_fvg': 0,
            'trades_with_both': 0,
            'ob_win_rate': 0.0,
            'fvg_win_rate': 0.0,
            'both_win_rate': 0.0,
            'no_confluence_win_rate': 0.0,
        }

        ob_trades = []
        fvg_trades = []
        both_trades = []
        no_conf_trades = []

        for trade in trades:
            confluence = self.analyze_confluence(
                df,
                trade['entry_idx'],
                trade['direction'],
            )

            has_ob = confluence['has_ob_confluence']
            has_fvg = confluence['has_fvg_confluence']
            won = trade['pnl_r'] > 0

            if has_ob and has_fvg:
                results['trades_with_both'] += 1
                both_trades.append(won)
            elif has_ob:
                results['trades_with_ob'] += 1
                ob_trades.append(won)
            elif has_fvg:
                results['trades_with_fvg'] += 1
                fvg_trades.append(won)
            else:
                no_conf_trades.append(won)

        # Calculate win rates
        if ob_trades:
            results['ob_win_rate'] = sum(ob_trades) / len(ob_trades)
        if fvg_trades:
            results['fvg_win_rate'] = sum(fvg_trades) / len(fvg_trades)
        if both_trades:
            results['both_win_rate'] = sum(both_trades) / len(both_trades)
        if no_conf_trades:
            results['no_confluence_win_rate'] = sum(no_conf_trades) / len(no_conf_trades)

        return results


# Convenience functions for future integration
def get_smc_features(df: pd.DataFrame, bar_idx: int) -> dict:
    """
    Get SMC features for a specific bar.

    SKELETON - Returns basic structure until fully implemented.

    Args:
        df: Price DataFrame
        bar_idx: Bar index to analyze

    Returns:
        Dict of SMC features
    """
    return {
        'nearest_ob_distance': None,
        'nearest_ob_type': None,
        'nearest_fvg_distance': None,
        'nearest_fvg_type': None,
        'ob_confluence': False,
        'fvg_confluence': False,
    }
