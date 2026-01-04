"""
Rolling Window QML Pattern Detector (v1)
=========================================
Version: 1.1.0

Detection algorithm that slides a fixed-size window across historical data
and runs pattern detection at regular intervals.

This is the "v1" approach preserved for comparison and backward compatibility.
For new development, prefer v2_atr.py which uses ATR-driven detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
import uuid

import numpy as np
import pandas as pd

from src.core.models import Candle, CandleList, Signal, SignalType, PatternDirection
from src.detection.base import BaseDetector, DetectorConfig


# =============================================================================
# V1 ROLLING WINDOW CONFIGURATION
# =============================================================================

@dataclass
class RollingWindowConfig(DetectorConfig):
    """Configuration for rolling window detector."""
    
    name: str = "rolling_window"
    version: str = "1.1.0"
    
    # Rolling window parameters
    window_size: int = 200  # Number of bars per detection window
    step_size: int = 12     # Bars to advance between detections
    
    # Swing detection
    swing_window: int = 5   # Bars on each side for swing confirmation
    
    # CHoCH/BoS parameters
    choch_min_break_atr: float = 0.3
    choch_confirmation_bars: int = 2
    bos_min_break_atr: float = 0.5
    bos_volume_spike_threshold: float = 1.5
    
    # QML pattern parameters
    min_head_depth_atr: float = 0.5
    max_head_depth_atr: float = 3.0


# =============================================================================
# ROLLING WINDOW DETECTOR
# =============================================================================

class RollingWindowDetector(BaseDetector):
    """
    QML Pattern Detector using Rolling Window approach (v1.1.0).
    
    This detector slides a fixed-size window across the data and
    runs pattern detection at each step. It's conceptually simple
    but may miss patterns between step intervals.
    
    Algorithm:
    1. Start at bar 0 with window of size `window_size`
    2. Run full detection pipeline on the window
    3. Advance by `step_size` bars
    4. Repeat until end of data
    5. Deduplicate patterns by key (head_time + left_shoulder_time)
    
    Usage:
        config = RollingWindowConfig(window_size=200, step_size=12)
        detector = RollingWindowDetector(config)
        signals = detector.detect(df, symbol='BTCUSDT', timeframe='1h')
    """
    
    def __init__(self, config: Optional[RollingWindowConfig] = None):
        """
        Initialize rolling window detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config or RollingWindowConfig())
        self.config: RollingWindowConfig = self.config
        
        # Pattern tracking for deduplication
        self._seen_patterns: Set[str] = set()
        self._detected_patterns: List[Dict[str, Any]] = []
    
    def detect(
        self, 
        candles: Union[List[Candle], CandleList, pd.DataFrame],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Signal]:
        """
        Detect QML patterns using rolling window approach.
        
        Args:
            candles: Price data (DataFrame or list of Candles)
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (e.g., '4h')
        
        Returns:
            List of Signal objects for detected patterns
        """
        # Normalize input to DataFrame
        df = self._normalize_candles(candles)
        n_bars = len(df)
        
        # Validate data
        if n_bars < self.config.window_size:
            return []
        
        # Reset state
        self._seen_patterns = set()
        self._detected_patterns = []
        
        # Calculate ATR for the full dataset
        atr = self._calculate_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        
        # Calculate number of windows
        n_windows = (n_bars - self.config.window_size) // self.config.step_size + 1
        
        signals: List[Signal] = []
        
        # Slide window across data
        for window_idx in range(n_windows):
            start_idx = window_idx * self.config.step_size
            end_idx = start_idx + self.config.window_size
            
            if end_idx > n_bars:
                break
            
            # Extract window
            window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            window_atr = atr[start_idx:end_idx]
            
            # Detect patterns in this window
            window_signals = self._detect_in_window(
                window_df, window_atr, symbol, timeframe
            )
            
            # Add unique signals
            for signal in window_signals:
                pattern_key = self._get_pattern_key(signal)
                if pattern_key not in self._seen_patterns:
                    self._seen_patterns.add(pattern_key)
                    signals.append(signal)
        
        return signals
    
    def _detect_in_window(
        self,
        df: pd.DataFrame,
        atr: np.ndarray,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Signal]:
        """
        Detect patterns within a single window.
        
        Args:
            df: Window DataFrame
            atr: ATR values for the window
            symbol: Trading pair
            timeframe: Candle timeframe
        
        Returns:
            List of signals found in this window
        """
        signals = []
        n = len(df)
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(df, 'high')
        swing_lows = self._find_swing_points(df, 'low')
        
        # Need at least 3 swing points to form a pattern
        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return signals
        
        # Look for QML bullish patterns (swing low as head)
        bullish_signals = self._find_bullish_patterns(
            df, swing_highs, swing_lows, atr, symbol, timeframe
        )
        signals.extend(bullish_signals)
        
        # Look for QML bearish patterns (swing high as head)
        bearish_signals = self._find_bearish_patterns(
            df, swing_highs, swing_lows, atr, symbol, timeframe
        )
        signals.extend(bearish_signals)
        
        return signals
    
    def _find_swing_points(
        self, 
        df: pd.DataFrame, 
        price_col: str
    ) -> List[Dict[str, Any]]:
        """
        Find swing highs or lows in the data.
        
        Args:
            df: OHLCV DataFrame
            price_col: 'high' for swing highs, 'low' for swing lows
        
        Returns:
            List of swing point dictionaries
        """
        swings = []
        prices = df[price_col].values
        n = len(prices)
        window = self.config.swing_window
        
        for i in range(window, n - window):
            if price_col == 'high':
                # Swing high: higher than surrounding bars
                is_swing = all(
                    prices[i] > prices[i - j] and prices[i] > prices[i + j]
                    for j in range(1, window + 1)
                )
            else:
                # Swing low: lower than surrounding bars
                is_swing = all(
                    prices[i] < prices[i - j] and prices[i] < prices[i + j]
                    for j in range(1, window + 1)
                )
            
            if is_swing:
                swings.append({
                    'index': i,
                    'price': prices[i],
                    'time': df.iloc[i]['time'] if 'time' in df.columns else i,
                    'type': 'high' if price_col == 'high' else 'low'
                })
        
        return swings
    
    def _find_bullish_patterns(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        atr: np.ndarray,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Signal]:
        """
        Find bullish QML patterns (head is a swing low).
        
        Pattern structure:
        - Left shoulder: swing high (P1)
        - Head: swing low below neckline (P3)
        - Right shoulder: swing high near left shoulder (P5)
        - BUY signal on P5 break
        """
        signals = []
        
        # Need at least 2 highs and 1 low
        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return signals
        
        # For each potential head (swing low)
        for head in swing_lows:
            head_idx = head['index']
            head_price = head['price']
            head_time = head['time']
            
            # Find left shoulder (swing high before head)
            left_shoulders = [sh for sh in swing_highs if sh['index'] < head_idx]
            if not left_shoulders:
                continue
            left_shoulder = left_shoulders[-1]  # Most recent before head
            
            # Find right shoulder (swing high after head)
            right_shoulders = [sh for sh in swing_highs if sh['index'] > head_idx]
            if not right_shoulders:
                continue
            right_shoulder = right_shoulders[0]  # First after head
            
            # Validate pattern geometry
            left_price = left_shoulder['price']
            right_price = right_shoulder['price']
            
            # Head must be below both shoulders
            if head_price >= left_price or head_price >= right_price:
                continue
            
            # Shoulders should be at similar levels (within 2%)
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > 0.02:
                continue
            
            # Calculate head depth in ATR
            head_atr = atr[head_idx] if not np.isnan(atr[head_idx]) else np.nanmean(atr)
            head_depth = left_price - head_price
            head_depth_atr = head_depth / head_atr if head_atr > 0 else 0
            
            # Validate head depth
            if head_depth_atr < self.config.min_head_depth_atr:
                continue
            if head_depth_atr > self.config.max_head_depth_atr:
                continue
            
            # Calculate validity score
            validity_score = self._calculate_validity_score(
                head_depth_atr, shoulder_diff, left_price, right_price, head_price
            )
            
            if validity_score < self.config.min_validity_score:
                continue
            
            # Entry is at right shoulder (neckline break)
            entry_price = right_price
            
            # Stop loss below head with ATR buffer
            stop_loss = head_price - (head_atr * self.config.stop_loss_atr_mult)
            
            # Take profits at R:R ratios
            risk = entry_price - stop_loss
            take_profit_1 = entry_price + risk  # 1R
            take_profit_2 = entry_price + (2 * risk)  # 2R
            take_profit_3 = entry_price + (3 * risk)  # 3R
            
            # Get signal timestamp (right shoulder time)
            signal_time = right_shoulder['time']
            if not isinstance(signal_time, datetime):
                signal_time = df.iloc[right_shoulder['index']]['time']
            
            # Create signal
            signal = Signal(
                timestamp=signal_time,
                signal_type=SignalType.BUY,
                price=entry_price,
                strategy_name=f"QML_v{self.config.version}",
                confidence=validity_score,
                validity_score=validity_score,
                stop_loss=stop_loss,
                take_profit=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                pattern_type="QML_BULLISH",
                pattern_id=str(uuid.uuid4())[:8],
                symbol=symbol,
                timeframe=timeframe,
                atr_at_signal=head_atr,
                metadata={
                    'head_price': head_price,
                    'head_time': str(head_time),
                    'left_shoulder_price': left_price,
                    'left_shoulder_time': str(left_shoulder['time']),
                    'right_shoulder_price': right_price,
                    'head_depth_atr': head_depth_atr,
                    'detector_version': self.config.version,
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _find_bearish_patterns(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        atr: np.ndarray,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Signal]:
        """
        Find bearish QML patterns (head is a swing high).
        
        Pattern structure:
        - Left shoulder: swing low (P1)
        - Head: swing high above neckline (P3)
        - Right shoulder: swing low near left shoulder (P5)
        - SELL signal on P5 break
        """
        signals = []
        
        # Need at least 2 lows and 1 high
        if len(swing_lows) < 2 or len(swing_highs) < 1:
            return signals
        
        # For each potential head (swing high)
        for head in swing_highs:
            head_idx = head['index']
            head_price = head['price']
            head_time = head['time']
            
            # Find left shoulder (swing low before head)
            left_shoulders = [sl for sl in swing_lows if sl['index'] < head_idx]
            if not left_shoulders:
                continue
            left_shoulder = left_shoulders[-1]
            
            # Find right shoulder (swing low after head)
            right_shoulders = [sl for sl in swing_lows if sl['index'] > head_idx]
            if not right_shoulders:
                continue
            right_shoulder = right_shoulders[0]
            
            # Validate pattern geometry
            left_price = left_shoulder['price']
            right_price = right_shoulder['price']
            
            # Head must be above both shoulders
            if head_price <= left_price or head_price <= right_price:
                continue
            
            # Shoulders should be at similar levels
            shoulder_diff = abs(left_price - right_price) / left_price
            if shoulder_diff > 0.02:
                continue
            
            # Calculate head depth in ATR
            head_atr = atr[head_idx] if not np.isnan(atr[head_idx]) else np.nanmean(atr)
            head_depth = head_price - left_price
            head_depth_atr = head_depth / head_atr if head_atr > 0 else 0
            
            # Validate head depth
            if head_depth_atr < self.config.min_head_depth_atr:
                continue
            if head_depth_atr > self.config.max_head_depth_atr:
                continue
            
            # Calculate validity score
            validity_score = self._calculate_validity_score(
                head_depth_atr, shoulder_diff, left_price, right_price, head_price
            )
            
            if validity_score < self.config.min_validity_score:
                continue
            
            # Entry is at right shoulder
            entry_price = right_price
            
            # Stop loss above head
            stop_loss = head_price + (head_atr * self.config.stop_loss_atr_mult)
            
            # Take profits
            risk = stop_loss - entry_price
            take_profit_1 = entry_price - risk
            take_profit_2 = entry_price - (2 * risk)
            take_profit_3 = entry_price - (3 * risk)
            
            signal_time = right_shoulder['time']
            if not isinstance(signal_time, datetime):
                signal_time = df.iloc[right_shoulder['index']]['time']
            
            signal = Signal(
                timestamp=signal_time,
                signal_type=SignalType.SELL,
                price=entry_price,
                strategy_name=f"QML_v{self.config.version}",
                confidence=validity_score,
                validity_score=validity_score,
                stop_loss=stop_loss,
                take_profit=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                pattern_type="QML_BEARISH",
                pattern_id=str(uuid.uuid4())[:8],
                symbol=symbol,
                timeframe=timeframe,
                atr_at_signal=head_atr,
                metadata={
                    'head_price': head_price,
                    'head_time': str(head_time),
                    'left_shoulder_price': left_price,
                    'left_shoulder_time': str(left_shoulder['time']),
                    'right_shoulder_price': right_price,
                    'head_depth_atr': head_depth_atr,
                    'detector_version': self.config.version,
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_validity_score(
        self,
        head_depth_atr: float,
        shoulder_diff: float,
        left_price: float,
        right_price: float,
        head_price: float
    ) -> float:
        """
        Calculate pattern validity score (0.0 to 1.0).
        
        Components:
        - Head depth quality (40%)
        - Shoulder symmetry (30%)
        - Pattern geometry (30%)
        """
        score = 0.0
        
        # Head depth quality (optimal is 1-2 ATR)
        if 1.0 <= head_depth_atr <= 2.0:
            head_score = 1.0
        elif 0.5 <= head_depth_atr < 1.0:
            head_score = 0.7
        elif 2.0 < head_depth_atr <= 3.0:
            head_score = 0.7
        else:
            head_score = 0.4
        score += head_score * 0.4
        
        # Shoulder symmetry (lower diff is better)
        symmetry_score = max(0, 1.0 - (shoulder_diff * 20))  # 0% diff = 1.0, 5% diff = 0
        score += symmetry_score * 0.3
        
        # Pattern geometry (head properly positioned)
        avg_shoulder = (left_price + right_price) / 2
        geometry_ratio = abs(head_price - avg_shoulder) / avg_shoulder
        geometry_score = min(1.0, geometry_ratio * 10)  # More separation = better
        score += geometry_score * 0.3
        
        return round(score, 3)
    
    def _get_pattern_key(self, signal: Signal) -> str:
        """
        Generate unique key for pattern deduplication.
        
        Args:
            signal: Signal to generate key for
        
        Returns:
            Unique string key
        """
        head_time = signal.metadata.get('head_time', '')
        left_time = signal.metadata.get('left_shoulder_time', '')
        pattern_type = signal.pattern_type or ''
        
        return f"{head_time}_{left_time}_{pattern_type}"
