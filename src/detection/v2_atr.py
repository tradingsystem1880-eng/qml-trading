"""
ATR Directional Change QML Pattern Detector (v2)
=================================================
Version: 2.0.0

Price-action driven detection using ATR Directional Change.
Instead of blindly iterating every N bars, this approach:
1. Processes data bar-by-bar
2. Monitors for ATR-confirmed swing points
3. Triggers pattern detection ONLY when a new market extreme is confirmed

This aligns software logic with market structure, resulting in:
- More accurate detection at actual swing confirmations
- More efficient processing (skips bars with no structural change)
- Reduced noise during consolidation periods
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
import uuid

import numpy as np
import pandas as pd

from src.core.models import Candle, CandleList, Signal, SignalType, SwingPoint, SwingType
from src.detection.base import BaseDetector, DetectorConfig


# =============================================================================
# V2 ATR CONFIGURATION
# =============================================================================

@dataclass
class ATRDetectorConfig(DetectorConfig):
    """Configuration for ATR-driven detector."""
    
    name: str = "atr_directional_change"
    version: str = "2.0.0"
    
    # ATR Directional Change parameters
    atr_lookback: int = 14  # Period for ATR calculation
    window_size: int = 200  # Lookback for pattern context
    
    # Swing detection (ATR-based)
    swing_window: int = 5  # Bars for swing confirmation
    
    # CHoCH/BoS parameters
    choch_min_break_atr: float = 0.3
    choch_confirmation_bars: int = 2
    bos_min_break_atr: float = 0.5
    bos_volume_spike_threshold: float = 1.5
    
    # QML pattern parameters
    min_head_depth_atr: float = 0.5
    max_head_depth_atr: float = 3.0


# =============================================================================
# ATR DIRECTIONAL CHANGE ENGINE
# =============================================================================

@dataclass
class LocalExtreme:
    """
    Confirmed local extreme (swing high or swing low).
    
    Created when price reverses by 1 ATR from a pending extreme.
    """
    ext_type: int       # 1 = High, -1 = Low
    index: int          # Bar index of extreme
    price: float        # Price at extreme
    timestamp: datetime
    
    conf_index: int     # Bar index of confirmation
    conf_price: float   # Price at confirmation
    conf_timestamp: datetime


class ATRDirectionalChange:
    """
    ATR-based Directional Change detector.
    
    Identifies market pivots (swing highs/lows) using Average True Range.
    A new extreme is confirmed when price reverses by 1 ATR from the
    pending extreme.
    
    This is mathematically superior to fixed-lookback swing detection
    because it adapts to current market volatility.
    
    Algorithm:
    1. Track pending maximum (potential swing high)
    2. Track pending minimum (potential swing low)
    3. When price drops 1 ATR from pending max -> CONFIRM SWING HIGH
    4. When price rises 1 ATR from pending min -> CONFIRM SWING LOW
    5. Alternate between tracking up moves and down moves
    """
    
    def __init__(self, atr_lookback: int = 14):
        """
        Initialize ATR Directional Change detector.
        
        Args:
            atr_lookback: Period for ATR calculation (default 14)
        """
        self._up_move = True  # Currently tracking upward move
        self._pend_max = np.nan  # Pending maximum (potential swing high)
        self._pend_min = np.nan  # Pending minimum (potential swing low)
        self._pend_max_i = 0     # Index of pending max
        self._pend_min_i = 0     # Index of pending min
        
        self._atr_lb = atr_lookback
        self._atr_sum = np.nan
        self._curr_atr = np.nan
        
        self.extremes: List[LocalExtreme] = []
    
    def reset(self) -> None:
        """Reset detector state for new data series."""
        self._up_move = True
        self._pend_max = np.nan
        self._pend_min = np.nan
        self._pend_max_i = 0
        self._pend_min_i = 0
        self._atr_sum = np.nan
        self._curr_atr = np.nan
        self.extremes = []
    
    def _create_extreme(
        self, 
        ext_type: str, 
        ext_i: int, 
        conf_i: int,
        time_index: pd.DatetimeIndex,
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray
    ) -> LocalExtreme:
        """Create and store a new confirmed extreme."""
        if ext_type == 'high':
            ext_type_val = 1
            arr = high
        else:
            ext_type_val = -1
            arr = low
        
        ext = LocalExtreme(
            ext_type=ext_type_val,
            index=ext_i,
            price=arr[ext_i],
            timestamp=time_index[ext_i],
            conf_index=conf_i,
            conf_price=close[conf_i],
            conf_timestamp=time_index[conf_i]
        )
        self.extremes.append(ext)
        return ext
    
    def update(
        self,
        i: int,
        time_index: pd.DatetimeIndex,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Optional[LocalExtreme]:
        """
        Process bar at index i and check for new extreme confirmation.
        
        Args:
            i: Current bar index
            time_index: Full datetime index
            high: Full high prices array
            low: Full low prices array
            close: Full close prices array
            
        Returns:
            LocalExtreme if a new extreme was confirmed, None otherwise
        """
        # Need enough bars for ATR calculation
        if i < self._atr_lb:
            return None
        
        # Initialize ATR on first valid bar
        if i == self._atr_lb:
            h_window = high[i - self._atr_lb + 1: i + 1]
            l_window = low[i - self._atr_lb + 1: i + 1]
            c_window = close[i - self._atr_lb: i]  # Lagged by 1
            
            tr1 = h_window - l_window
            tr2 = np.abs(h_window - c_window)
            tr3 = np.abs(l_window - c_window)
            self._atr_sum = np.sum(np.max(np.stack([tr1, tr2, tr3]), axis=0))
        else:
            # Rolling ATR: add newest TR, remove oldest
            tr_curr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            
            rm_i = i - self._atr_lb
            tr_remove = max(
                high[rm_i] - low[rm_i],
                abs(high[rm_i] - close[rm_i - 1]),
                abs(low[rm_i] - close[rm_i - 1])
            )
            
            self._atr_sum += tr_curr - tr_remove
        
        atr = self._atr_sum / self._atr_lb
        self._curr_atr = atr
        
        # Initialize pending extremes
        if np.isnan(self._pend_max):
            self._pend_max = high[i]
            self._pend_min = low[i]
            self._pend_max_i = self._pend_min_i = i
            return None
        
        new_extreme = None
        
        if self._up_move:
            # Tracking upward move - looking for swing HIGH
            if high[i] > self._pend_max:
                # New higher high - update pending
                self._pend_max = high[i]
                self._pend_max_i = i
            elif low[i] < self._pend_max - atr:
                # Price dropped 1 ATR from pending high -> CONFIRM SWING HIGH
                new_extreme = self._create_extreme(
                    'high', self._pend_max_i, i, time_index, high, low, close
                )
                
                # Switch to tracking downward move
                self._up_move = False
                self._pend_min = low[i]
                self._pend_min_i = i
        else:
            # Tracking downward move - looking for swing LOW
            if low[i] < self._pend_min:
                # New lower low - update pending
                self._pend_min = low[i]
                self._pend_min_i = i
            elif high[i] > self._pend_min + atr:
                # Price rose 1 ATR from pending low -> CONFIRM SWING LOW
                new_extreme = self._create_extreme(
                    'low', self._pend_min_i, i, time_index, high, low, close
                )
                
                # Switch to tracking upward move
                self._up_move = True
                self._pend_max = high[i]
                self._pend_max_i = i
        
        return new_extreme
    
    @property
    def current_atr(self) -> float:
        """Get current ATR value."""
        return self._curr_atr
    
    @property
    def extreme_count(self) -> int:
        """Get count of confirmed extremes."""
        return len(self.extremes)


# =============================================================================
# ATR DETECTOR
# =============================================================================

class ATRDetector(BaseDetector):
    """
    QML Pattern Detector using ATR Directional Change (v2.0.0).
    
    This detector processes data bar-by-bar and triggers pattern detection
    only when ATR confirms a new market extreme. This aligns detection
    with actual market structure rather than arbitrary time intervals.
    
    Algorithm:
    1. Initialize ATR Directional Change engine
    2. Process each bar sequentially
    3. When a swing is confirmed, analyze the context window
    4. Look for QML pattern formation around the confirmed swing
    5. Generate signals only for valid patterns
    
    Benefits over v1 (Rolling Window):
    - Detection triggered at confirmed swing points, not arbitrary intervals
    - ATR-adaptive: Adjusts to current market volatility
    - More efficient: Skips consolidation periods with no structural change
    - More accurate: Aligned with market structure
    
    Usage:
        config = ATRDetectorConfig(atr_lookback=14, window_size=200)
        detector = ATRDetector(config)
        signals = detector.detect(df, symbol='BTCUSDT', timeframe='1h')
    """
    
    def __init__(self, config: Optional[ATRDetectorConfig] = None):
        """
        Initialize ATR-driven detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config or ATRDetectorConfig())
        self.config: ATRDetectorConfig = self.config
        
        # ATR Directional Change engine
        self._dc: Optional[ATRDirectionalChange] = None
        
        # Pattern tracking
        self._seen_patterns: Set[str] = set()
        
        # Statistics
        self.extremes_detected = 0
        self.detection_triggers = 0
    
    def detect(
        self, 
        candles: Union[List[Candle], CandleList, pd.DataFrame],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Signal]:
        """
        Detect QML patterns using ATR-driven approach.
        
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
        self.extremes_detected = 0
        self.detection_triggers = 0
        
        # Initialize ATR Directional Change engine
        self._dc = ATRDirectionalChange(atr_lookback=self.config.atr_lookback)
        
        # Extract arrays for performance
        time_index = df['time'] if 'time' in df.columns else df.index
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.DatetimeIndex(time_index)
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        # Pre-calculate ATR for the full dataset
        full_atr = self._calculate_atr(high, low, close)
        
        signals: List[Signal] = []
        
        # Process each bar
        for i in range(n_bars):
            # Update ATR DC and check for new extreme
            new_extreme = self._dc.update(i, time_index, high, low, close)
            
            # Only trigger detection when an extreme is confirmed
            if new_extreme is not None:
                self.extremes_detected += 1
                
                # Check if we have enough lookback for pattern detection
                if i >= self.config.window_size:
                    self.detection_triggers += 1
                    
                    # Analyze the context window around this extreme
                    window_start = max(0, i - self.config.window_size + 1)
                    window_end = i + 1
                    
                    window_df = df.iloc[window_start:window_end].copy().reset_index(drop=True)
                    window_atr = full_atr[window_start:window_end]
                    
                    # Get all confirmed extremes in this window
                    window_extremes = [
                        e for e in self._dc.extremes 
                        if window_start <= e.index < window_end
                    ]
                    
                    # Find patterns using confirmed extremes
                    window_signals = self._find_patterns_from_extremes(
                        window_df, window_extremes, window_atr,
                        new_extreme, symbol, timeframe
                    )
                    
                    # Add unique signals
                    for signal in window_signals:
                        pattern_key = self._get_pattern_key(signal)
                        if pattern_key not in self._seen_patterns:
                            self._seen_patterns.add(pattern_key)
                            signals.append(signal)
        
        return signals
    
    def _find_patterns_from_extremes(
        self,
        df: pd.DataFrame,
        extremes: List[LocalExtreme],
        atr: np.ndarray,
        trigger_extreme: LocalExtreme,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Signal]:
        """
        Find QML patterns using confirmed ATR extremes.
        
        We look for 3 consecutive extremes that form a pattern:
        - Bullish: HIGH -> LOW -> HIGH (trigger is right shoulder HIGH)
        - Bearish: LOW -> HIGH -> LOW (trigger is right shoulder LOW)
        
        Args:
            df: Window DataFrame
            extremes: List of confirmed extremes in window
            atr: ATR values for window
            trigger_extreme: The extreme that triggered this detection
            symbol: Trading pair
            timeframe: Candle timeframe
        
        Returns:
            List of signals found
        """
        signals = []
        
        # Need at least 3 extremes to form a pattern
        if len(extremes) < 3:
            return signals
        
        # Find the trigger's position in the extremes list
        trigger_idx = None
        for i, e in enumerate(extremes):
            if e.index == trigger_extreme.index and e.ext_type == trigger_extreme.ext_type:
                trigger_idx = i
                break
        
        if trigger_idx is None or trigger_idx < 2:
            return signals  # Need at least 2 prior extremes
        
        # Get the last 3 extremes ending with the trigger
        e1 = extremes[trigger_idx - 2]  # Left shoulder
        e2 = extremes[trigger_idx - 1]  # Head
        e3 = extremes[trigger_idx]      # Right shoulder (trigger)
        
        # Check for BULLISH pattern: HIGH -> LOW -> HIGH
        if e1.ext_type == 1 and e2.ext_type == -1 and e3.ext_type == 1:
            signal = self._validate_bullish_pattern(
                e1, e2, e3, atr, symbol, timeframe
            )
            if signal:
                signals.append(signal)
        
        # Check for BEARISH pattern: LOW -> HIGH -> LOW
        elif e1.ext_type == -1 and e2.ext_type == 1 and e3.ext_type == -1:
            signal = self._validate_bearish_pattern(
                e1, e2, e3, atr, symbol, timeframe
            )
            if signal:
                signals.append(signal)
        
        return signals
    
    def _validate_bullish_pattern(
        self,
        left_shoulder: LocalExtreme,  # HIGH
        head: LocalExtreme,           # LOW
        right_shoulder: LocalExtreme, # HIGH
        atr: np.ndarray,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[Signal]:
        """
        Validate and create a bullish QML pattern.
        
        Pattern structure: Left shoulder HIGH -> Head LOW -> Right shoulder HIGH
        Entry is at right shoulder confirmation.
        """
        left_price = left_shoulder.price
        head_price = head.price
        right_price = right_shoulder.price
        head_idx = head.index
        
        # Head must be below both shoulders
        if head_price >= left_price or head_price >= right_price:
            return None
        
        # Shoulders should be at similar levels (within 10% for volatile crypto)
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > 0.10:
            return None
        
        # Calculate head depth in ATR
        local_atr = atr[min(head_idx, len(atr) - 1)]
        if np.isnan(local_atr):
            local_atr = np.nanmean(atr)
        
        head_depth = left_price - head_price
        head_depth_atr = head_depth / local_atr if local_atr > 0 else 0
        
        # Validate head depth
        if head_depth_atr < self.config.min_head_depth_atr:
            return None
        if head_depth_atr > self.config.max_head_depth_atr:
            return None
        
        # Calculate validity score
        validity_score = self._calculate_validity_score(head_depth_atr, shoulder_diff)
        
        if validity_score < self.config.min_validity_score:
            return None
        
        # Entry at right shoulder confirmation
        entry_price = right_price
        
        # Stop loss below head with ATR buffer
        stop_loss = head_price - (local_atr * self.config.stop_loss_atr_mult)
        
        # Take profits
        risk = entry_price - stop_loss
        take_profit_1 = entry_price + risk
        take_profit_2 = entry_price + (2 * risk)
        take_profit_3 = entry_price + (3 * risk)
        
        return Signal(
            timestamp=right_shoulder.conf_timestamp,
            signal_type=SignalType.BUY,
            price=entry_price,
            strategy_name=f"QML_ATR_v{self.config.version}",
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
            atr_at_signal=local_atr,
            metadata={
                'head_price': head_price,
                'head_time': str(head.timestamp),
                'head_bar_index': head_idx,
                'left_shoulder_price': left_price,
                'left_shoulder_time': str(left_shoulder.timestamp),
                'right_shoulder_price': right_price,
                'right_shoulder_time': str(right_shoulder.timestamp),
                'head_depth_atr': head_depth_atr,
                'shoulder_diff_pct': round(shoulder_diff * 100, 2),
                'detector_version': self.config.version,
                'detection_method': 'atr_directional_change',
            }
        )
    
    def _validate_bearish_pattern(
        self,
        left_shoulder: LocalExtreme,  # LOW
        head: LocalExtreme,           # HIGH
        right_shoulder: LocalExtreme, # LOW
        atr: np.ndarray,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[Signal]:
        """
        Validate and create a bearish QML pattern.
        
        Pattern structure: Left shoulder LOW -> Head HIGH -> Right shoulder LOW
        Entry is at right shoulder confirmation.
        """
        left_price = left_shoulder.price
        head_price = head.price
        right_price = right_shoulder.price
        head_idx = head.index
        
        # Head must be above both shoulders
        if head_price <= left_price or head_price <= right_price:
            return None
        
        # Shoulders should be at similar levels (within 10% for volatile crypto)
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > 0.10:
            return None
        
        # Calculate head depth in ATR
        local_atr = atr[min(head_idx, len(atr) - 1)]
        if np.isnan(local_atr):
            local_atr = np.nanmean(atr)
        
        head_depth = head_price - left_price
        head_depth_atr = head_depth / local_atr if local_atr > 0 else 0
        
        # Validate head depth
        if head_depth_atr < self.config.min_head_depth_atr:
            return None
        if head_depth_atr > self.config.max_head_depth_atr:
            return None
        
        # Calculate validity score
        validity_score = self._calculate_validity_score(head_depth_atr, shoulder_diff)
        
        if validity_score < self.config.min_validity_score:
            return None
        
        # Entry at right shoulder
        entry_price = right_price
        
        # Stop loss above head with ATR buffer
        stop_loss = head_price + (local_atr * self.config.stop_loss_atr_mult)
        
        # Take profits
        risk = stop_loss - entry_price
        take_profit_1 = entry_price - risk
        take_profit_2 = entry_price - (2 * risk)
        take_profit_3 = entry_price - (3 * risk)
        
        return Signal(
            timestamp=right_shoulder.conf_timestamp,
            signal_type=SignalType.SELL,
            price=entry_price,
            strategy_name=f"QML_ATR_v{self.config.version}",
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
            atr_at_signal=local_atr,
            metadata={
                'head_price': head_price,
                'head_time': str(head.timestamp),
                'head_bar_index': head_idx,
                'left_shoulder_price': left_price,
                'left_shoulder_time': str(left_shoulder.timestamp),
                'right_shoulder_price': right_price,
                'right_shoulder_time': str(right_shoulder.timestamp),
                'head_depth_atr': head_depth_atr,
                'shoulder_diff_pct': round(shoulder_diff * 100, 2),
                'detector_version': self.config.version,
                'detection_method': 'atr_directional_change',
            }
        )
    
    def _calculate_validity_score(
        self,
        head_depth_atr: float,
        shoulder_diff: float
    ) -> float:
        """
        Calculate pattern validity score (0.0 to 1.0).
        
        Components:
        - Head depth quality (50%)
        - Shoulder symmetry (30%)
        - ATR confirmation bonus (20%)
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
        score += head_score * 0.5
        
        # Shoulder symmetry
        symmetry_score = max(0, 1.0 - (shoulder_diff * 20))
        score += symmetry_score * 0.3
        
        # ATR confirmation bonus (always get this since we use ATR DC)
        score += 0.2
        
        return round(score, 3)
    
    def _get_pattern_key(self, signal: Signal) -> str:
        """Generate unique key for pattern deduplication."""
        head_time = signal.metadata.get('head_time', '')
        left_time = signal.metadata.get('left_shoulder_time', '')
        pattern_type = signal.pattern_type or ''
        
        return f"{head_time}_{left_time}_{pattern_type}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with detection metrics
        """
        return {
            'extremes_detected': self.extremes_detected,
            'detection_triggers': self.detection_triggers,
            'patterns_found': len(self._seen_patterns),
            'current_atr': self._dc.current_atr if self._dc else None,
        }
