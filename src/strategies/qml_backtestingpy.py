"""
QML Strategy for Backtesting.py
================================
Version: 2.0.0

Properly implemented backtesting.py Strategy using framework conventions.
Uses the framework's built-in position management and order execution.

Usage:
    from backtesting import Backtest
    from src.strategies.qml_backtestingpy import QMLStrategy, prepare_backtesting_data
    from src.data_engine import load_master_data
    
    df = load_master_data(timeframe='4h')
    df = prepare_backtesting_data(df)
    
    bt = Backtest(df, QMLStrategy, cash=100000, commission=0.001)
    stats = bt.run()
    print(stats)
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from backtesting import Strategy


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_backtesting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for backtesting.py compatibility.
    
    Converts columns to capitalized format and sets DatetimeIndex.
    
    Args:
        df: DataFrame from load_master_data() with columns like
            ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR']
            
    Returns:
        DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        and DatetimeIndex.
    """
    df = df.copy()
    
    # Rename time column to Date
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'Date'})
    
    # Handle lowercase columns (normalize to capitalized)
    column_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Drop ATR (strategy calculates its own)
    if 'ATR' in df.columns:
        df = df.drop(columns=['ATR'])
    
    # Set DatetimeIndex
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    # Return only required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df[[c for c in required if c in df.columns]]


# =============================================================================
# ATR INDICATOR FUNCTION (for self.I())
# =============================================================================

def compute_atr(high, low, close, period=14):
    """
    Compute ATR for use with self.I().
    
    Args:
        high: High prices array
        low: Low prices array  
        close: Close prices array
        period: ATR period
        
    Returns:
        ATR array
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    n = len(high)
    tr = np.zeros(n)
    atr_values = np.full(n, np.nan)
    
    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # ATR via EMA
    if n >= period:
        atr_values[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i - 1]
    
    return atr_values


# =============================================================================
# QML STRATEGY
# =============================================================================

class QMLStrategy(Strategy):
    """
    QML (Quasimodo) Pattern Strategy using ATR Directional Change.
    
    Detects 3-point patterns:
    - Bullish: HIGH -> LOW -> HIGH (head is the low between shoulders)
    - Bearish: LOW -> HIGH -> LOW (head is the high between shoulders)
    
    All parameters are optimizable via Backtest.optimize().
    
    Pattern Registration:
        Set _register_patterns = True and _pattern_registry to a PatternRegistry
        instance to store detected patterns for ML training.
    """
    
    # Strategy parameters (class-level for optimization)
    atr_period = 14
    min_depth_ratio = 0.5
    max_depth_ratio = 3.0
    stop_loss_atr = 1.5
    take_profit_rr = 2.0  # Reduced for faster trade closes
    shoulder_tolerance = 0.10
    
    # Pattern registry integration (class-level for strategy inheritance)
    _register_patterns = False
    _pattern_registry = None
    _feature_extractor = None
    _symbol = "BTC/USDT"
    _timeframe = "4h"
    
    def init(self):
        """Initialize indicators."""
        # Pre-compute ATR using self.I()
        self.atr = self.I(
            compute_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
        )
        
        # Swing detection state
        self._up_move = True
        self._pend_max = np.nan
        self._pend_min = np.nan
        self._pend_max_i = 0
        self._pend_min_i = 0
        
        # Confirmed extremes: List of (type, index, price)
        # type: 1 = HIGH, -1 = LOW
        self._extremes: List[Tuple[int, int, float]] = []
    
    def next(self):
        """Process each bar - detect swings and patterns."""
        i = len(self.data) - 1
        
        # Need enough bars for ATR
        if i < self.atr_period:
            return
        
        current_atr = self.atr[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            return
        
        # Skip if already in a position (let SL/TP manage exit)
        if self.position:
            return
        
        high_i = self.data.High[-1]
        low_i = self.data.Low[-1]
        
        # Initialize pending extremes
        if np.isnan(self._pend_max):
            self._pend_max = high_i
            self._pend_min = low_i
            self._pend_max_i = i
            self._pend_min_i = i
            return
        
        # ATR Directional Change logic
        new_extreme = None
        
        if self._up_move:
            if high_i > self._pend_max:
                self._pend_max = high_i
                self._pend_max_i = i
            elif low_i < self._pend_max - current_atr:
                # Confirm swing HIGH
                new_extreme = (1, self._pend_max_i, self._pend_max)
                self._extremes.append(new_extreme)
                self._up_move = False
                self._pend_min = low_i
                self._pend_min_i = i
        else:
            if low_i < self._pend_min:
                self._pend_min = low_i
                self._pend_min_i = i
            elif high_i > self._pend_min + current_atr:
                # Confirm swing LOW
                new_extreme = (-1, self._pend_min_i, self._pend_min)
                self._extremes.append(new_extreme)
                self._up_move = True
                self._pend_max = high_i
                self._pend_max_i = i
        
        # Check for pattern when we have a new extreme and 3+ total
        if new_extreme is not None and len(self._extremes) >= 3:
            self._check_and_trade(current_atr)
    
    def _check_and_trade(self, current_atr: float):
        """Check for valid pattern and execute trade."""
        e1 = self._extremes[-3]  # Left shoulder
        e2 = self._extremes[-2]  # Head
        e3 = self._extremes[-1]  # Right shoulder
        
        # BULLISH: HIGH -> LOW -> HIGH
        if e1[0] == 1 and e2[0] == -1 and e3[0] == 1:
            self._try_bullish_trade(e1, e2, e3, current_atr)
        
        # BEARISH: LOW -> HIGH -> LOW
        elif e1[0] == -1 and e2[0] == 1 and e3[0] == -1:
            self._try_bearish_trade(e1, e2, e3, current_atr)
    
    def _try_bullish_trade(self, left, head, right, atr):
        """Validate and execute bullish trade."""
        left_price, head_price, right_price = left[2], head[2], right[2]
        
        # Head must be below both shoulders
        if head_price >= left_price or head_price >= right_price:
            return
        
        # Shoulders at similar levels
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > self.shoulder_tolerance:
            return
        
        # Head depth validation
        head_depth = left_price - head_price
        head_depth_atr = head_depth / atr
        if not (self.min_depth_ratio <= head_depth_atr <= self.max_depth_ratio):
            return
        
        # Calculate levels
        entry = self.data.Close[-1]
        sl = head_price - (atr * self.stop_loss_atr)
        
        # SL must be below entry for long
        if sl >= entry:
            return
        
        risk = entry - sl
        tp = entry + (risk * self.take_profit_rr)
        
        # Register pattern before trade
        self._register_pattern('bullish', left, head, right, atr, entry, sl, tp)
        
        # Execute trade
        self.buy(sl=sl, tp=tp)
    
    def _try_bearish_trade(self, left, head, right, atr):
        """Validate and execute bearish trade."""
        left_price, head_price, right_price = left[2], head[2], right[2]
        
        # Head must be above both shoulders
        if head_price <= left_price or head_price <= right_price:
            return
        
        # Shoulders at similar levels
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > self.shoulder_tolerance:
            return
        
        # Head depth validation
        head_depth = head_price - left_price
        head_depth_atr = head_depth / atr
        if not (self.min_depth_ratio <= head_depth_atr <= self.max_depth_ratio):
            return
        
        # Calculate levels
        entry = self.data.Close[-1]
        sl = head_price + (atr * self.stop_loss_atr)
        
        # SL must be above entry for short
        if sl <= entry:
            return
        
        risk = sl - entry
        tp = entry - (risk * self.take_profit_rr)
        
        # TP must be below entry for short
        if tp >= entry:
            return
        
        # Register pattern before trade
        self._register_pattern('bearish', left, head, right, atr, entry, sl, tp)
        
        # Execute trade
        self.sell(sl=sl, tp=tp)
    
    def _register_pattern(self, pattern_type, left, head, right, atr, entry, sl, tp):
        """Register detected pattern to the ML registry."""
        if not self._register_patterns or self._pattern_registry is None:
            return
        
        try:
            i = len(self.data) - 1
            detection_time = self.data.index[-1] if hasattr(self.data.index[-1], 'isoformat') else None
            
            # Build pattern data
            pattern_data = {
                'symbol': self._symbol,
                'timeframe': self._timeframe,
                'pattern_type': pattern_type,
                'detection_time': detection_time,
                'detection_idx': i,
                'left_shoulder_price': left[2],
                'left_shoulder_idx': left[1],
                'head_price': head[2],
                'head_idx': head[1],
                'right_shoulder_price': right[2],
                'right_shoulder_idx': right[1],
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'atr': atr,
                'validity_score': 0.7,  # Basic validation passed
            }
            
            # Extract features if extractor is available
            if self._feature_extractor is not None:
                # Convert backtesting.py data to DataFrame
                df = self._data_to_df()
                features = self._feature_extractor.extract_pattern_features(pattern_data, df, i)
            else:
                # Minimal features
                features = {
                    'geo_head_depth_atr': abs(left[2] - head[2]) / atr,
                    'geo_shoulder_symmetry': right[2] / left[2] if left[2] > 0 else 1.0,
                }
            
            # Register pattern
            self._pattern_registry.register_pattern(pattern_data, features)
            
        except Exception as e:
            # Don't let registration failures stop trading
            pass
    
    def _data_to_df(self):
        """Convert backtesting.py data to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            'time': self.data.index,
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume if hasattr(self.data, 'Volume') else 0,
        })
