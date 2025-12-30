#!/usr/bin/env python3
"""
ATR DIRECTIONAL CHANGE QML PATTERN DETECTOR
============================================
Version: 2.0.0

Refactored from v1.1.0 Rolling Window approach to use ATR Directional Change
as the primary detection engine.

OLD (v1.1.0): Blindly slice data every X bars (step_size=12)
NEW (v2.0.0): Watch every bar, trigger detection only when ATR confirms
             a new market extreme (top or bottom)

Why This Works:
===============
QML patterns are composed of swing highs and lows (Head, Shoulders).
By using an ATR-based pivot detector to drive the pattern detector,
we align software logic with market structure. Detection runs only
at mathematically significant pivot points, not arbitrary intervals.

Benefits:
- More accurate: Detection triggered at actual swing confirmations
- More efficient: Skip bars where nothing structural happens
- Reduced noise: Fewer redundant pattern checks during consolidation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
import json

from src.detection.detector import QMLDetector, DetectorConfig
from src.data.models import QMLPattern, PatternType

# Try to import ccxt for data fetching
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


# =============================================================================
# LOCAL EXTREME DATACLASS (Inline to avoid import issues)
# =============================================================================

@dataclass
class LocalExtreme:
    """
    Represents a confirmed local extreme (swing high or swing low).
    
    Attributes:
        ext_type: 1 for swing high, -1 for swing low
        index: Bar index of the extreme
        price: Price at the extreme point
        timestamp: Datetime of the extreme
        conf_index: Bar index where the extreme was confirmed
        conf_price: Price at confirmation
        conf_timestamp: Datetime of confirmation
    """
    ext_type: int       # 1 = High, -1 = Low
    index: int          # Bar index of extreme
    price: float        # Price at extreme
    timestamp: pd.Timestamp
    
    conf_index: int     # Bar index of confirmation
    conf_price: float   # Price at confirmation
    conf_timestamp: pd.Timestamp


# =============================================================================
# ATR DIRECTIONAL CHANGE CLASS
# =============================================================================

class ATRDirectionalChange:
    """
    ATR-based Directional Change detector.
    
    Identifies market pivots (swing highs/lows) using Average True Range.
    A new extreme is confirmed when price reverses by 1 ATR from the
    pending extreme.
    
    This is mathematically superior to fixed-lookback swing detection
    because it adapts to current market volatility.
    """
    
    def __init__(self, atr_lookback: int = 14):
        """
        Initialize the ATR Directional Change detector.
        
        Args:
            atr_lookback: Period for ATR calculation (default 14)
        """
        self._up_move = True  # Currently tracking upward move (last confirmed is a low)
        self._pend_max = np.nan  # Pending maximum (potential swing high)
        self._pend_min = np.nan  # Pending minimum (potential swing low)
        self._pend_max_i = 0     # Index of pending max
        self._pend_min_i = 0     # Index of pending min
        
        self._atr_lb = atr_lookback
        self._atr_sum = np.nan
        self._curr_atr = np.nan
        
        self.extremes: List[LocalExtreme] = []
    
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
# ATR-DRIVEN ROLLING PATTERN DETECTOR
# =============================================================================

class ATRPatternDetector:
    """
    QML Pattern Detection driven by ATR Directional Change.
    
    Instead of blindly iterating every N bars:
    - Processes data bar-by-bar
    - Monitors ATR Directional Change for confirmed pivots
    - Triggers pattern detection ONLY when a new swing is confirmed
    
    This aligns detection with market structure rather than arbitrary time steps.
    """
    
    def __init__(
        self, 
        window_size: int = 200,
        atr_lookback: int = 14,
        params_path: Optional[str] = None
    ):
        """
        Initialize ATR-driven pattern detector.
        
        Args:
            window_size: Number of bars to analyze for pattern detection
            atr_lookback: ATR period for directional change detection
            params_path: Optional path to params.json for configuration
        """
        # Load params from JSON if provided
        if params_path:
            with open(params_path, 'r') as f:
                params = json.load(f)
            self.window_size = params.get('parameters', {}).get('window_size', window_size)
            self.atr_lookback = params.get('parameters', {}).get('atr_period', atr_lookback)
        else:
            self.window_size = window_size
            self.atr_lookback = atr_lookback
        
        # Core components
        self.detector = QMLDetector()
        self.dc: Optional[ATRDirectionalChange] = None
        
        # Pattern tracking
        self.detected_patterns: List[QMLPattern] = []
        self.seen_patterns: Set[str] = set()
        
        # Statistics
        self.extremes_detected = 0
        self.detection_triggers = 0
    
    def _pattern_key(self, pattern: QMLPattern) -> str:
        """Create unique key for pattern deduplication."""
        return f"{pattern.head_time}_{pattern.left_shoulder_time}_{pattern.pattern_type.value}"
    
    def detect_all(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> List[QMLPattern]:
        """
        Detect all QML patterns using ATR-driven approach.
        
        Args:
            df: Full OHLCV DataFrame with 'time', 'high', 'low', 'close' columns
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            
        Returns:
            List of unique detected QMLPattern objects
        """
        n_bars = len(df)
        
        if n_bars < self.window_size:
            print(f"⚠️ Insufficient data: {n_bars} bars < {self.window_size} window")
            return []
        
        print(f"\n{'='*70}")
        print(f"  ATR DIRECTIONAL CHANGE QML DETECTOR v2.0.0")
        print(f"{'='*70}")
        print(f"  Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"  Data: {n_bars} bars")
        print(f"  Window Size: {self.window_size} bars")
        print(f"  ATR Lookback: {self.atr_lookback} bars")
        print(f"{'='*70}")
        
        # Reset state
        self.detected_patterns = []
        self.seen_patterns = set()
        self.extremes_detected = 0
        self.detection_triggers = 0
        
        # Initialize ATR Directional Change detector
        self.dc = ATRDirectionalChange(atr_lookback=self.atr_lookback)
        
        # Extract numpy arrays for performance
        time_index = df['time'] if 'time' in df.columns else df.index
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.DatetimeIndex(time_index)
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        # =====================================================================
        # MAIN LOOP: Bar-by-bar iteration with ATR-driven triggers
        # =====================================================================
        
        print(f"\n🔄 Processing {n_bars} bars with ATR directional change...")
        
        for i in range(n_bars):
            # Update ATR DC and check for new extreme
            new_extreme = self.dc.update(i, time_index, high, low, close)
            
            # =================================================================
            # THE TRIGGER: Only run pattern detection when extreme is confirmed
            # =================================================================
            if new_extreme is not None:
                self.extremes_detected += 1
                
                ext_type_str = "HIGH" if new_extreme.ext_type == 1 else "LOW"
                
                # Check if we have enough lookback for pattern detection
                if i >= self.window_size:
                    self.detection_triggers += 1
                    
                    # Slice the context window (lookback from confirmation point)
                    window_start = max(0, i - self.window_size + 1)
                    window_end = i + 1
                    window_df = df.iloc[window_start:window_end].copy().reset_index(drop=True)
                    
                    # Run QML detector on this window
                    patterns = self.detector.detect(symbol, timeframe, window_df)
                    
                    # Add unique patterns
                    for p in patterns:
                        key = self._pattern_key(p)
                        if key not in self.seen_patterns:
                            self.seen_patterns.add(key)
                            self.detected_patterns.append(p)
                            
                            print(f"  ✅ [{self.detection_triggers:3d}] {ext_type_str} @ bar {i} "
                                  f"-> Found {p.pattern_type.value} pattern "
                                  f"(validity: {p.validity_score:.2f})")
                    
                    # Progress update (reduced frequency)
                    if self.detection_triggers % 20 == 0:
                        print(f"  ... Triggers: {self.detection_triggers} | "
                              f"Patterns: {len(self.detected_patterns)} | "
                              f"ATR: {self.dc.current_atr:.2f}")
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        
        print(f"\n{'='*70}")
        print(f"  DETECTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Total bars processed: {n_bars}")
        print(f"  ATR extremes detected: {self.extremes_detected}")
        print(f"  Pattern triggers: {self.detection_triggers}")
        print(f"  Unique patterns found: {len(self.detected_patterns)}")
        print(f"{'='*70}")
        
        # Pattern breakdown
        if self.detected_patterns:
            bullish = sum(1 for p in self.detected_patterns if p.pattern_type == PatternType.BULLISH)
            bearish = len(self.detected_patterns) - bullish
            print(f"  Bullish: {bullish} | Bearish: {bearish}")
            
            avg_validity = np.mean([p.validity_score for p in self.detected_patterns])
            print(f"  Average validity: {avg_validity:.3f}")
        
        return self.detected_patterns


# =============================================================================
# DATA FETCHING (Preserved from v1.1.0)
# =============================================================================

def fetch_historical_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance."""
    
    if not HAS_CCXT:
        raise ImportError("ccxt is required for data fetching")
    
    print(f"📡 Fetching {symbol} {timeframe}...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int((end_date or datetime.now()).timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= current_ts:
                break
            current_ts = last_ts + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    if end_date:
        end_ts_pd = pd.Timestamp(end_date).tz_localize('UTC')
        df = df[df['time'] <= end_ts_pd]
    
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    print(f"   ✅ {len(df)} candles: {df['time'].min()} to {df['time'].max()}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run ATR-driven QML detection on BTC/USDT."""
    
    print("\n" + "="*70)
    print("  ATR DIRECTIONAL CHANGE QML PATTERN DETECTOR")
    print("  Version 2.0.0 - Price-Action Driven")
    print("="*70)
    
    # Configuration
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 25)
    
    # Fetch data
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        print("❌ No data fetched")
        return
    
    # Initialize detector with params
    params_path = Path(__file__).parent.parent / "qml_strategy_vrd/detection_logic/v1.1.0_rolling_window/params.json"
    
    detector = ATRPatternDetector(
        window_size=200,
        atr_lookback=14,
        params_path=str(params_path) if params_path.exists() else None
    )
    
    # Run detection
    patterns = detector.detect_all(df, symbol, timeframe)
    
    # Export results
    if patterns:
        results = []
        for p in patterns:
            results.append({
                'time': p.detection_time,
                'pattern_type': f"{p.pattern_type.value}_qml",
                'validity_score': p.validity_score,
                'head_price': p.head_price,
                'left_shoulder_price': p.left_shoulder_price,
                'entry_price': p.trading_levels.entry if p.trading_levels else None,
                'stop_loss': p.trading_levels.stop_loss if p.trading_levels else None,
                'take_profit': p.trading_levels.take_profit_1 if p.trading_levels else None,
            })
        
        results_df = pd.DataFrame(results)
        output_path = "btc_atr_patterns.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n📊 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
