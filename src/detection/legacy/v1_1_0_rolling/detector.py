#!/usr/bin/env python3
"""
ROLLING WINDOW QML PATTERN DETECTION
====================================
Detects patterns throughout historical data by running detection
in rolling windows, mimicking how patterns would be detected in real-time.

This is the FIX for the detection issue where only 1-3 patterns
were found instead of the expected 100+ patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import time
import ccxt
from collections import defaultdict

from src.detection.detector import QMLDetector, DetectorConfig
from src.detection.swing import SwingDetector
from src.detection.choch import CHoCHDetector
from src.detection.bos import BoSDetector
from src.detection.structure import StructureAnalyzer
from src.data.models import QMLPattern, PatternType, SwingType


def fetch_historical_data(symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """Fetch historical data for a specific period."""
    
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


class RollingPatternDetector:
    """
    Detects QML patterns using a rolling window approach.
    
    This mimics real-time detection by running the detector
    at each point in time and collecting all detected patterns.
    """
    
    def __init__(self, window_size: int = 200, step_size: int = 24):
        """
        Initialize rolling detector.
        
        Args:
            window_size: Number of bars to look back for each detection
            step_size: Number of bars to advance between detections
        """
        self.window_size = window_size
        self.step_size = step_size
        self.detector = QMLDetector()
        self.detected_patterns: List[QMLPattern] = []
        self.seen_patterns: Set[str] = set()  # Track unique patterns
    
    def _pattern_key(self, pattern: QMLPattern) -> str:
        """Create unique key for pattern deduplication."""
        return f"{pattern.head_time}_{pattern.left_shoulder_time}_{pattern.pattern_type.value}"
    
    def detect_all(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[QMLPattern]:
        """
        Detect all patterns in historical data using rolling windows.
        
        Args:
            df: Full historical OHLCV DataFrame
            symbol: Trading pair
            timeframe: Candle timeframe
            
        Returns:
            List of all unique patterns detected
        """
        n_bars = len(df)
        
        if n_bars < self.window_size:
            print(f"⚠️ Insufficient data: {n_bars} bars < {self.window_size} window")
            return []
        
        print(f"\n🔄 Running rolling detection on {symbol} {timeframe}")
        print(f"   Data: {n_bars} bars")
        print(f"   Window: {self.window_size} bars")
        print(f"   Step: {self.step_size} bars")
        
        self.detected_patterns = []
        self.seen_patterns = set()
        
        # Calculate number of windows
        n_windows = (n_bars - self.window_size) // self.step_size + 1
        print(f"   Windows: {n_windows}")
        
        patterns_found = 0
        
        for i in range(0, n_bars - self.window_size + 1, self.step_size):
            # Extract window
            window_df = df.iloc[i:i + self.window_size].copy().reset_index(drop=True)
            
            # Run detection
            patterns = self.detector.detect(symbol, timeframe, window_df)
            
            # Add unique patterns
            for p in patterns:
                key = self._pattern_key(p)
                if key not in self.seen_patterns:
                    self.seen_patterns.add(key)
                    self.detected_patterns.append(p)
                    patterns_found += 1
            
            # Progress update every 50 windows
            if (i // self.step_size) % 50 == 0:
                window_time = window_df['time'].iloc[-1]
                print(f"   Progress: window {i // self.step_size + 1}/{n_windows} "
                      f"({window_time.strftime('%Y-%m-%d')}) - {patterns_found} patterns")
        
        # Sort by detection time
        self.detected_patterns.sort(key=lambda p: p.detection_time)
        
        print(f"\n✅ Total unique patterns detected: {len(self.detected_patterns)}")
        
        return self.detected_patterns


def main():
    print("\n" + "="*80)
    print("  🔄 ROLLING WINDOW QML PATTERN DETECTION")
    print("  Detecting ALL historical patterns")
    print("="*80)
    
    # Fetch data for the same period as original backtest (2023)
    print("\n📊 Loading data for 2023 (same period as original backtest)...")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    df = fetch_historical_data("BTC/USDT", "1h", start_date, end_date)
    
    if len(df) < 200:
        print("❌ Insufficient data")
        return
    
    # Initialize rolling detector
    # window_size=200 gives enough context for pattern detection
    # step_size=12 (12 hours) gives reasonable granularity
    rolling_detector = RollingPatternDetector(window_size=200, step_size=12)
    
    # Detect all patterns
    patterns = rolling_detector.detect_all(df, "BTC/USDT", "1h")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 DETECTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n   Total Patterns: {len(patterns)}")
    
    if patterns:
        bullish = len([p for p in patterns if p.pattern_type == PatternType.BULLISH])
        bearish = len([p for p in patterns if p.pattern_type == PatternType.BEARISH])
        print(f"   🟢 Bullish: {bullish}")
        print(f"   🔴 Bearish: {bearish}")
        
        # Date range
        first = patterns[0].detection_time
        last = patterns[-1].detection_time
        print(f"   📅 Date Range: {first} to {last}")
        
        # Monthly distribution
        print(f"\n   Monthly distribution:")
        monthly = defaultdict(int)
        for p in patterns:
            month = p.detection_time.strftime('%Y-%m')
            monthly[month] += 1
        
        for month in sorted(monthly.keys()):
            print(f"      {month}: {monthly[month]} patterns")
    
    # Compare to original
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON TO ORIGINAL BACKTEST")
    print(f"{'='*80}")
    
    original_file = Path(__file__).parent.parent / "data" / "comprehensive_features.csv"
    if original_file.exists():
        original = pd.read_csv(original_file)
        btc_original = original[original['symbol'] == 'BTC/USDT']
        
        # Filter to 2023
        btc_original['detection_time'] = pd.to_datetime(btc_original['detection_time'])
        btc_2023 = btc_original[
            (btc_original['detection_time'] >= '2023-01-01') &
            (btc_original['detection_time'] < '2024-01-01')
        ]
        
        print(f"\n   Original backtest (2023): {len(btc_2023)} BTC patterns")
        print(f"   Rolling detector (2023): {len(patterns)} BTC patterns")
        
        ratio = len(patterns) / len(btc_2023) if len(btc_2023) > 0 else 0
        print(f"\n   Ratio: {ratio:.2f}x")
        
        if ratio > 0.5:
            print(f"   ✅ Rolling detection is finding patterns at similar rate!")
        else:
            print(f"   ⚠️ Still finding fewer patterns - may need parameter tuning")
    
    # Export to CSV
    if patterns:
        export_data = []
        for p in patterns:
            export_data.append({
                'timestamp': int(p.detection_time.timestamp() * 1000) if p.detection_time else 0,
                'time': p.detection_time.strftime('%Y-%m-%d %H:%M:%S') if p.detection_time else '',
                'symbol': 'BTCUSDT',
                'timeframe': p.timeframe,
                'pattern_type': f"{'bullish' if p.pattern_type == PatternType.BULLISH else 'bearish'}_qml",
                'direction': 1 if p.pattern_type == PatternType.BULLISH else -1,
                'validity_score': round(p.validity_score, 4),
                'head_price': round(p.head_price, 2),
                'left_shoulder_price': round(p.left_shoulder_price, 2),
                'choch_level': round(p.left_shoulder_price, 2),
                'entry_price': round(p.trading_levels.entry, 2) if p.trading_levels else 0,
                'stop_loss': round(p.trading_levels.stop_loss, 2) if p.trading_levels else 0,
                'take_profit': round(p.trading_levels.take_profit_1, 2) if p.trading_levels else 0,
            })
        
        export_df = pd.DataFrame(export_data)
        output_path = Path(__file__).parent.parent / "btc_backtest_labels.csv"
        export_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Exported {len(export_df)} patterns to: {output_path}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()


