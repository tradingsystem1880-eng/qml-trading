#!/usr/bin/env python3
"""
Export BTC/USDT QML Patterns to CSV for TradingView
====================================================
Comprehensive export with all price levels for Pine Script visualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import ccxt

from src.detection.detector import QMLDetector
from src.data.models import QMLPattern, PatternType


def fetch_historical_data(symbol: str, timeframe: str, 
                          start_date: datetime = None, 
                          limit: int = 10000) -> pd.DataFrame:
    """Fetch extensive historical data directly from exchange."""
    
    print(f"📡 Fetching {symbol} {timeframe} historical data...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Calculate start time (default: ~2 years for 1h)
    if start_date is None:
        if timeframe == '1h':
            start_date = datetime.now() - timedelta(days=400)  # ~400 days
        elif timeframe == '4h':
            start_date = datetime.now() - timedelta(days=800)  # ~800 days
        else:
            start_date = datetime.now() - timedelta(days=365)
    
    start_ts = int(start_date.timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    batch_size = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=batch_size)
            
            if not ohlcv:
                break
            
            all_candles.extend(ohlcv)
            
            # Update timestamp for next batch
            last_ts = ohlcv[-1][0]
            if last_ts <= current_ts:
                break
            current_ts = last_ts + 1
            
            # Progress
            date_str = datetime.fromtimestamp(last_ts/1000).strftime('%Y-%m-%d')
            print(f"   📈 {len(all_candles)} candles loaded (up to {date_str})")
            
            # Stop if we have enough or reached current time
            if len(all_candles) >= limit or last_ts >= int(datetime.now().timestamp() * 1000) - 3600000:
                break
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    print(f"   ✅ Total: {len(df)} unique candles")
    print(f"   📅 Range: {df['time'].min()} to {df['time'].max()}")
    
    return df


def export_btc_patterns():
    """Detect and export all BTC/USDT QML patterns with full price levels."""
    
    print("\n" + "="*70)
    print("  📊 BTC/USDT COMPREHENSIVE PATTERN EXPORT")
    print("  For TradingView Pine Script Visualization")
    print("="*70)
    
    detector = QMLDetector()
    
    all_patterns: List[Dict[str, Any]] = []
    
    # Process multiple timeframes
    timeframes_config = [
        ("1h", 400),   # ~400 days of 1h data
        ("4h", 800),   # ~800 days of 4h data
    ]
    
    for tf, days_back in timeframes_config:
        print(f"\n{'='*60}")
        print(f"🔍 Processing BTC/USDT {tf}")
        print(f"{'='*60}")
        
        # Fetch historical data
        start = datetime.now() - timedelta(days=days_back)
        df = fetch_historical_data("BTC/USDT", tf, start_date=start, limit=15000)
        
        if df is None or len(df) < 200:
            print(f"   ❌ Insufficient data")
            continue
        
        # Detect patterns
        print(f"\n   🔬 Running QML detection...")
        patterns = detector.detect("BTC/USDT", tf, df)
        print(f"   ✅ Found {len(patterns)} patterns")
        
        # Convert each pattern to export format
        for pattern in patterns:
            try:
                # Get timestamps
                det_time = pattern.detection_time
                if hasattr(det_time, 'timestamp'):
                    if det_time.tzinfo is not None:
                        det_time = det_time.replace(tzinfo=None)
                    unix_ts = int(det_time.timestamp() * 1000)
                    time_str = det_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    unix_ts = 0
                    time_str = str(det_time)
                
                head_time = pattern.head_time
                if hasattr(head_time, 'timestamp'):
                    if head_time.tzinfo is not None:
                        head_time = head_time.replace(tzinfo=None)
                    head_unix = int(head_time.timestamp() * 1000)
                    head_time_str = head_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    head_unix = 0
                    head_time_str = str(head_time)
                
                choch_time = pattern.left_shoulder_time
                if hasattr(choch_time, 'timestamp'):
                    if choch_time.tzinfo is not None:
                        choch_time = choch_time.replace(tzinfo=None)
                    choch_unix = int(choch_time.timestamp() * 1000)
                    choch_time_str = choch_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    choch_unix = 0
                    choch_time_str = str(choch_time)
                
                # Trading levels
                entry = pattern.trading_levels.entry if pattern.trading_levels else 0
                stop = pattern.trading_levels.stop_loss if pattern.trading_levels else 0
                tp1 = pattern.trading_levels.take_profit_1 if pattern.trading_levels else 0
                tp2 = pattern.trading_levels.take_profit_2 if pattern.trading_levels else 0
                tp3 = pattern.trading_levels.take_profit_3 if pattern.trading_levels else 0
                
                # Pattern type
                is_bullish = pattern.pattern_type == PatternType.BULLISH
                pattern_type = "bullish_qml" if is_bullish else "bearish_qml"
                direction = 1 if is_bullish else -1
                
                pattern_data = {
                    # Primary timestamp (Unix ms for TradingView)
                    'timestamp': unix_ts,
                    'time': time_str,
                    
                    # Pattern component timestamps
                    'head_timestamp': head_unix,
                    'head_time': head_time_str,
                    'choch_timestamp': choch_unix,
                    'choch_time': choch_time_str,
                    
                    # Basic info
                    'symbol': 'BTCUSDT',
                    'timeframe': tf,
                    'pattern_type': pattern_type,
                    'direction': direction,
                    'validity_score': round(pattern.validity_score, 4),
                    
                    # KEY PRICE LEVELS (what user needs)
                    'left_shoulder_price': round(pattern.left_shoulder_price, 2),
                    'head_price': round(pattern.head_price, 2),
                    'choch_level': round(pattern.left_shoulder_price, 2),
                    'neckline': round(pattern.neckline_start, 2) if pattern.neckline_start else round(pattern.left_shoulder_price, 2),
                    
                    # Trading levels
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': round(tp1, 2),
                    'take_profit_2': round(tp2, 2),
                    'take_profit_3': round(tp3, 2),
                    
                    # Risk metrics
                    'risk': round(abs(entry - stop), 2),
                    'reward': round(abs(tp1 - entry), 2),
                    'risk_reward': round(abs(tp1 - entry) / abs(entry - stop), 2) if stop != entry else 0,
                }
                
                all_patterns.append(pattern_data)
                
            except Exception as e:
                print(f"   ⚠️ Error processing pattern: {e}")
                continue
    
    # Sort by timestamp
    all_patterns.sort(key=lambda x: x['timestamp'])
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 EXPORT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n   Total Patterns: {len(all_patterns)}")
    
    if all_patterns:
        bullish = len([p for p in all_patterns if p['direction'] == 1])
        bearish = len([p for p in all_patterns if p['direction'] == -1])
        print(f"   🟢 Bullish: {bullish}")
        print(f"   🔴 Bearish: {bearish}")
        
        # Date range
        first_date = all_patterns[0]['time']
        last_date = all_patterns[-1]['time']
        print(f"\n   📅 Date Range: {first_date} to {last_date}")
        
        # By timeframe
        tf_counts = {}
        for p in all_patterns:
            tf = p['timeframe']
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
        
        print(f"\n   By Timeframe:")
        for tf, count in sorted(tf_counts.items()):
            print(f"      {tf}: {count} patterns")
    
    # Export to CSV
    df_export = pd.DataFrame(all_patterns)
    output_path = Path(__file__).parent.parent / "btc_backtest_labels.csv"
    df_export.to_csv(output_path, index=False)
    
    print(f"\n✅ Exported to: {output_path}")
    print(f"   📝 {len(df_export)} rows × {len(df_export.columns)} columns")
    
    # Show sample
    if len(df_export) > 0:
        print(f"\n📋 First 5 patterns:")
        print("-" * 110)
        cols = ['time', 'timeframe', 'pattern_type', 'head_price', 'entry_price', 'stop_loss', 'take_profit']
        print(df_export[cols].head(5).to_string())
        
        print(f"\n📋 Last 5 patterns:")
        print("-" * 110)
        print(df_export[cols].tail(5).to_string())
    
    # Column reference
    print(f"\n📊 Columns (for TradingView):")
    print("-" * 50)
    for i, col in enumerate(df_export.columns, 1):
        print(f"   {i:2}. {col}")
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETE")
    print("   Ready for TradingView Pine Script import")
    print("="*70 + "\n")
    
    return df_export


if __name__ == "__main__":
    export_btc_patterns()
