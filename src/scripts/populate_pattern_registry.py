"""
Pattern Registry Population Script
==================================
Scan historical data for QML patterns and populate the ML registry
with full 170+ VRD features.

Usage:
    python src/scripts/populate_pattern_registry.py --symbol BTC/USDT --timeframe 4h

This script:
1. Loads historical OHLCV data
2. Runs ATR-based pattern detection
3. Extracts ALL 170+ features for each pattern
4. Registers patterns in ml_pattern_registry
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.ml.pattern_registry import PatternRegistry
from src.ml.feature_extractor import PatternFeatureExtractor


# =============================================================================
# PATTERN DETECTION (Simplified ATR-based)
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def find_swing_high(df: pd.DataFrame, idx: int, lookback: int = 10) -> Optional[Tuple[int, float]]:
    """Find recent swing high before index."""
    if idx < lookback:
        return None
    
    high_col = 'high' if 'high' in df.columns else 'High'
    subset = df.iloc[max(0, idx - lookback):idx]
    max_idx = subset[high_col].idxmax()
    return (max_idx, df.loc[max_idx, high_col])


def find_swing_low(df: pd.DataFrame, idx: int, lookback: int = 10) -> Optional[Tuple[int, float]]:
    """Find recent swing low before index."""
    if idx < lookback:
        return None
    
    low_col = 'low' if 'low' in df.columns else 'Low'
    subset = df.iloc[max(0, idx - lookback):idx]
    min_idx = subset[low_col].idxmin()
    return (min_idx, df.loc[min_idx, low_col])


def detect_bullish_qml(df: pd.DataFrame, idx: int, atr: float) -> Optional[Dict[str, Any]]:
    """
    Detect bullish QML pattern at given index.
    
    Bullish QML: Higher High → Lower Low (head) → Higher Low
    """
    if idx < 30:
        return None
    
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    close_col = 'close' if 'close' in df.columns else 'Close'
    
    current_low = df.iloc[idx][low_col]
    
    # Find head (lowest point in recent history)
    head_result = find_swing_low(df, idx, lookback=20)
    if head_result is None:
        return None
    head_idx, head_price = head_result
    
    # Head should be at least 1 ATR below current low
    if current_low - head_price < 0.5 * atr:
        return None
    
    # Find left shoulder (high before head)
    left_result = find_swing_high(df, head_idx, lookback=15)
    if left_result is None:
        return None
    left_idx, left_price = left_result
    
    # Current is right shoulder (higher low than head)
    if current_low <= head_price:
        return None
    
    # Entry, SL, TP
    entry = current_low
    sl = head_price - 0.5 * atr
    tp = entry + 2 * (entry - sl)
    
    return {
        'pattern_type': 'bullish_qml',
        'left_shoulder_idx': left_idx,
        'left_shoulder_price': left_price,
        'head_idx': head_idx,
        'head_price': head_price,
        'right_shoulder_idx': idx,
        'right_shoulder_price': current_low,
        'entry_price': entry,
        'stop_loss': sl,
        'take_profit': tp,
        'atr': atr,
    }


def detect_bearish_qml(df: pd.DataFrame, idx: int, atr: float) -> Optional[Dict[str, Any]]:
    """
    Detect bearish QML pattern at given index.
    
    Bearish QML: Lower Low → Higher High (head) → Lower High
    """
    if idx < 30:
        return None
    
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    
    current_high = df.iloc[idx][high_col]
    
    # Find head (highest point in recent history)
    head_result = find_swing_high(df, idx, lookback=20)
    if head_result is None:
        return None
    head_idx, head_price = head_result
    
    # Head should be at least 1 ATR above current high
    if head_price - current_high < 0.5 * atr:
        return None
    
    # Find left shoulder (low before head)
    left_result = find_swing_low(df, head_idx, lookback=15)
    if left_result is None:
        return None
    left_idx, left_price = left_result
    
    # Current is right shoulder (lower high than head)
    if current_high >= head_price:
        return None
    
    # Entry, SL, TP
    entry = current_high
    sl = head_price + 0.5 * atr
    tp = entry - 2 * (sl - entry)
    
    return {
        'pattern_type': 'bearish_qml',
        'left_shoulder_idx': left_idx,
        'left_shoulder_price': left_price,
        'head_idx': head_idx,
        'head_price': head_price,
        'right_shoulder_idx': idx,
        'right_shoulder_price': current_high,
        'entry_price': entry,
        'stop_loss': sl,
        'take_profit': tp,
        'atr': atr,
    }


# =============================================================================
# MAIN POPULATION FUNCTION
# =============================================================================

def populate_from_historical(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sample_rate: int = 10,  # Check every N bars for patterns
    max_patterns: int = 100,
) -> int:
    """
    Scan historical data and populate ML registry with patterns.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '4h')
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        sample_rate: Check every N bars for patterns
        max_patterns: Maximum patterns to register
        
    Returns:
        Number of patterns registered
    """
    
    # Load OHLCV data
    symbol_clean = symbol.replace("/", "")
    parquet_path = Path(f"data/processed/{symbol_clean}/{timeframe}_master.parquet")
    
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return 0
    
    logger.info(f"Loading OHLCV data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Ensure time column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['time'] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df['time'] <= end_dt]
    
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} candles from {df['time'].min()} to {df['time'].max()}")
    
    # Calculate ATR
    df['atr'] = calculate_atr(df)
    
    # Initialize ML infrastructure
    registry = PatternRegistry()
    extractor = PatternFeatureExtractor()
    
    patterns_registered = 0
    
    # Scan for patterns
    logger.info(f"Scanning for patterns (every {sample_rate} bars)...")
    
    for i in range(50, len(df) - 10, sample_rate):
        if patterns_registered >= max_patterns:
            break
        
        atr = df.iloc[i]['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        # Try bullish pattern
        pattern = detect_bullish_qml(df, i, atr)
        if pattern is None:
            # Try bearish pattern
            pattern = detect_bearish_qml(df, i, atr)
        
        if pattern is None:
            continue
        
        # Build pattern data
        detection_time = df.iloc[i]['time']
        
        pattern_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern_type': pattern['pattern_type'],
            'detection_time': detection_time,
            'detection_idx': i,
            'left_shoulder_price': pattern['left_shoulder_price'],
            'left_shoulder_idx': pattern['left_shoulder_idx'],
            'head_price': pattern['head_price'],
            'head_idx': pattern['head_idx'],
            'right_shoulder_price': pattern['right_shoulder_price'],
            'right_shoulder_idx': pattern['right_shoulder_idx'],
            'entry_price': pattern['entry_price'],
            'stop_loss': pattern['stop_loss'],
            'take_profit': pattern['take_profit'],
            'atr': pattern['atr'],
            'validity_score': 0.75,
        }
        
        # Extract FULL 170+ features
        try:
            features = extractor.extract_pattern_features(pattern_data, df, i)
            feature_count = len(features)
            
            if feature_count < 50:
                logger.warning(f"Low feature count ({feature_count}) at bar {i}, skipping")
                continue
                
        except Exception as e:
            logger.warning(f"Feature extraction failed at bar {i}: {e}")
            continue
        
        # Register pattern
        try:
            pattern_id = registry.register_pattern(pattern_data, features)
            patterns_registered += 1
            
            if patterns_registered % 10 == 0:
                logger.info(f"Registered {patterns_registered} patterns ({feature_count} features each)")
                
        except Exception as e:
            logger.error(f"Failed to register pattern: {e}")
    
    logger.info(f"✅ Registered {patterns_registered} patterns with full feature extraction")
    
    return patterns_registered


def verify_feature_counts():
    """Verify feature counts in existing patterns."""
    import sqlite3
    import json
    
    conn = sqlite3.connect('results/experiments.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT pattern_id, features_json FROM ml_pattern_registry')
    rows = cursor.fetchall()
    conn.close()
    
    logger.info("=== FEATURE COUNT VERIFICATION ===")
    for row in rows:
        pattern_id, features_json = row
        try:
            features = json.loads(features_json) if features_json else {}
            has_geometry = any('p1_' in k for k in features.keys())
            logger.info(f"{pattern_id[:12]}... | Features: {len(features)} | Geometry: {has_geometry}")
        except:
            logger.info(f"{pattern_id[:12]}... | Error parsing JSON")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Populate ML Pattern Registry')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='4h', help='Candle timeframe')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample-rate', type=int, default=6, help='Check every N bars')
    parser.add_argument('--max-patterns', type=int, default=100, help='Max patterns to register')
    parser.add_argument('--verify', action='store_true', help='Only verify existing patterns')
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("🧠 ML PATTERN REGISTRY POPULATION")
    print("=" * 60)
    print()
    
    if args.verify:
        verify_feature_counts()
        return
    
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Sample Rate: Every {args.sample_rate} bars")
    print(f"Max Patterns: {args.max_patterns}")
    print()
    
    count = populate_from_historical(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        sample_rate=args.sample_rate,
        max_patterns=args.max_patterns,
    )
    
    print()
    print("=" * 60)
    print(f"✅ COMPLETE: Registered {count} patterns")
    print("=" * 60)
    print()
    
    # Verify
    verify_feature_counts()


if __name__ == "__main__":
    main()
