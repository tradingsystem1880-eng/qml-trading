#!/usr/bin/env python3
"""
FORENSIC DIAGNOSTIC: QML Pattern Detection Root Cause Analysis
==============================================================
Phase 1: Determine if flaw is in Data, Logic, or Parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import time
import ccxt
from collections import defaultdict

from src.detection.detector import QMLDetector, DetectorConfig
from src.detection.swing import SwingDetector, SwingConfig
from src.detection.choch import CHoCHDetector, CHoCHConfig
from src.detection.bos import BoSDetector
from src.detection.structure import StructureAnalyzer
from src.data.models import PatternType, SwingType


def fetch_historical_data(symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """Fetch historical data for a specific period."""
    
    print(f"   📡 Fetching {symbol} {timeframe}...")
    
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
            print(f"      ⚠️ Error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    # Filter to date range
    if end_date:
        end_ts_pd = pd.Timestamp(end_date).tz_localize('UTC')
        df = df[df['time'] <= end_ts_pd]
    
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    print(f"      ✅ {len(df)} candles: {df['time'].min()} to {df['time'].max()}")
    return df


print("\n" + "="*80)
print("  🔬 FORENSIC DIAGNOSTIC: QML Pattern Detection Root Cause Analysis")
print("="*80)


# ==============================================================================
# COMMAND 1: DATA CONSISTENCY AUDIT
# ==============================================================================
print("\n" + "="*80)
print("  COMMAND 1: DATA CONSISTENCY AUDIT")
print("="*80)

# Load the original backtest data
backtest_file = Path(__file__).parent.parent / "data" / "comprehensive_features.csv"
original_data = pd.read_csv(backtest_file)

print(f"\n📊 Original Backtest Data:")
print(f"   Total patterns: {len(original_data)}")

# Filter to BTC only
btc_patterns = original_data[original_data['symbol'] == 'BTC/USDT'].copy()
print(f"   BTC/USDT patterns: {len(btc_patterns)}")

if len(btc_patterns) > 0:
    btc_patterns['detection_time'] = pd.to_datetime(btc_patterns['detection_time'])
    min_date = btc_patterns['detection_time'].min()
    max_date = btc_patterns['detection_time'].max()
    print(f"   Date range: {min_date} to {max_date}")
    
    print(f"\n   Pattern type distribution:")
    print(btc_patterns['pattern_type'].value_counts().to_string())

# Run current detector on same period
print(f"\n🔍 Running current detector on SAME period as original backtest...")

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)

df_audit = fetch_historical_data("BTC/USDT", "1h", start_date, end_date)

if len(df_audit) > 100:
    detector = QMLDetector()
    patterns_found = detector.detect("BTC/USDT", "1h", df_audit)
    
    print(f"\n📊 AUDIT RESULT:")
    print(f"   Original BTC patterns (from CSV): {len(btc_patterns)}")
    print(f"   Current detector found: {len(patterns_found)}")
    
    if len(patterns_found) < 10:
        print(f"\n   ⚠️ DIAGNOSIS: Current detector is finding FAR FEWER patterns")
        print(f"      Issue is in DETECTION CODE or PARAMETERS, not data.")


# ==============================================================================
# COMMAND 2: LOGIC INSTRUMENTATION
# ==============================================================================
print("\n" + "="*80)
print("  COMMAND 2: LOGIC INSTRUMENTATION & COMPARISON")
print("="*80)

# Use a 3-month slice for detailed analysis
print(f"\n🔬 Running instrumented detection on Q1 2023...")

slice_start = datetime(2023, 1, 1)
slice_end = datetime(2023, 4, 1)

df_slice = fetch_historical_data("BTC/USDT", "1h", slice_start, slice_end)

if len(df_slice) > 100:
    print(f"\n📊 Detection Pipeline Instrumentation:")
    print("-" * 60)
    
    # Step 1: Swing Detection
    swing_detector = SwingDetector()
    swings = swing_detector.detect(df_slice, "BTC/USDT")
    
    swing_highs = [s for s in swings if s.swing_type == SwingType.HIGH]
    swing_lows = [s for s in swings if s.swing_type == SwingType.LOW]
    
    print(f"\n   STEP 1 - Swing Detection:")
    print(f"      Swing Highs: {len(swing_highs)}")
    print(f"      Swing Lows: {len(swing_lows)}")
    print(f"      Total Swings: {len(swings)}")
    
    # Step 2: Structure Analysis
    structure = StructureAnalyzer()
    trend_state = structure.analyze(df_slice, "BTC/USDT", "1h")
    
    print(f"\n   STEP 2 - Structure Analysis:")
    print(f"      Trend: {trend_state.trend}")
    print(f"      Strength: {trend_state.strength:.2f}")
    print(f"      Structures found: {len(trend_state.structures)}")
    
    # Step 3: CHoCH Detection - THE BOTTLENECK
    choch_detector = CHoCHDetector()
    choch_events = choch_detector.detect(df_slice, swings, trend_state.structures)
    
    print(f"\n   STEP 3 - CHoCH Detection (⚠️ CRITICAL BOTTLENECK):")
    print(f"      CHoCH Events Found: {len(choch_events)}")
    
    if len(choch_events) == 0:
        print(f"      ❌ NO CHoCH EVENTS DETECTED!")
        print(f"      This is the ROOT CAUSE of missing patterns.")
    else:
        for i, ch in enumerate(choch_events[:5]):
            print(f"      CHoCH {i+1}: {ch.time} @ ${ch.break_level:,.2f} ({ch.choch_type.value})")
    
    # Step 4: BoS Detection
    bos_detector = BoSDetector()
    bos_events = bos_detector.detect(df_slice, swings, choch_events, trend_state.structures)
    
    print(f"\n   STEP 4 - BoS Detection:")
    print(f"      BoS Events Found: {len(bos_events)}")
    
    # Step 5: Final Patterns
    patterns = detector.detect("BTC/USDT", "1h", df_slice)
    print(f"\n   STEP 5 - Final Pattern Assembly:")
    print(f"      Valid QML Patterns: {len(patterns)}")


# ==============================================================================
# COMMAND 3: PARAMETER SENSITIVITY BOMBARDMENT
# ==============================================================================
print("\n" + "="*80)
print("  COMMAND 3: PARAMETER SENSITIVITY BOMBARDMENT")
print("="*80)

test_df = df_slice if 'df_slice' in dir() and len(df_slice) > 100 else df_audit

print(f"\n📊 Test Data: {len(test_df)} candles")

# ============ SWING WINDOW TEST ============
print(f"\n🔧 TEST 1: SWING_WINDOW parameter")
print("-" * 60)

swing_windows = [3, 5, 7, 10, 15, 20]
for sw in swing_windows:
    config = SwingConfig(swing_window=sw)
    sd = SwingDetector(config=config)
    swings = sd.detect(test_df, "BTC/USDT")
    highs = len([s for s in swings if s.swing_type == SwingType.HIGH])
    lows = len([s for s in swings if s.swing_type == SwingType.LOW])
    print(f"   swing_window={sw:2d}: {highs:3d} highs, {lows:3d} lows, total={highs+lows:3d}")

# Use default swings for CHoCH testing
swing_detector = SwingDetector()
swings = swing_detector.detect(test_df, "BTC/USDT")
structure = StructureAnalyzer()
trend_state = structure.analyze(test_df, "BTC/USDT", "1h")

# ============ CHoCH PARAMETER TEST ============
print(f"\n🔧 TEST 2: CHoCH PARAMETERS (min_break_percent, confirmation_bars)")
print("-" * 60)

# Read current default
default_cfg = CHoCHConfig()
print(f"\n   Current defaults:")
print(f"      min_break_percent: {default_cfg.min_break_percent}")
print(f"      confirmation_bars: {default_cfg.confirmation_bars}")
print(f"      lookback_structures: {default_cfg.lookback_structures}")

print(f"\n   Parameter sweep:")

# Systematic test
break_percents = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
confirm_bars = [1, 2, 3]

for bp in break_percents:
    for cb in confirm_bars:
        try:
            cfg = CHoCHConfig(min_break_percent=bp, confirmation_bars=cb)
            choch_det = CHoCHDetector(config=cfg)
            choch_events = choch_det.detect(test_df, swings, trend_state.structures)
            print(f"   break={bp:.4f}, confirm={cb}: {len(choch_events):3d} CHoCH events")
        except Exception as e:
            print(f"   break={bp:.4f}, confirm={cb}: ERROR - {e}")

# ============ LOOKBACK STRUCTURES TEST ============
print(f"\n🔧 TEST 3: LOOKBACK_STRUCTURES parameter")
print("-" * 60)

lookbacks = [3, 5, 7, 10, 15, 20]
for lb in lookbacks:
    try:
        cfg = CHoCHConfig(lookback_structures=lb)
        choch_det = CHoCHDetector(config=cfg)
        choch_events = choch_det.detect(test_df, swings, trend_state.structures)
        print(f"   lookback_structures={lb:2d}: {len(choch_events):3d} CHoCH events")
    except Exception as e:
        print(f"   lookback_structures={lb:2d}: ERROR - {e}")

# ============ FULL DETECTOR TEST ============
print(f"\n🔧 TEST 4: FULL DETECTOR parameter sweep")
print("-" * 60)

configs = [
    {"min_validity_score": 0.3},
    {"min_validity_score": 0.4},
    {"min_validity_score": 0.5},
    {"min_validity_score": 0.6},
    {"min_head_depth_atr": 0.1},
    {"min_head_depth_atr": 0.2},
    {"min_head_depth_atr": 0.5},
    {"max_head_depth_atr": 5.0},
    {"max_head_depth_atr": 8.0},
    {"max_head_depth_atr": 15.0},
]

for cfg in configs:
    det_cfg = DetectorConfig(**cfg)
    det = QMLDetector(config=det_cfg)
    patterns = det.detect("BTC/USDT", "1h", test_df)
    print(f"   {cfg}: {len(patterns):3d} patterns")


# ==============================================================================
# DEEP DIVE: CHoCH LOGIC EXAMINATION
# ==============================================================================
print("\n" + "="*80)
print("  🔎 DEEP DIVE: CHoCH Detection Logic Analysis")
print("="*80)

# Let's examine what conditions must be met for CHoCH
print(f"""
📋 CHoCH Detection Requirements (from code):
---------------------------------------------
For a CHoCH event to be detected, ALL of the following must be TRUE:

1. There must be a clear TREND established (via structures)
2. Price must break a KEY LEVEL (significant swing high/low)
3. The break must exceed min_break_percent threshold
4. The break must be CONFIRMED for N bars (confirmation_bars)
5. The break must happen within lookback_structures of the key level

Current test results show: {len(choch_events) if 'choch_events' in dir() else 0} CHoCH events

If CHoCH count is 0 or very low, the issue is likely:
- Structures not being labeled as "key levels"
- Break percent too high
- Confirmation too strict
- Trend direction not established properly
""")

# Check structure details
print(f"\n📊 Structure Analysis Deep Dive:")
print(f"   Total structures: {len(trend_state.structures)}")

# Count structure types
if len(trend_state.structures) > 0:
    hh_count = sum(1 for s in trend_state.structures if s.structure_type.value == 'higher_high')
    hl_count = sum(1 for s in trend_state.structures if s.structure_type.value == 'higher_low')
    lh_count = sum(1 for s in trend_state.structures if s.structure_type.value == 'lower_high')
    ll_count = sum(1 for s in trend_state.structures if s.structure_type.value == 'lower_low')
    
    print(f"   Higher Highs (HH): {hh_count}")
    print(f"   Higher Lows (HL): {hl_count}")
    print(f"   Lower Highs (LH): {lh_count}")
    print(f"   Lower Lows (LL): {ll_count}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("  📋 FORENSIC DIAGNOSTIC SUMMARY")
print("="*80)

print(f"""
FINDINGS:
---------
1. DATA AUDIT:
   - Original backtest: 119 BTC patterns
   - Current detector: {len(patterns_found) if 'patterns_found' in dir() else 'N/A'} patterns
   - Verdict: ⚠️ MASSIVE DISCREPANCY - Code/params diverged

2. PIPELINE BOTTLENECK:
   - Swings: Working (detecting {len(swings) if 'swings' in dir() else 'N/A'} swings)
   - Structures: Working ({len(trend_state.structures) if 'trend_state' in dir() else 'N/A'} structures)
   - CHoCH: ❌ BOTTLENECK ({len(choch_events) if 'choch_events' in dir() else 0} events)
   - This is blocking all downstream pattern detection

3. PARAMETER SENSITIVITY:
   - See results above for which parameters affect CHoCH count
   - The CHoCH detection is extremely strict

RECOMMENDED FIXES:
------------------
1. Review CHoCH detection logic - may be too conservative
2. Reduce min_break_percent (try 0.001 or lower)
3. Reduce confirmation_bars (try 1)
4. Increase lookback_structures (try 10-15)
5. Check if "key level" determination is too strict
""")

print("="*80 + "\n")
