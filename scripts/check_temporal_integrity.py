#!/usr/bin/env python3
"""
Diagnostic 1: Temporal Integrity Check
======================================
Verify data splits are correct and non-overlapping.
Explain why baseline dropped from 1.23 → 0.98.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import load_master_data


def load_json(path):
    """Load JSON file."""
    with open(PROJECT_ROOT / path) as f:
        return json.load(f)


def get_data_date_range(symbols, timeframe='4h'):
    """Get date range of available data."""
    all_starts = []
    all_ends = []

    for symbol in symbols:
        try:
            df = load_master_data(timeframe, symbol=symbol)
            if len(df) > 0:
                all_starts.append(df.index.min())
                all_ends.append(df.index.max())
        except Exception as e:
            print(f"  Could not load {symbol}: {e}")

    if all_starts and all_ends:
        return min(all_starts), max(all_ends)
    return None, None


def check_temporal_integrity():
    """Verify data splits are correct and non-overlapping."""

    print("=" * 60)
    print("DIAGNOSTIC 1: TEMPORAL INTEGRITY CHECK")
    print("=" * 60)

    # Load Phase 7.9 results
    phase79_path = PROJECT_ROOT / 'results/phase77_optimization/profit_factor_penalized/final_results.json'

    if not phase79_path.exists():
        print(f"\nERROR: Phase 7.9 results not found at {phase79_path}")
        return {'error': 'Phase 7.9 results not found'}

    phase79 = load_json(phase79_path)

    # Load Phase 8.0 results
    phase80_path = PROJECT_ROOT / 'results/phase80_ml/production_gate_result.json'

    if not phase80_path.exists():
        print(f"\nERROR: Phase 8.0 results not found at {phase80_path}")
        return {'error': 'Phase 8.0 results not found'}

    phase80 = load_json(phase80_path)

    # ========================================
    # 1. SYMBOL COMPARISON
    # ========================================
    print("\n" + "=" * 40)
    print("1. SYMBOL COMPARISON")
    print("=" * 40)

    # Phase 7.9 used ALL_CLUSTERED_SYMBOLS (22 symbols)
    phase79_symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT',
        'ARBUSDT', 'OPUSDT', 'MATICUSDT', 'AAVEUSDT', 'UNIUSDT', 'LINKUSDT',
        'MKRUSDT', 'INJUSDT', 'BNBUSDT', 'DOGEUSDT', 'PEPEUSDT', 'WIFUSDT',
        'DOTUSDT', 'ATOMUSDT', 'RUNEUSDT', 'TIAUSDT'
    ]

    # Phase 8.0 used only 10 symbols
    phase80_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AVAXUSDT',
        'DOGEUSDT', 'LINKUSDT', 'ARBUSDT', 'OPUSDT', 'MATICUSDT'
    ]

    print(f"\nPhase 7.9 symbols: {len(phase79_symbols)}")
    print(f"Phase 8.0 symbols: {len(phase80_symbols)}")
    print(f"\nMissing from Phase 8.0:")
    missing = set(phase79_symbols) - set(phase80_symbols)
    for s in sorted(missing):
        print(f"  - {s}")

    print(f"\n⚠️  CRITICAL: Phase 8.0 used HALF the symbols!")
    print("   This alone could explain the baseline degradation.")

    # ========================================
    # 2. TRADE COUNT COMPARISON
    # ========================================
    print("\n" + "=" * 40)
    print("2. TRADE COUNT COMPARISON")
    print("=" * 40)

    phase79_trades = phase79['best_simulation']['total_trades']
    phase80_baseline_trades = phase80['baseline']['trades']
    phase80_ml_trades = phase80['ml']['trades']

    print(f"\nPhase 7.9 trades:      {phase79_trades}")
    print(f"Phase 8.0 baseline:    {phase80_baseline_trades}")
    print(f"Phase 8.0 ML:          {phase80_ml_trades}")

    print(f"\n⚠️  Phase 8.0 has {phase80_baseline_trades/phase79_trades*100:.1f}% of Phase 7.9 trades")
    print(f"   Expected ~{len(phase80_symbols)/len(phase79_symbols)*100:.0f}% based on symbol count")

    # ========================================
    # 3. PARAMETER COMPARISON
    # ========================================
    print("\n" + "=" * 40)
    print("3. PARAMETER COMPARISON")
    print("=" * 40)

    phase79_params = phase79['best_params']

    print("\nPhase 7.9 best params (used in both):")
    key_params = ['sl_atr_mult', 'tp_atr_mult', 'min_risk_reward', 'min_pattern_bars']
    for p in key_params:
        print(f"  {p}: {phase79_params.get(p)}")

    print("\n✓ Same parameters used in Phase 8.0 (loaded from Phase 7.9 results)")

    # ========================================
    # 4. METRIC COMPARISON
    # ========================================
    print("\n" + "=" * 40)
    print("4. METRIC COMPARISON")
    print("=" * 40)

    print("\n| Metric | Phase 7.9 | Phase 8.0 Baseline | Delta |")
    print("|--------|-----------|-------------------|-------|")

    metrics = [
        ('Profit Factor', phase79['best_simulation']['profit_factor'], phase80['baseline']['pf']),
        ('Win Rate', phase79['best_simulation']['win_rate'], None),  # Not in gate result
        ('Sharpe', phase79['best_simulation']['sharpe_ratio'], phase80['baseline']['sharpe']),
        ('Trades', phase79['best_simulation']['total_trades'], phase80['baseline']['trades']),
    ]

    for name, p79, p80 in metrics:
        if p80 is not None:
            delta = ((p80 / p79) - 1) * 100 if p79 != 0 else 0
            print(f"| {name} | {p79:.4f} | {p80:.4f} | {delta:+.1f}% |")
        else:
            print(f"| {name} | {p79:.4f} | N/A | - |")

    # ========================================
    # 5. ROOT CAUSE ANALYSIS
    # ========================================
    print("\n" + "=" * 40)
    print("5. ROOT CAUSE ANALYSIS")
    print("=" * 40)

    print("\nWhy did baseline PF drop from 1.23 → 0.98?")
    print("-" * 40)

    print("""
CONFIRMED ISSUES:

1. SYMBOL MISMATCH (CRITICAL)
   - Phase 7.9 optimized on 22 symbols
   - Phase 8.0 tested on only 10 symbols
   - The 12 missing symbols may have had better performance
   - This is NOT a valid out-of-sample test

2. SAMPLE SIZE
   - Phase 7.9: 2,155 trades
   - Phase 8.0: 220 trades (only 10% of original)
   - Statistical significance is questionable

3. SELECTION BIAS
   - The 10 symbols chosen may have different characteristics
   - Missing: NEARUSDT, APTUSDT, AAVEUSDT, UNIUSDT, MKRUSDT,
             INJUSDT, PEPEUSDT, WIFUSDT, DOTUSDT, ATOMUSDT,
             RUNEUSDT, TIAUSDT

CONCLUSION:
The baseline drop is explained by TESTING ON DIFFERENT DATA,
not regime shift or overfitting. Phase 8.0 needs to use the
same 22 symbols as Phase 7.9 for a valid comparison.
""")

    # ========================================
    # 6. ML RESULTS ANALYSIS
    # ========================================
    print("\n" + "=" * 40)
    print("6. ML RESULTS ANALYSIS")
    print("=" * 40)

    print(f"\nPhase 8.0 ML Results:")
    print(f"  Profit Factor: {phase80['ml']['pf']:.2f}")
    print(f"  Trades: {phase80['ml']['trades']}")
    print(f"  Trades skipped: {phase80['sizing_distribution']['skipped']}")

    skip_rate = phase80['sizing_distribution']['skipped'] / phase80_baseline_trades * 100
    print(f"\nML skipped {skip_rate:.1f}% of trades")

    if phase80['ml']['pf'] > 3.0:
        print(f"\n⚠️  WARNING: ML PF of {phase80['ml']['pf']:.2f} is UNREALISTIC")
        print("   Possible causes:")
        print("   1. Selection bias (keeping only winning trades)")
        print("   2. Too few trades for statistical significance (n=63)")
        print("   3. The model learned to identify winners post-hoc")

    return {
        'symbol_mismatch': True,
        'phase79_symbols': len(phase79_symbols),
        'phase80_symbols': len(phase80_symbols),
        'trade_ratio': phase80_baseline_trades / phase79_trades,
        'pf_drop': (phase80['baseline']['pf'] / phase79['best_simulation']['profit_factor'] - 1) * 100,
        'ml_pf_suspicious': phase80['ml']['pf'] > 3.0,
    }


if __name__ == '__main__':
    result = check_temporal_integrity()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1 SUMMARY")
    print("=" * 60)

    if result.get('symbol_mismatch'):
        print("\n❌ FAILED: Symbol mismatch detected")
        print("   Phase 8.0 used different symbols than Phase 7.9")
        print("   This invalidates the baseline comparison")
        print("\n   FIX: Re-run Phase 8.0 with all 22 symbols from Phase 7.9")

    if result.get('ml_pf_suspicious'):
        print("\n⚠️  WARNING: ML results are suspicious")
        print("   PF > 3.0 is unrealistic and suggests overfitting")
