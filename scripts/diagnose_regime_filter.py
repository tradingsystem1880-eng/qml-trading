"""
Phase 9.1: Regime Filter Diagnostic
=====================================
Diagnose why 0 patterns are passing the scorer despite valid patterns existing.

This script:
1. Shows raw pattern count WITHOUT regime filter
2. Shows rejection breakdown by regime type
3. Recommends optimized filter settings

Usage:
    python scripts/diagnose_regime_filter.py --symbols BTCUSDT,ETHUSDT,BNBUSDT
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.historical_detector import HistoricalSwingDetector
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.regime import MarketRegimeDetector, MarketRegime
from src.detection.config import (
    SwingDetectionConfig,
    PatternValidationConfig,
    PatternScoringConfig,
)

SWING_CONFIG = SwingDetectionConfig(
    atr_period=14,
    lookback=5,
    lookforward=3,
    min_zscore=0.5,
    min_threshold_pct=0.0005,
    atr_multiplier=0.5,
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=10.0,
    p5_max_symmetry_atr=5.0,
    min_pattern_bars=8,
    max_pattern_bars=200,
)

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]


def load_data(symbol: str) -> pd.DataFrame:
    """Load price data."""
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / "4h_master.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})
    return df


def diagnose_symbol(symbol: str) -> Dict:
    """Run full diagnostic on a symbol."""
    normalized = normalize_symbol(symbol)
    df = load_data(symbol)

    # 1. Detect swings and patterns
    detector = HistoricalSwingDetector(SWING_CONFIG, normalized, "4h")
    swings = detector.detect(df)

    validator = PatternValidator(PATTERN_CONFIG)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {symbol}")
    print(f"{'='*60}")
    print(f"  Bars: {len(df)}")
    print(f"  Swings detected: {len(swings)}")
    print(f"  Valid patterns (geometry): {len(valid_patterns)}")

    if not valid_patterns:
        return {'valid': 0, 'passed': 0}

    # 2. Score WITHOUT regime filter
    scorer_no_regime = PatternScorer(PatternScoringConfig())
    passed_no_regime = 0
    for p in valid_patterns:
        sr = scorer_no_regime.score(p, df=df, regime_result=None)
        if sr.tier != PatternTier.REJECT:
            passed_no_regime += 1

    print(f"  Passed scoring (NO regime filter): {passed_no_regime}")

    # 3. Score WITH current regime filter (ADX=35 hard reject)
    regime_detector = MarketRegimeDetector()
    scorer_current = PatternScorer(PatternScoringConfig())

    passed_current = 0
    rejection_reasons = {
        'hard_reject_trending': 0,
        'low_score_trending': 0,
        'low_score_other': 0,
        'passed': 0,
    }
    regime_breakdown = {r.value: 0 for r in MarketRegime}
    adx_values = []

    for p in valid_patterns:
        # Get regime at pattern time
        # Need 110+ bars for regime detector (ADX/RSI periods + vol lookback)
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)  # 150 bars to ensure enough data
        window_df = df.iloc[window_start:p5_idx + 1].copy()
        regime_result = regime_detector.get_regime(window_df)

        regime_breakdown[regime_result.regime.value] += 1
        adx_values.append(regime_result.adx)

        sr = scorer_current.score(p, df=df, regime_result=regime_result)

        if sr.tier != PatternTier.REJECT:
            rejection_reasons['passed'] += 1
            passed_current += 1
        else:
            # Why rejected?
            if regime_result.regime == MarketRegime.TRENDING and regime_result.adx > 35:
                rejection_reasons['hard_reject_trending'] += 1
            elif regime_result.regime == MarketRegime.TRENDING:
                rejection_reasons['low_score_trending'] += 1
            else:
                rejection_reasons['low_score_other'] += 1

    print(f"  Passed scoring (WITH regime filter): {passed_current}")

    print(f"\n  REGIME BREAKDOWN:")
    for regime, count in regime_breakdown.items():
        print(f"    {regime}: {count}")

    print(f"\n  REJECTION BREAKDOWN:")
    print(f"    Hard reject (TRENDING + ADX>35): {rejection_reasons['hard_reject_trending']}")
    print(f"    Low score (TRENDING, ADX<=35):   {rejection_reasons['low_score_trending']}")
    print(f"    Low score (other regimes):        {rejection_reasons['low_score_other']}")
    print(f"    PASSED:                           {rejection_reasons['passed']}")

    if adx_values:
        print(f"\n  ADX STATISTICS:")
        print(f"    Min: {min(adx_values):.1f}")
        print(f"    Max: {max(adx_values):.1f}")
        print(f"    Mean: {np.mean(adx_values):.1f}")
        print(f"    Median: {np.median(adx_values):.1f}")
        print(f"    % above 35: {100 * sum(1 for a in adx_values if a > 35) / len(adx_values):.1f}%")
        print(f"    % above 45: {100 * sum(1 for a in adx_values if a > 45) / len(adx_values):.1f}%")

    # 4. Test profit-optimized config
    profit_config = PatternScoringConfig(
        regime_hard_reject_adx=45.0,  # Only reject extreme trends
        regime_trending_score=0.5,     # Moderate penalty
        tier_c_min=0.35,               # Lower threshold
    )
    scorer_profit = PatternScorer(profit_config)

    passed_profit = 0
    for p in valid_patterns:
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)  # 150 bars for proper regime calc
        window_df = df.iloc[window_start:p5_idx + 1].copy()
        regime_result = regime_detector.get_regime(window_df)

        sr = scorer_profit.score(p, df=df, regime_result=regime_result)
        if sr.tier != PatternTier.REJECT:
            passed_profit += 1

    print(f"\n  PROFIT-OPTIMIZED CONFIG (ADX>45, trending=0.5, tier_c=0.35):")
    print(f"    Patterns passed: {passed_profit}")

    # 5. Test with regime disabled
    passed_disabled = 0
    for p in valid_patterns:
        sr = scorer_no_regime.score(p, df=df, regime_result=None)
        if sr.tier != PatternTier.REJECT:
            passed_disabled += 1

    print(f"\n  REGIME DISABLED (baseline):")
    print(f"    Patterns passed: {passed_disabled}")

    return {
        'valid': len(valid_patterns),
        'no_regime': passed_no_regime,
        'current': passed_current,
        'profit_optimized': passed_profit,
        'regime_breakdown': regime_breakdown,
        'rejection_reasons': rejection_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose regime filter issues")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 60)
    print("PHASE 9.1: REGIME FILTER DIAGNOSTIC")
    print("=" * 60)
    print(f"Analyzing {len(symbols)} symbols...")

    total_valid = 0
    total_current = 0
    total_profit = 0
    total_disabled = 0

    for symbol in symbols:
        try:
            result = diagnose_symbol(symbol)
            total_valid += result['valid']
            total_current += result['current']
            total_profit += result['profit_optimized']
            total_disabled += result['no_regime']
        except Exception as e:
            print(f"\nError processing {symbol}: {e}")

    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Total valid patterns (geometry): {total_valid}")
    print(f"Passed with CURRENT filter:      {total_current} ({100*total_current/total_valid:.1f}% pass rate)" if total_valid > 0 else "")
    print(f"Passed with PROFIT-OPTIMIZED:    {total_profit} ({100*total_profit/total_valid:.1f}% pass rate)" if total_valid > 0 else "")
    print(f"Passed with REGIME DISABLED:     {total_disabled} ({100*total_disabled/total_valid:.1f}% pass rate)" if total_valid > 0 else "")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if total_profit > total_current * 2:
        print("Use PROFIT-OPTIMIZED config:")
        print("  regime_hard_reject_adx: 45.0")
        print("  regime_trending_score: 0.5")
        print("  tier_c_min: 0.35")
    elif total_disabled > total_current * 3:
        print("Consider DISABLING regime filter entirely for testing")
    else:
        print("Current filter seems reasonable")

    print("=" * 60)


if __name__ == "__main__":
    main()
