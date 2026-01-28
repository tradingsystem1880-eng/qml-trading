"""
Phase 9.5: Out-of-Sample Holdout Test
=====================================
Tests strategy performance on unseen (most recent) data.

Methodology:
1. Reserve most recent 3 months as holdout set
2. All previous optimization/analysis was done on prior data
3. Run strategy on holdout period
4. Verify edge persists on unseen data

Success Criteria:
- Profit Factor > 2.0
- Win Rate > 48%

Usage:
    python scripts/phase95_oos_holdout.py
    python scripts/phase95_oos_holdout.py --holdout-months 3
    python scripts/phase95_oos_holdout.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase95_utils import (
    run_full_backtest,
    calculate_metrics,
    load_data,
    DEFAULT_SYMBOLS,
    ValidationResult,
)


def get_date_ranges(
    symbols: List[str],
    timeframe: str,
    holdout_months: int = 3,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Get date ranges for in-sample and out-of-sample periods.

    Returns:
        Tuple of ((is_start, is_end), (oos_start, oos_end))
    """
    # Find latest date across all symbols
    latest_date = None
    earliest_date = None

    for symbol in symbols:
        df = load_data(symbol, timeframe)
        if df is None:
            continue

        df_time = df['time']
        if df_time.dt.tz is not None:
            df_time = df_time.dt.tz_localize(None)

        symbol_latest = df_time.max()
        symbol_earliest = df_time.min()

        if latest_date is None or symbol_latest > latest_date:
            latest_date = symbol_latest
        if earliest_date is None or symbol_earliest < earliest_date:
            earliest_date = symbol_earliest

    if latest_date is None:
        raise ValueError("No data found for any symbols")

    # Calculate split point (3 months before latest)
    split_date = latest_date - timedelta(days=holdout_months * 30)

    # In-sample: all data before split
    is_start = earliest_date.strftime("%Y-%m-%d")
    is_end = (split_date - timedelta(days=1)).strftime("%Y-%m-%d")

    # Out-of-sample: split to latest
    oos_start = split_date.strftime("%Y-%m-%d")
    oos_end = latest_date.strftime("%Y-%m-%d")

    return (is_start, is_end), (oos_start, oos_end)


def main():
    parser = argparse.ArgumentParser(description="Out-of-sample holdout test")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--holdout-months', type=int, default=3, help='Months to hold out')
    parser.add_argument('--min-pf', type=float, default=2.0, help='Minimum profit factor')
    parser.add_argument('--min-wr', type=float, default=0.48, help='Minimum win rate')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.5: OUT-OF-SAMPLE HOLDOUT TEST")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Holdout period: {args.holdout_months} months")
    print(f"Success criteria: PF > {args.min_pf:.1f}, WR > {args.min_wr:.0%}")

    # Get date ranges
    (is_start, is_end), (oos_start, oos_end) = get_date_ranges(
        symbols, args.timeframe, args.holdout_months
    )

    print(f"\nIn-Sample Period:     {is_start} to {is_end}")
    print(f"Out-of-Sample Period: {oos_start} to {oos_end}")

    # Run in-sample backtest (for reference)
    print(f"\n{'=' * 70}")
    print("IN-SAMPLE BACKTEST (Reference)")
    print(f"{'=' * 70}\n")

    is_trades, is_metrics = run_full_backtest(
        symbols,
        args.timeframe,
        date_filter=(is_start, is_end),
        verbose=True,
    )

    print(f"\nIn-Sample Results:")
    print(f"  Trades:        {is_metrics['total_trades']}")
    print(f"  Win Rate:      {is_metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {is_metrics['profit_factor']:.2f}")
    print(f"  Expectancy:    {is_metrics['expectancy']:.2f}R")

    # Run out-of-sample backtest
    print(f"\n{'=' * 70}")
    print("OUT-OF-SAMPLE BACKTEST (Validation)")
    print(f"{'=' * 70}\n")

    oos_trades, oos_metrics = run_full_backtest(
        symbols,
        args.timeframe,
        date_filter=(oos_start, oos_end),
        verbose=True,
    )

    if len(oos_trades) < 10:
        print(f"\nWARNING: Only {len(oos_trades)} OOS trades. Results may not be reliable.")

    print(f"\nOut-of-Sample Results:")
    print(f"  Trades:        {oos_metrics['total_trades']}")
    print(f"  Win Rate:      {oos_metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {oos_metrics['profit_factor']:.2f}")
    print(f"  Expectancy:    {oos_metrics['expectancy']:.2f}R")

    # Comparison
    print(f"\n{'=' * 70}")
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'=' * 70}")

    metrics_comparison = [
        ("Win Rate", is_metrics['win_rate'], oos_metrics['win_rate'], "%"),
        ("Profit Factor", is_metrics['profit_factor'], oos_metrics['profit_factor'], ""),
        ("Expectancy", is_metrics['expectancy'], oos_metrics['expectancy'], "R"),
        ("Avg Win", is_metrics['avg_win_r'], oos_metrics['avg_win_r'], "R"),
        ("Avg Loss", is_metrics['avg_loss_r'], oos_metrics['avg_loss_r'], "R"),
    ]

    print(f"\n{'Metric':<18} {'In-Sample':>12} {'Out-of-Sample':>15} {'Change':>12}")
    print("-" * 60)

    for name, is_val, oos_val, unit in metrics_comparison:
        if unit == "%":
            is_str = f"{is_val:.1%}"
            oos_str = f"{oos_val:.1%}"
            change = (oos_val - is_val) * 100
            change_str = f"{change:+.1f}pp"
        else:
            is_str = f"{is_val:.2f}{unit}"
            oos_str = f"{oos_val:.2f}{unit}"
            if is_val != 0:
                change_pct = 100 * (oos_val - is_val) / abs(is_val)
                change_str = f"{change_pct:+.0f}%"
            else:
                change_str = "N/A"

        print(f"{name:<18} {is_str:>12} {oos_str:>15} {change_str:>12}")

    # Calculate degradation ratio
    if is_metrics['profit_factor'] > 0:
        pf_degradation = oos_metrics['profit_factor'] / is_metrics['profit_factor']
    else:
        pf_degradation = 0

    print(f"\nDegradation Ratio (OOS/IS):")
    print(f"  Profit Factor: {pf_degradation:.2f}x")

    # Verdict
    pf_pass = oos_metrics['profit_factor'] >= args.min_pf
    wr_pass = oos_metrics['win_rate'] >= args.min_wr
    passed = pf_pass and wr_pass

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    status_pf = "✓" if pf_pass else "✗"
    status_wr = "✓" if wr_pass else "✗"

    print(f"\n{status_pf} Profit Factor: {oos_metrics['profit_factor']:.2f} (threshold: {args.min_pf:.1f})")
    print(f"{status_wr} Win Rate: {oos_metrics['win_rate']:.1%} (threshold: {args.min_wr:.0%})")

    if passed:
        print(f"\n✅ PASS: Strategy maintains edge on unseen data")
        print(f"   OOS performance validates in-sample results")
        if pf_degradation >= 0.7:
            print(f"   Minimal degradation ({pf_degradation:.2f}x) suggests robust edge")
        else:
            print(f"   Some degradation ({pf_degradation:.2f}x) but still profitable")
    else:
        print(f"\n❌ FAIL: Strategy does NOT maintain edge on unseen data")
        reasons = []
        if not pf_pass:
            reasons.append(f"PF {oos_metrics['profit_factor']:.2f} < {args.min_pf:.1f}")
        if not wr_pass:
            reasons.append(f"WR {oos_metrics['win_rate']:.1%} < {args.min_wr:.0%}")
        print(f"   Failed criteria: {', '.join(reasons)}")
        print(f"   Possible overfitting or regime change")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"oos_holdout_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "oos_holdout",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "holdout_months": args.holdout_months,
            "min_pf": args.min_pf,
            "min_wr": args.min_wr,
        },
        "date_ranges": {
            "in_sample": {"start": is_start, "end": is_end},
            "out_of_sample": {"start": oos_start, "end": oos_end},
        },
        "in_sample_metrics": {
            "total_trades": is_metrics['total_trades'],
            "win_rate": is_metrics['win_rate'],
            "profit_factor": is_metrics['profit_factor'],
            "expectancy": is_metrics['expectancy'],
        },
        "out_of_sample_metrics": {
            "total_trades": oos_metrics['total_trades'],
            "win_rate": oos_metrics['win_rate'],
            "profit_factor": oos_metrics['profit_factor'],
            "expectancy": oos_metrics['expectancy'],
        },
        "degradation_ratio": pf_degradation,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="oos_holdout",
        passed=passed,
        metric_value=oos_metrics['profit_factor'],
        threshold=args.min_pf,
        details=results,
    )


if __name__ == "__main__":
    main()
