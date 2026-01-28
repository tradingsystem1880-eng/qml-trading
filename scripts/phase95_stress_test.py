"""
Phase 9.5: Stress Test on Historical Market Events
===================================================
Tests strategy performance during known market stress periods.

Stress Events Tested:
1. Luna/UST crash (May 2022)
2. FTX collapse (November 2022)
3. August 2024 volatility spike

Methodology:
1. Run backtest on each stress period
2. Measure performance vs baseline
3. Check for catastrophic drawdowns

Success Criteria:
- No single event with PF < 0.5 (catastrophic loss)
- Average stress period PF > 1.0

Usage:
    python scripts/phase95_stress_test.py
    python scripts/phase95_stress_test.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase95_utils import (
    run_full_backtest,
    calculate_metrics,
    load_data,
    DEFAULT_SYMBOLS,
    ValidationResult,
)


# Historical stress events with date ranges
STRESS_EVENTS = {
    "luna_crash": {
        "name": "Luna/UST Crash",
        "start": "2022-05-01",
        "end": "2022-05-31",
        "description": "Algorithmic stablecoin death spiral",
        "expected_behavior": "High volatility, trend patterns",
    },
    "ftx_collapse": {
        "name": "FTX Collapse",
        "start": "2022-11-01",
        "end": "2022-11-30",
        "description": "Exchange insolvency, market contagion",
        "expected_behavior": "Panic selling, correlation spike",
    },
    "aug_2024_volatility": {
        "name": "August 2024 Volatility",
        "start": "2024-08-01",
        "end": "2024-08-31",
        "description": "Market volatility spike",
        "expected_behavior": "Rapid price swings, regime change",
    },
}


def check_data_availability(
    symbols: List[str],
    timeframe: str,
    event_dates: Tuple[str, str],
) -> Tuple[bool, int]:
    """
    Check if data is available for the stress test period.

    Returns:
        Tuple of (has_data, bar_count)
    """
    start_date, end_date = event_dates
    total_bars = 0

    for symbol in symbols:
        df = load_data(symbol, timeframe)
        if df is None:
            continue

        df_time = df['time']
        if df_time.dt.tz is not None:
            df_time = df_time.dt.tz_localize(None)

        mask = (df_time >= start_date) & (df_time <= end_date)
        bars = mask.sum()
        total_bars += bars

    has_data = total_bars > 0
    return has_data, total_bars


def run_stress_test(
    symbols: List[str],
    timeframe: str,
    verbose: bool = True,
) -> Dict:
    """
    Run stress tests across all historical events.

    Returns:
        Dict with results for each stress event
    """
    results = {}

    # First run baseline (full period)
    if verbose:
        print(f"\n{'=' * 50}")
        print("BASELINE (Full Period)")
        print(f"{'=' * 50}\n")

    baseline_trades, baseline_metrics = run_full_backtest(
        symbols, timeframe, verbose=verbose
    )

    baseline_pf = baseline_metrics['profit_factor']
    results['baseline'] = {
        "trades": baseline_metrics['total_trades'],
        "win_rate": baseline_metrics['win_rate'],
        "profit_factor": baseline_pf,
        "expectancy": baseline_metrics['expectancy'],
    }

    if verbose:
        print(f"\nBaseline PF: {baseline_pf:.2f}")

    # Test each stress event
    for event_id, event_info in STRESS_EVENTS.items():
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"STRESS EVENT: {event_info['name']}")
            print(f"{'=' * 50}")
            print(f"Period: {event_info['start']} to {event_info['end']}")
            print(f"Description: {event_info['description']}\n")

        # Check data availability
        has_data, bar_count = check_data_availability(
            symbols, timeframe, (event_info['start'], event_info['end'])
        )

        if not has_data:
            if verbose:
                print("SKIPPED: No data available for this period")
            results[event_id] = {
                "name": event_info['name'],
                "status": "SKIPPED",
                "reason": "No data available",
            }
            continue

        # Run backtest on stress period
        stress_trades, stress_metrics = run_full_backtest(
            symbols,
            timeframe,
            date_filter=(event_info['start'], event_info['end']),
            verbose=verbose,
        )

        stress_pf = stress_metrics['profit_factor']

        # Calculate performance relative to baseline
        if baseline_pf > 0:
            relative_performance = stress_pf / baseline_pf
        else:
            relative_performance = 0

        results[event_id] = {
            "name": event_info['name'],
            "period": {
                "start": event_info['start'],
                "end": event_info['end'],
            },
            "description": event_info['description'],
            "trades": stress_metrics['total_trades'],
            "win_rate": stress_metrics['win_rate'],
            "profit_factor": stress_pf,
            "expectancy": stress_metrics['expectancy'],
            "max_drawdown": stress_metrics['max_drawdown'],
            "relative_to_baseline": relative_performance,
            "status": "COMPLETED",
        }

        if verbose:
            print(f"\nStress Period Results:")
            print(f"  Trades:        {stress_metrics['total_trades']}")
            print(f"  Win Rate:      {stress_metrics['win_rate']:.1%}")
            print(f"  Profit Factor: {stress_pf:.2f}")
            print(f"  Expectancy:    {stress_metrics['expectancy']:.2f}R")
            print(f"  Max Drawdown:  {stress_metrics['max_drawdown']:.2f}R")
            print(f"  vs Baseline:   {relative_performance:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stress test on historical events")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--min-pf', type=float, default=0.5, help='Min acceptable PF per event')
    parser.add_argument('--avg-pf', type=float, default=1.0, help='Min average PF across events')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.5: STRESS TEST ON HISTORICAL EVENTS")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"\nStress events to test:")
    for event_id, event_info in STRESS_EVENTS.items():
        print(f"  - {event_info['name']} ({event_info['start']} to {event_info['end']})")

    print(f"\nSuccess criteria:")
    print(f"  - No event with PF < {args.min_pf:.1f}")
    print(f"  - Average stress PF > {args.avg_pf:.1f}")

    # Run stress tests
    stress_results = run_stress_test(symbols, args.timeframe, verbose=True)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Event':<30} {'Trades':>8} {'WR':>8} {'PF':>8} {'vs Base':>10} {'Status':>10}")
    print("-" * 80)

    # Baseline row
    bl = stress_results['baseline']
    print(f"{'Baseline (Full Period)':<30} {bl['trades']:>8} {bl['win_rate']:>7.1%} "
          f"{bl['profit_factor']:>8.2f} {'100%':>10} {'REF':>10}")

    # Stress events
    stress_pfs = []
    any_catastrophic = False
    min_stress_pf = float('inf')

    for event_id, event_info in STRESS_EVENTS.items():
        result = stress_results.get(event_id, {})

        if result.get('status') == 'SKIPPED':
            print(f"{result['name']:<30} {'--':>8} {'--':>8} {'--':>8} {'--':>10} {'SKIPPED':>10}")
            continue

        trades = result['trades']
        wr = result['win_rate']
        pf = result['profit_factor']
        rel = result['relative_to_baseline']

        if trades == 0:
            status = "NO TRADES"
        elif pf < args.min_pf:
            status = "CRITICAL"
            any_catastrophic = True
        elif pf < 1.0:
            status = "POOR"
        else:
            status = "OK"

        stress_pfs.append(pf)
        if pf < min_stress_pf:
            min_stress_pf = pf

        print(f"{result['name']:<30} {trades:>8} {wr:>7.1%} {pf:>8.2f} {rel:>9.0%} {status:>10}")

    # Calculate average stress PF
    if stress_pfs:
        avg_stress_pf = np.mean(stress_pfs)
    else:
        avg_stress_pf = 0

    print(f"\n{'Average Stress Period PF:':<30} {avg_stress_pf:>8.2f}")
    print(f"{'Minimum Stress Period PF:':<30} {min_stress_pf:>8.2f}")

    # Verdict
    pf_check = not any_catastrophic and (min_stress_pf >= args.min_pf or min_stress_pf == float('inf'))
    avg_check = avg_stress_pf >= args.avg_pf or len(stress_pfs) == 0
    passed = pf_check and avg_check

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if len(stress_pfs) == 0:
        print(f"\n⚠️  INCONCLUSIVE: No stress test data available")
        print(f"   Cannot validate performance during market stress")
        passed = None
    elif passed:
        print(f"\n✅ PASS: Strategy survives historical stress events")
        print(f"   No catastrophic losses (min PF = {min_stress_pf:.2f})")
        print(f"   Average stress PF ({avg_stress_pf:.2f}) >= {args.avg_pf:.1f}")
        print(f"   Strategy demonstrates resilience to market shocks")
    else:
        print(f"\n❌ FAIL: Strategy vulnerable to market stress")
        if any_catastrophic:
            print(f"   Catastrophic loss detected (PF < {args.min_pf:.1f})")
        if avg_stress_pf < args.avg_pf:
            print(f"   Average stress PF ({avg_stress_pf:.2f}) < {args.avg_pf:.1f}")
        print(f"   Consider adding stress regime detection or reducing exposure")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"stress_test_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "stress_test_events",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "min_pf_threshold": args.min_pf,
            "avg_pf_threshold": args.avg_pf,
        },
        "stress_results": stress_results,
        "summary": {
            "events_tested": len(stress_pfs),
            "avg_stress_pf": avg_stress_pf,
            "min_stress_pf": min_stress_pf if min_stress_pf != float('inf') else None,
            "any_catastrophic": any_catastrophic,
        },
        "passed": passed,
        "verdict": "PASS" if passed else ("FAIL" if passed is False else "INCONCLUSIVE"),
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="stress_test_events",
        passed=passed if passed is not None else False,
        metric_value=avg_stress_pf,
        threshold=args.avg_pf,
        details=results,
    )


if __name__ == "__main__":
    main()
