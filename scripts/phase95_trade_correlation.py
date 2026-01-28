"""
Phase 9.5: Trade Autocorrelation Test
=====================================
Tests for serial correlation in consecutive trade returns.

Methodology:
1. Calculate autocorrelation of trade R-returns at lag 1
2. High autocorrelation indicates dependent trades (reduces effective sample size)
3. Could indicate streak patterns or regime dependencies

Success Criteria: |autocorrelation| < 0.1

Why This Matters:
- Independent trades: N trades = N independent observations
- Correlated trades: Effective sample size < N
- High correlation invalidates statistical significance tests

Usage:
    python scripts/phase95_trade_correlation.py
    python scripts/phase95_trade_correlation.py --threshold 0.1
    python scripts/phase95_trade_correlation.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase95_utils import (
    run_full_backtest,
    get_r_returns,
    DEFAULT_SYMBOLS,
    ValidationResult,
)


def calculate_autocorrelation(returns: np.ndarray, lag: int = 1) -> Tuple[float, float]:
    """
    Calculate autocorrelation at specified lag.

    Args:
        returns: Array of returns
        lag: Lag for autocorrelation calculation

    Returns:
        Tuple of (autocorrelation coefficient, p-value)
    """
    if len(returns) <= lag + 1:
        return 0.0, 1.0

    # Get lagged series
    y = returns[lag:]
    x = returns[:-lag]

    # Calculate correlation
    corr, p_value = stats.pearsonr(x, y)

    return float(corr), float(p_value)


def calculate_runs_test(returns: np.ndarray) -> Tuple[float, float]:
    """
    Perform Wald-Wolfowitz runs test for randomness.

    Tests if wins/losses are randomly distributed or show streaks.

    Returns:
        Tuple of (z-statistic, p-value)
    """
    # Convert to binary (win/loss)
    binary = (returns > 0).astype(int)

    if len(binary) < 10:
        return 0.0, 1.0

    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1

    # Count wins and losses
    n1 = np.sum(binary)  # wins
    n2 = len(binary) - n1  # losses

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Expected runs and variance
    n = n1 + n2
    expected_runs = (2 * n1 * n2) / n + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))

    if variance <= 0:
        return 0.0, 1.0

    # Z-statistic
    z = (runs - expected_runs) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(z), float(p_value)


def calculate_ljung_box(returns: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """
    Perform Ljung-Box test for autocorrelation at multiple lags.

    Returns:
        Tuple of (Q-statistic, p-value)
    """
    n = len(returns)

    if n <= lags + 1:
        return 0.0, 1.0

    # Calculate autocorrelations at each lag
    autocorrs = []
    for k in range(1, lags + 1):
        corr, _ = calculate_autocorrelation(returns, lag=k)
        autocorrs.append(corr)

    autocorrs = np.array(autocorrs)

    # Ljung-Box Q statistic
    weights = 1 / (n - np.arange(1, lags + 1))
    q_stat = n * (n + 2) * np.sum(weights * autocorrs ** 2)

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(q_stat, df=lags)

    return float(q_stat), float(p_value)


def analyze_streak_patterns(returns: np.ndarray) -> Dict:
    """
    Analyze winning and losing streak patterns.

    Returns:
        Dict with streak statistics
    """
    if len(returns) == 0:
        return {"max_win_streak": 0, "max_loss_streak": 0}

    # Convert to binary
    wins = returns > 0

    # Find streaks
    win_streaks = []
    loss_streaks = []
    current_streak = 0
    current_is_win = None

    for w in wins:
        if current_is_win is None:
            current_is_win = w
            current_streak = 1
        elif w == current_is_win:
            current_streak += 1
        else:
            if current_is_win:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_is_win = w
            current_streak = 1

    # Add final streak
    if current_is_win is not None:
        if current_is_win:
            win_streaks.append(current_streak)
        else:
            loss_streaks.append(current_streak)

    return {
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
        "avg_win_streak": np.mean(win_streaks) if win_streaks else 0,
        "avg_loss_streak": np.mean(loss_streaks) if loss_streaks else 0,
        "num_win_streaks": len(win_streaks),
        "num_loss_streaks": len(loss_streaks),
    }


def main():
    parser = argparse.ArgumentParser(description="Trade autocorrelation test")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--threshold', type=float, default=0.1, help='Max acceptable |autocorrelation|')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.5: TRADE AUTOCORRELATION TEST")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Threshold: |r| < {args.threshold}")

    # Run backtest to get trades
    print(f"\n{'=' * 70}")
    print("RUNNING BACKTEST")
    print(f"{'=' * 70}\n")

    trades, metrics = run_full_backtest(symbols, args.timeframe, verbose=True)

    if len(trades) < 30:
        print(f"\nERROR: Insufficient trades ({len(trades)}). Need at least 30.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("BASELINE METRICS")
    print(f"{'=' * 70}")
    print(f"\nTotal Trades:   {metrics['total_trades']}")
    print(f"Win Rate:       {metrics['win_rate']:.1%}")
    print(f"Profit Factor:  {metrics['profit_factor']:.2f}")
    print(f"Expectancy:     {metrics['expectancy']:.2f}R")

    # Get R-returns
    r_returns = get_r_returns(trades)

    # Run correlation tests
    print(f"\n{'=' * 70}")
    print("CORRELATION ANALYSIS")
    print(f"{'=' * 70}")

    # Lag-1 autocorrelation
    autocorr_1, autocorr_p = calculate_autocorrelation(r_returns, lag=1)
    print(f"\nLag-1 Autocorrelation:")
    print(f"  r = {autocorr_1:+.4f}")
    print(f"  p-value = {autocorr_p:.4f}")
    print(f"  Interpretation: {'Strong' if abs(autocorr_1) > 0.3 else 'Moderate' if abs(autocorr_1) > 0.1 else 'Weak'} serial correlation")

    # Lag-2 autocorrelation
    autocorr_2, _ = calculate_autocorrelation(r_returns, lag=2)
    print(f"\nLag-2 Autocorrelation:")
    print(f"  r = {autocorr_2:+.4f}")

    # Runs test
    runs_z, runs_p = calculate_runs_test(r_returns)
    print(f"\nWald-Wolfowitz Runs Test:")
    print(f"  z = {runs_z:+.4f}")
    print(f"  p-value = {runs_p:.4f}")
    if runs_p < 0.05:
        print(f"  Interpretation: Win/loss sequence shows non-random pattern")
    else:
        print(f"  Interpretation: Win/loss sequence appears random")

    # Ljung-Box test
    lb_q, lb_p = calculate_ljung_box(r_returns, lags=10)
    print(f"\nLjung-Box Test (10 lags):")
    print(f"  Q = {lb_q:.2f}")
    print(f"  p-value = {lb_p:.4f}")
    if lb_p < 0.05:
        print(f"  Interpretation: Significant autocorrelation at some lag")
    else:
        print(f"  Interpretation: No significant autocorrelation detected")

    # Streak analysis
    print(f"\n{'=' * 70}")
    print("STREAK ANALYSIS")
    print(f"{'=' * 70}")

    streaks = analyze_streak_patterns(r_returns)
    print(f"\nWinning Streaks:")
    print(f"  Maximum: {streaks['max_win_streak']} consecutive wins")
    print(f"  Average: {streaks['avg_win_streak']:.1f} wins")
    print(f"  Count:   {streaks['num_win_streaks']} streaks")

    print(f"\nLosing Streaks:")
    print(f"  Maximum: {streaks['max_loss_streak']} consecutive losses")
    print(f"  Average: {streaks['avg_loss_streak']:.1f} losses")
    print(f"  Count:   {streaks['num_loss_streaks']} streaks")

    # Effective sample size calculation
    effective_n = len(r_returns) * (1 - abs(autocorr_1)) / (1 + abs(autocorr_1))
    print(f"\nEffective Sample Size:")
    print(f"  Actual trades: {len(r_returns)}")
    print(f"  Effective N:   {effective_n:.0f} ({100*effective_n/len(r_returns):.0f}% of actual)")

    # Verdict
    passed = abs(autocorr_1) < args.threshold

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if passed:
        print(f"\n✅ PASS: Trade returns show minimal autocorrelation")
        print(f"   |r| = {abs(autocorr_1):.4f} < {args.threshold} threshold")
        print(f"   Trades can be treated as approximately independent")
        print(f"   Statistical significance tests remain valid")
    else:
        print(f"\n❌ FAIL: Trade returns show significant autocorrelation")
        print(f"   |r| = {abs(autocorr_1):.4f} >= {args.threshold} threshold")
        print(f"   Effective sample size reduced to ~{effective_n:.0f} trades")
        print(f"   Consider:")
        print(f"   - Using longer time gaps between trades")
        print(f"   - Adjusting confidence intervals for dependence")
        print(f"   - Investigating regime/streak patterns")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"trade_correlation_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "trade_autocorrelation",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "threshold": args.threshold,
        },
        "baseline_metrics": {
            "total_trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "profit_factor": metrics['profit_factor'],
            "expectancy": metrics['expectancy'],
        },
        "correlation_tests": {
            "lag1_autocorrelation": {
                "r": autocorr_1,
                "p_value": autocorr_p,
            },
            "lag2_autocorrelation": {
                "r": autocorr_2,
            },
            "runs_test": {
                "z_statistic": runs_z,
                "p_value": runs_p,
            },
            "ljung_box": {
                "q_statistic": lb_q,
                "p_value": lb_p,
                "lags": 10,
            },
        },
        "streak_analysis": streaks,
        "effective_sample_size": {
            "actual_trades": len(r_returns),
            "effective_n": effective_n,
            "reduction_pct": 100 * (1 - effective_n / len(r_returns)),
        },
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="trade_autocorrelation",
        passed=passed,
        metric_value=abs(autocorr_1),
        threshold=args.threshold,
        details=results,
    )


if __name__ == "__main__":
    main()
