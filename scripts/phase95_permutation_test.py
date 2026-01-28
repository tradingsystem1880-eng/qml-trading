"""
Phase 9.5: Permutation Test for Profit Factor
==============================================
Tests statistical significance of the trading edge by shuffling trade outcomes.

Methodology:
1. Calculate real profit factor from actual trades
2. Shuffle trade R-returns 1000x times
3. Recalculate PF for each shuffle
4. Check if real PF is in top 5% (p-value < 0.05)

Success Criteria: Real PF must be in top 5% (>95th percentile of shuffled PFs)

Usage:
    python scripts/phase95_permutation_test.py
    python scripts/phase95_permutation_test.py --iterations 5000
    python scripts/phase95_permutation_test.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase95_utils import (
    run_full_backtest,
    calculate_metrics,
    get_r_returns,
    DEFAULT_SYMBOLS,
    ValidationResult,
)


def calculate_profit_factor(r_returns: np.ndarray) -> float:
    """Calculate profit factor from R-multiple returns."""
    winners = r_returns[r_returns > 0]
    losers = r_returns[r_returns <= 0]

    gross_profit = np.sum(winners) if len(winners) > 0 else 0
    gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 0

    return gross_profit / gross_loss if gross_loss > 0 else float('inf')


def run_permutation_test(
    r_returns: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[float, float, np.ndarray]:
    """
    Run permutation test on trade returns.

    Tests the null hypothesis that the observed edge could occur by chance.
    Method: Randomly flip the sign of returns (simulating random long/short
    entry decisions), then calculate PF. If real PF >> permuted PFs, the
    edge is statistically significant.

    Args:
        r_returns: Array of R-multiple returns
        n_iterations: Number of permutations
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (real_pf, p_value, permuted_pfs)
    """
    np.random.seed(seed)

    # Calculate real profit factor
    real_pf = calculate_profit_factor(r_returns)

    if verbose:
        print(f"\nReal Profit Factor: {real_pf:.2f}")
        print(f"Running {n_iterations} permutations...")

    # Run permutations with sign randomization
    # This tests: "If we randomly chose long/short, would we get this PF?"
    permuted_pfs = []
    abs_returns = np.abs(r_returns)

    for i in range(n_iterations):
        # Randomly assign signs (simulates random long/short decisions)
        random_signs = np.random.choice([-1, 1], size=len(r_returns))
        randomized = abs_returns * random_signs
        perm_pf = calculate_profit_factor(randomized)
        permuted_pfs.append(perm_pf)

        if verbose and (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_iterations} permutations...")

    permuted_pfs = np.array(permuted_pfs)

    # Calculate p-value (proportion of permuted PFs >= real PF)
    p_value = np.mean(permuted_pfs >= real_pf)

    return real_pf, p_value, permuted_pfs


def main():
    parser = argparse.ArgumentParser(description="Permutation test for profit factor")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of permutations')
    parser.add_argument('--threshold', type=float, default=0.05, help='Significance threshold (default 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.5: PERMUTATION TEST FOR PROFIT FACTOR")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Iterations: {args.iterations}")
    print(f"Significance threshold: {args.threshold}")

    # Run backtest to get trades
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70 + "\n")

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

    # Run permutation test
    print(f"\n{'=' * 70}")
    print("PERMUTATION TEST")
    print(f"{'=' * 70}")

    r_returns = get_r_returns(trades)
    real_pf, p_value, permuted_pfs = run_permutation_test(
        r_returns,
        n_iterations=args.iterations,
        seed=args.seed,
        verbose=True,
    )

    # Calculate percentile of real PF
    percentile = 100 * (1 - p_value)

    # Distribution statistics
    pf_mean = np.mean(permuted_pfs)
    pf_std = np.std(permuted_pfs)
    pf_median = np.median(permuted_pfs)
    pf_p95 = np.percentile(permuted_pfs, 95)
    pf_p99 = np.percentile(permuted_pfs, 99)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"\nNull Distribution (shuffled PFs):")
    print(f"  Mean:   {pf_mean:.3f}")
    print(f"  Std:    {pf_std:.3f}")
    print(f"  Median: {pf_median:.3f}")
    print(f"  95th:   {pf_p95:.3f}")
    print(f"  99th:   {pf_p99:.3f}")

    print(f"\nReal Profit Factor: {real_pf:.3f}")
    print(f"Percentile:         {percentile:.1f}th")
    print(f"P-value:            {p_value:.4f}")

    # Verdict
    passed = p_value < args.threshold

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if passed:
        print(f"\n✅ PASS: Real PF ({real_pf:.2f}) is statistically significant")
        print(f"   P-value {p_value:.4f} < {args.threshold} threshold")
        print(f"   Rank: {percentile:.1f}th percentile (top {100-percentile:.1f}%)")
        print(f"   The edge is NOT due to random chance")
    else:
        print(f"\n❌ FAIL: Real PF ({real_pf:.2f}) is NOT statistically significant")
        print(f"   P-value {p_value:.4f} >= {args.threshold} threshold")
        print(f"   Rank: {percentile:.1f}th percentile")
        print(f"   The edge could be due to random chance")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"permutation_test_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "permutation_profit_factor",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "iterations": args.iterations,
            "threshold": args.threshold,
            "seed": args.seed,
        },
        "baseline_metrics": {
            "total_trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "profit_factor": metrics['profit_factor'],
            "expectancy": metrics['expectancy'],
        },
        "results": {
            "real_pf": real_pf,
            "p_value": p_value,
            "percentile": percentile,
            "null_distribution": {
                "mean": pf_mean,
                "std": pf_std,
                "median": pf_median,
                "p95": pf_p95,
                "p99": pf_p99,
            },
        },
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="permutation_profit_factor",
        passed=passed,
        metric_value=p_value,
        threshold=args.threshold,
        details=results,
    )


if __name__ == "__main__":
    main()
