"""
Phase 9.5: Monte Carlo Drawdown Analysis
=========================================
Bootstrap simulation to calculate confidence intervals on maximum drawdown.

Methodology:
1. Get R-returns from actual trades
2. Bootstrap resample 10,000 equity curves
3. Calculate max drawdown for each simulated curve
4. Build 95% confidence interval on max drawdown

Success Criteria: Expected max drawdown at 95% CI < 20%

Usage:
    python scripts/phase95_monte_carlo_drawdown.py
    python scripts/phase95_monte_carlo_drawdown.py --simulations 10000
    python scripts/phase95_monte_carlo_drawdown.py --account-size 10000
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


def calculate_max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown as percentage from equity curve."""
    if len(equity_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = (running_max - equity_curve) / running_max
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)

    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def simulate_equity_curve(
    r_returns: np.ndarray,
    starting_equity: float,
    risk_per_trade_pct: float,
) -> np.ndarray:
    """
    Simulate equity curve from R-returns.

    Args:
        r_returns: Array of R-multiple returns (bootstrapped)
        starting_equity: Starting account equity
        risk_per_trade_pct: Risk per trade as percentage (e.g., 0.01 for 1%)

    Returns:
        Equity curve array
    """
    equity = starting_equity
    curve = [equity]

    for r_mult in r_returns:
        # Dollar P&L = equity * risk_pct * R-multiple
        pnl = equity * risk_per_trade_pct * r_mult
        equity = equity + pnl
        curve.append(max(equity, 0))  # Can't go negative

    return np.array(curve)


def run_monte_carlo(
    r_returns: np.ndarray,
    n_simulations: int = 10000,
    starting_equity: float = 10000,
    risk_per_trade_pct: float = 0.01,
    trades_per_sim: int = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Run Monte Carlo simulation of equity curves.

    Args:
        r_returns: Array of R-multiple returns from actual trades
        n_simulations: Number of simulations to run
        starting_equity: Starting account equity
        risk_per_trade_pct: Risk per trade (default 1%)
        trades_per_sim: Number of trades per simulation (default: same as actual)
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (max_drawdowns array, statistics dict)
    """
    np.random.seed(seed)

    if trades_per_sim is None:
        trades_per_sim = len(r_returns)

    if verbose:
        print(f"\nRunning {n_simulations:,} Monte Carlo simulations...")
        print(f"Starting equity: ${starting_equity:,.0f}")
        print(f"Risk per trade: {risk_per_trade_pct:.1%}")
        print(f"Trades per simulation: {trades_per_sim}")

    max_drawdowns = []
    final_equities = []
    ruin_count = 0  # Equity drops below 20% of starting

    for i in range(n_simulations):
        # Bootstrap resample trades with replacement
        bootstrapped = np.random.choice(r_returns, size=trades_per_sim, replace=True)

        # Simulate equity curve
        equity_curve = simulate_equity_curve(
            bootstrapped, starting_equity, risk_per_trade_pct
        )

        # Calculate max drawdown
        max_dd = calculate_max_drawdown_pct(equity_curve)
        max_drawdowns.append(max_dd)

        # Track final equity
        final_equities.append(equity_curve[-1])

        # Check for ruin (equity < 20% of starting)
        if np.min(equity_curve) < starting_equity * 0.2:
            ruin_count += 1

        if verbose and (i + 1) % 2000 == 0:
            print(f"  Completed {i + 1:,}/{n_simulations:,} simulations...")

    max_drawdowns = np.array(max_drawdowns)
    final_equities = np.array(final_equities)

    # Calculate statistics
    stats = {
        "n_simulations": n_simulations,
        "trades_per_sim": trades_per_sim,
        "starting_equity": starting_equity,
        "risk_per_trade_pct": risk_per_trade_pct,
        "max_drawdown": {
            "mean": float(np.mean(max_drawdowns)),
            "std": float(np.std(max_drawdowns)),
            "median": float(np.median(max_drawdowns)),
            "p50": float(np.percentile(max_drawdowns, 50)),
            "p75": float(np.percentile(max_drawdowns, 75)),
            "p90": float(np.percentile(max_drawdowns, 90)),
            "p95": float(np.percentile(max_drawdowns, 95)),
            "p99": float(np.percentile(max_drawdowns, 99)),
            "worst": float(np.max(max_drawdowns)),
        },
        "final_equity": {
            "mean": float(np.mean(final_equities)),
            "median": float(np.median(final_equities)),
            "p5": float(np.percentile(final_equities, 5)),
            "p95": float(np.percentile(final_equities, 95)),
            "worst": float(np.min(final_equities)),
            "best": float(np.max(final_equities)),
        },
        "risk_of_ruin_pct": 100.0 * ruin_count / n_simulations,
    }

    return max_drawdowns, stats


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo drawdown analysis")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--account-size', type=float, default=10000, help='Starting account size')
    parser.add_argument('--risk-pct', type=float, default=1.0, help='Risk per trade (%)')
    parser.add_argument('--threshold', type=float, default=0.20, help='Max acceptable drawdown at 95% CI')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    risk_per_trade = args.risk_pct / 100.0

    print("=" * 70)
    print("PHASE 9.5: MONTE CARLO DRAWDOWN ANALYSIS")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Simulations: {args.simulations:,}")
    print(f"Starting equity: ${args.account_size:,.0f}")
    print(f"Risk per trade: {args.risk_pct:.1f}%")
    print(f"Threshold: {args.threshold:.0%} max drawdown at 95% CI")

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

    # Run Monte Carlo simulation
    print(f"\n{'=' * 70}")
    print("MONTE CARLO SIMULATION")
    print(f"{'=' * 70}")

    r_returns = get_r_returns(trades)
    max_drawdowns, stats = run_monte_carlo(
        r_returns,
        n_simulations=args.simulations,
        starting_equity=args.account_size,
        risk_per_trade_pct=risk_per_trade,
        seed=args.seed,
        verbose=True,
    )

    # Results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    dd_stats = stats['max_drawdown']
    eq_stats = stats['final_equity']

    print(f"\nMax Drawdown Distribution:")
    print(f"  Mean:   {dd_stats['mean']:.1%}")
    print(f"  Median: {dd_stats['median']:.1%}")
    print(f"  75th:   {dd_stats['p75']:.1%}")
    print(f"  90th:   {dd_stats['p90']:.1%}")
    print(f"  95th:   {dd_stats['p95']:.1%}")
    print(f"  99th:   {dd_stats['p99']:.1%}")
    print(f"  Worst:  {dd_stats['worst']:.1%}")

    print(f"\nFinal Equity Distribution (starting ${args.account_size:,.0f}):")
    print(f"  Mean:   ${eq_stats['mean']:,.0f}")
    print(f"  Median: ${eq_stats['median']:,.0f}")
    print(f"  5th:    ${eq_stats['p5']:,.0f}")
    print(f"  95th:   ${eq_stats['p95']:,.0f}")
    print(f"  Worst:  ${eq_stats['worst']:,.0f}")
    print(f"  Best:   ${eq_stats['best']:,.0f}")

    print(f"\nRisk of Ruin (<20% of starting): {stats['risk_of_ruin_pct']:.2f}%")

    # Verdict
    p95_drawdown = dd_stats['p95']
    passed = p95_drawdown < args.threshold

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if passed:
        print(f"\n✅ PASS: Expected max drawdown at 95% CI ({p95_drawdown:.1%}) < {args.threshold:.0%}")
        print(f"   Median drawdown: {dd_stats['median']:.1%}")
        print(f"   Risk of ruin: {stats['risk_of_ruin_pct']:.2f}%")
        print(f"   The strategy has acceptable risk characteristics")
    else:
        print(f"\n❌ FAIL: Expected max drawdown at 95% CI ({p95_drawdown:.1%}) >= {args.threshold:.0%}")
        print(f"   Median drawdown: {dd_stats['median']:.1%}")
        print(f"   Risk of ruin: {stats['risk_of_ruin_pct']:.2f}%")
        print(f"   Consider reducing position size or improving risk management")

    # Recommendations based on drawdown
    print(f"\nRisk Recommendations:")
    if p95_drawdown > 0.30:
        print(f"  ⚠️  High risk at {args.risk_pct}% per trade. Consider 0.5%")
    elif p95_drawdown > 0.20:
        print(f"  ⚠️  Moderate risk. Consider 0.75% per trade")
    else:
        print(f"  ✓  {args.risk_pct}% per trade is appropriate")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"monte_carlo_drawdown_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "monte_carlo_drawdown",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "simulations": args.simulations,
            "account_size": args.account_size,
            "risk_per_trade_pct": risk_per_trade,
            "threshold": args.threshold,
            "seed": args.seed,
        },
        "baseline_metrics": {
            "total_trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "profit_factor": metrics['profit_factor'],
            "expectancy": metrics['expectancy'],
        },
        "monte_carlo_stats": stats,
        "p95_drawdown": p95_drawdown,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="monte_carlo_drawdown",
        passed=passed,
        metric_value=p95_drawdown,
        threshold=args.threshold,
        details=results,
    )


if __name__ == "__main__":
    main()
