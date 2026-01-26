#!/usr/bin/env python3
"""
Walk-Forward Validation of Best Parameters
==========================================
Validates the best parameters from Phase 7.9 optimization
across multiple time periods to confirm edge stability.

Usage:
    python scripts/validate_best_params.py
    python scripts/validate_best_params.py --params-file results/phase77_optimization/profit_factor_penalized/final_results.json
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.extended_runner import (
    ExtendedDetectionRunner,
    ExtendedRunnerConfig,
    WalkForwardConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Validate best parameters with walk-forward testing")
    parser.add_argument(
        '--params-file', '-p',
        default='results/phase77_optimization/profit_factor_penalized/final_results.json',
        help='Path to results JSON with best_params'
    )
    parser.add_argument(
        '--symbol', '-s',
        default='BTCUSDT',
        help='Symbol to validate on (default: BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe', '-t',
        default='4h',
        help='Timeframe (default: 4h)'
    )
    parser.add_argument(
        '--n-folds', '-f',
        type=int,
        default=5,
        help='Number of walk-forward folds (default: 5)'
    )
    args = parser.parse_args()

    # Load best parameters
    params_path = PROJECT_ROOT / args.params_file
    if not params_path.exists():
        print(f"ERROR: Parameters file not found: {params_path}")
        sys.exit(1)

    with open(params_path) as f:
        data = json.load(f)

    best_params = data['best_params']
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"\nLoaded parameters from: {args.params_file}")
    print(f"Original score: {data.get('best_score', 'N/A')}")
    print(f"Original PF: {data['best_simulation'].get('profit_factor', 'N/A'):.4f}")
    print(f"Original Sharpe: {data['best_simulation'].get('sharpe_ratio', 'N/A'):.4f}")

    # Configure walk-forward
    wf_config = WalkForwardConfig(
        n_folds=args.n_folds,
        window_type='rolling',
        train_ratio=0.7,
        purge_bars=50,
        embargo_bars=20,
    )

    config = ExtendedRunnerConfig(
        walk_forward=wf_config,
        run_walk_forward=True,
    )

    # Create runner
    print(f"\nRunning {args.n_folds}-fold walk-forward on {args.symbol} {args.timeframe}...")
    runner = ExtendedDetectionRunner(
        symbols=[args.symbol],
        timeframes=[args.timeframe],
        config=config,
        n_jobs=1,  # Single thread for validation
    )

    # Run walk-forward
    wf_result = runner.run_walk_forward(
        params=best_params,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    # Print results
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)

    # Fold-by-fold results
    print("\n--- Fold-by-Fold Performance ---")
    print(f"{'Fold':<6} {'Train PF':<12} {'Test PF':<12} {'Train WR':<12} {'Test WR':<12}")
    print("-" * 54)

    for fold in wf_result.folds:
        train_pf = fold.train_sim_result.profit_factor if fold.train_sim_result else 0
        test_pf = fold.test_sim_result.profit_factor if fold.test_sim_result else 0
        train_wr = fold.train_sim_result.win_rate if fold.train_sim_result else 0
        test_wr = fold.test_sim_result.win_rate if fold.test_sim_result else 0

        print(f"{fold.fold_idx:<6} {train_pf:<12.4f} {test_pf:<12.4f} {train_wr*100:<12.1f}% {test_wr*100:<12.1f}%")

    # Aggregate metrics
    print("\n--- Aggregate Metrics ---")
    print(f"Train Patterns: {wf_result.train_aggregate.total_patterns if wf_result.train_aggregate else 0}")
    print(f"Test Patterns: {wf_result.test_aggregate.total_patterns if wf_result.test_aggregate else 0}")

    if wf_result.train_sim_aggregate and wf_result.test_sim_aggregate:
        train_pf = wf_result.train_sim_aggregate.profit_factor
        test_pf = wf_result.test_sim_aggregate.profit_factor

        print(f"\nTrain Profit Factor: {train_pf:.4f}")
        print(f"Test Profit Factor: {test_pf:.4f}")

        # Walk-forward efficiency
        if train_pf > 0:
            efficiency = test_pf / train_pf
            print(f"\n>>> Walk-Forward Efficiency: {efficiency*100:.1f}%")

            if efficiency > 0.5:
                print(">>> STATUS: PASS - Edge generalizes across time periods")
            elif efficiency > 0.3:
                print(">>> STATUS: MARGINAL - Some degradation, proceed with caution")
            else:
                print(">>> STATUS: FAIL - Significant overfitting detected")
        else:
            print("\n>>> Cannot calculate efficiency (train PF <= 0)")
    else:
        print("\nNo simulation results available")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
