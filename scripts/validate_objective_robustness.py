#!/usr/bin/env python3
"""
Phase 8.0: Walk-Forward Objective Robustness Validation
=======================================================
Validates that the profit_factor_penalized objective consistently
finds profitable parameters across DIFFERENT time periods.

This tests the OBJECTIVE FUNCTION, not just a single parameter set.

Success Criteria:
- PF > 1.1 in at least 4 of 5 folds
- No fold has PF < 0.95 (catastrophic failure)
- Walk-Forward Efficiency > 50%

If validation fails: Stop and investigate. Do not proceed to ML.

Usage:
    python scripts/validate_objective_robustness.py
    python scripts/validate_objective_robustness.py --n-folds 5 --iterations-per-fold 30
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not installed.")

from src.optimization.extended_runner import (
    ExtendedDetectionRunner,
    ExtendedRunnerConfig,
    WalkForwardConfig,
    ALL_CLUSTERED_SYMBOLS,
)
from src.optimization.trade_simulator import TradeManagementConfig
from src.optimization.objectives import ObjectiveType, create_objective
from src.data_engine import load_master_data


@dataclass
class FoldResult:
    """Result from a single time-period fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Optimization result
    best_score: float
    best_params: Dict[str, Any]

    # In-sample metrics (training period)
    is_profit_factor: float
    is_sharpe: float
    is_win_rate: float
    is_pattern_count: int

    # Out-of-sample metrics (test period)
    oos_profit_factor: float
    oos_sharpe: float
    oos_win_rate: float
    oos_pattern_count: int
    oos_expectancy_r: float

    # Walk-forward efficiency for this fold
    wf_efficiency: float


@dataclass
class ObjectiveValidationResult:
    """Result of walk-forward objective validation."""
    objective_type: str
    n_folds: int
    iterations_per_fold: int

    # Per-fold results
    fold_results: List[FoldResult]

    # Aggregate metrics
    folds_profitable: int  # PF > 1.0
    folds_above_threshold: int  # PF > 1.1
    folds_catastrophic: int  # PF < 0.95

    mean_oos_pf: float
    std_oos_pf: float
    mean_oos_sharpe: float
    mean_wf_efficiency: float

    # Validation decision
    validation_passed: bool
    failure_reasons: List[str]

    # Timing
    total_duration_hours: float


# Simplified parameter space for faster fold optimization
PARAM_SPACE_VALIDATION = [
    # Core detection params (8)
    Integer(3, 8, name='min_bar_separation'),
    Real(0.5, 1.5, name='min_move_atr'),
    Real(0.2, 0.4, name='forward_confirm_pct'),
    Integer(4, 7, name='lookback'),
    Integer(5, 8, name='lookforward'),
    Real(0.3, 1.0, name='p3_min_extension_atr'),
    Real(3.0, 8.0, name='p3_max_extension_atr'),
    Integer(10, 20, name='min_pattern_bars'),

    # Trade management params (5)
    Real(1.0, 2.0, name='sl_atr_mult'),
    Real(2.0, 5.0, name='tp_atr_mult'),
    Integer(30, 100, name='max_bars_held'),
    Real(1.5, 3.5, name='min_risk_reward'),
    Real(0.6, 0.9, name='min_r_squared'),
]

PARAM_NAMES = [p.name for p in PARAM_SPACE_VALIDATION]


def create_time_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_ratio: float = 0.6,
    purge_bars: int = 50,
) -> List[Tuple[int, int, int, int]]:
    """
    Create time-based folds for walk-forward validation.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    Each fold tests on a DIFFERENT time period.
    """
    n_bars = len(df)
    fold_size = n_bars // (n_folds + 1)  # +1 because we need test after train

    folds = []
    for i in range(n_folds):
        # Training period: expanding window
        train_start = 0
        train_end = (i + 1) * fold_size

        # Purge gap
        test_start = train_end + purge_bars
        test_end = min(test_start + fold_size, n_bars)

        if test_end - test_start < 100:  # Minimum test size
            continue

        folds.append((train_start, train_end, test_start, test_end))

    return folds


def run_fold_optimization(
    fold_idx: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    objective_type: str,
    iterations: int,
    symbols: List[str],
    timeframe: str,
    n_jobs: int = 2,
) -> FoldResult:
    """
    Run optimization on training period, evaluate on test period.
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}: Optimizing on {len(train_df)} bars, testing on {len(test_df)} bars")
    print(f"{'='*60}")

    train_start_time = train_df.index[0] if hasattr(train_df.index, '__getitem__') else train_df['time'].iloc[0]
    train_end_time = train_df.index[-1] if hasattr(train_df.index, '__getitem__') else train_df['time'].iloc[-1]
    test_start_time = test_df.index[0] if hasattr(test_df.index, '__getitem__') else test_df['time'].iloc[0]
    test_end_time = test_df.index[-1] if hasattr(test_df.index, '__getitem__') else test_df['time'].iloc[-1]

    # Create objective
    objective = create_objective(ObjectiveType(objective_type))

    # Track best result
    best_score = float('-inf')
    best_params = {}
    best_is_metrics = {}

    def objective_fn(param_values):
        nonlocal best_score, best_params, best_is_metrics

        params = dict(zip(PARAM_NAMES, param_values))

        # Add fixed defaults for params not in simplified space
        params.setdefault('p4_min_break_atr', 0.1)
        params.setdefault('p5_max_symmetry_atr', 4.0)
        params.setdefault('min_adx', 25.0)
        params.setdefault('min_trend_move_atr', 3.0)
        params.setdefault('min_trend_swings', 3)
        params.setdefault('head_extension_weight', 0.20)
        params.setdefault('bos_efficiency_weight', 0.18)
        params.setdefault('volume_spike_weight', 0.10)
        params.setdefault('path_efficiency_weight', 0.10)
        params.setdefault('trend_strength_weight', 0.08)
        params.setdefault('regime_suitability_weight', 0.08)
        params.setdefault('entry_buffer_atr', 0.0)
        params.setdefault('trailing_activation_atr', 0.0)
        params.setdefault('trailing_step_atr', 0.5)
        params.setdefault('slippage_pct', 0.02)
        params.setdefault('commission_pct', 0.05)

        try:
            # Create runner for training data
            config = ExtendedRunnerConfig()
            runner = ExtendedDetectionRunner(
                symbols=symbols,
                timeframes=[timeframe],
                config=config,
                n_jobs=n_jobs,
            )

            # Inject training data into cache
            for symbol in symbols:
                cache_key = f"{symbol}_{timeframe}"
                runner._data_cache[cache_key] = train_df.copy()

            # Run detection + simulation
            result = runner.run_single_evaluation(params)

            if result is None:
                return 1e9

            detection_result, sim_result = result

            # Evaluate with objective function
            obj_result = objective.evaluate(sim_result, detection_result)

            if not obj_result.is_valid:
                return 1e9

            score = obj_result.score

            # Track best
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_is_metrics = {
                    'profit_factor': sim_result.profit_factor,
                    'sharpe': sim_result.sharpe_ratio,
                    'win_rate': sim_result.win_rate,
                    'pattern_count': detection_result.get('total_patterns', 0),
                    'expectancy_r': sim_result.expectancy_r,
                }

            return -score  # Minimize negative score

        except Exception as e:
            print(f"  Error: {e}")
            return 1e9

    # Run Bayesian optimization
    print(f"  Running {iterations} optimization iterations...")
    start_time = time.time()

    result = gp_minimize(
        objective_fn,
        PARAM_SPACE_VALIDATION,
        n_calls=iterations,
        n_initial_points=min(10, iterations // 3),
        random_state=42 + fold_idx,
        verbose=False,
    )

    opt_duration = (time.time() - start_time) / 60
    print(f"  Optimization complete in {opt_duration:.1f} minutes")
    print(f"  Best IS score: {best_score:.4f}")
    print(f"  Best IS PF: {best_is_metrics.get('profit_factor', 0):.4f}")

    # Evaluate best params on TEST data (out-of-sample)
    print(f"\n  Evaluating on out-of-sample test period...")

    config = ExtendedRunnerConfig()
    runner = ExtendedDetectionRunner(
        symbols=symbols,
        timeframes=[timeframe],
        config=config,
        n_jobs=n_jobs,
    )

    # Inject TEST data into cache
    for symbol in symbols:
        cache_key = f"{symbol}_{timeframe}"
        runner._data_cache[cache_key] = test_df.copy()

    test_result = runner.run_single_evaluation(best_params)

    if test_result is None:
        oos_metrics = {
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'pattern_count': 0,
            'expectancy_r': 0.0,
        }
    else:
        detection_result, sim_result = test_result
        oos_metrics = {
            'profit_factor': sim_result.profit_factor,
            'sharpe': sim_result.sharpe_ratio,
            'win_rate': sim_result.win_rate,
            'pattern_count': detection_result.get('total_patterns', 0),
            'expectancy_r': sim_result.expectancy_r,
        }

    # Calculate walk-forward efficiency
    is_pf = best_is_metrics.get('profit_factor', 0)
    oos_pf = oos_metrics['profit_factor']
    wf_efficiency = oos_pf / is_pf if is_pf > 0 else 0

    print(f"  OOS Profit Factor: {oos_pf:.4f}")
    print(f"  OOS Sharpe: {oos_metrics['sharpe']:.4f}")
    print(f"  OOS Patterns: {oos_metrics['pattern_count']}")
    print(f"  Walk-Forward Efficiency: {wf_efficiency:.1%}")

    return FoldResult(
        fold_idx=fold_idx,
        train_start=str(train_start_time),
        train_end=str(train_end_time),
        test_start=str(test_start_time),
        test_end=str(test_end_time),
        best_score=best_score,
        best_params=best_params,
        is_profit_factor=is_pf,
        is_sharpe=best_is_metrics.get('sharpe', 0),
        is_win_rate=best_is_metrics.get('win_rate', 0),
        is_pattern_count=best_is_metrics.get('pattern_count', 0),
        oos_profit_factor=oos_pf,
        oos_sharpe=oos_metrics['sharpe'],
        oos_win_rate=oos_metrics['win_rate'],
        oos_pattern_count=oos_metrics['pattern_count'],
        oos_expectancy_r=oos_metrics['expectancy_r'],
        wf_efficiency=wf_efficiency,
    )


def run_objective_validation(
    objective_type: str = 'profit_factor_penalized',
    n_folds: int = 5,
    iterations_per_fold: int = 30,
    symbols: List[str] = None,
    timeframe: str = '4h',
    n_jobs: int = 2,
) -> ObjectiveValidationResult:
    """
    Run complete walk-forward validation of an objective function.
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize required")

    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD OBJECTIVE VALIDATION")
    print(f"{'='*60}")
    print(f"Objective: {objective_type}")
    print(f"Folds: {n_folds}")
    print(f"Iterations per fold: {iterations_per_fold}")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {timeframe}")

    start_time = time.time()

    # Load data for primary symbol (BTCUSDT) to create time folds
    print(f"\nLoading data...")
    primary_df = load_master_data(timeframe, symbol='BTCUSDT')
    primary_df.columns = [c.lower() for c in primary_df.columns]

    # Create time-based folds
    folds = create_time_folds(primary_df, n_folds=n_folds)
    print(f"Created {len(folds)} time folds")

    # Run each fold
    fold_results = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        train_df = primary_df.iloc[train_start:train_end].copy()
        test_df = primary_df.iloc[test_start:test_end].copy()

        fold_result = run_fold_optimization(
            fold_idx=fold_idx,
            train_df=train_df,
            test_df=test_df,
            objective_type=objective_type,
            iterations=iterations_per_fold,
            symbols=symbols,
            timeframe=timeframe,
            n_jobs=n_jobs,
        )
        fold_results.append(fold_result)

    # Calculate aggregate metrics
    oos_pfs = [f.oos_profit_factor for f in fold_results]
    wf_effs = [f.wf_efficiency for f in fold_results]

    folds_profitable = sum(1 for pf in oos_pfs if pf > 1.0)
    folds_above_threshold = sum(1 for pf in oos_pfs if pf > 1.1)
    folds_catastrophic = sum(1 for pf in oos_pfs if pf < 0.95)

    mean_oos_pf = np.mean(oos_pfs)
    std_oos_pf = np.std(oos_pfs)
    mean_oos_sharpe = np.mean([f.oos_sharpe for f in fold_results])
    mean_wf_efficiency = np.mean(wf_effs)

    # Validation decision
    failure_reasons = []

    if folds_above_threshold < 4:
        failure_reasons.append(f"Only {folds_above_threshold}/{n_folds} folds have PF > 1.1 (need 4+)")

    if folds_catastrophic > 0:
        failure_reasons.append(f"{folds_catastrophic} fold(s) have catastrophic PF < 0.95")

    if mean_wf_efficiency < 0.5:
        failure_reasons.append(f"Walk-Forward Efficiency {mean_wf_efficiency:.1%} < 50%")

    validation_passed = len(failure_reasons) == 0

    total_duration = (time.time() - start_time) / 3600

    result = ObjectiveValidationResult(
        objective_type=objective_type,
        n_folds=n_folds,
        iterations_per_fold=iterations_per_fold,
        fold_results=fold_results,
        folds_profitable=folds_profitable,
        folds_above_threshold=folds_above_threshold,
        folds_catastrophic=folds_catastrophic,
        mean_oos_pf=mean_oos_pf,
        std_oos_pf=std_oos_pf,
        mean_oos_sharpe=mean_oos_sharpe,
        mean_wf_efficiency=mean_wf_efficiency,
        validation_passed=validation_passed,
        failure_reasons=failure_reasons,
        total_duration_hours=total_duration,
    )

    # Print summary
    print_validation_summary(result)

    return result


def print_validation_summary(result: ObjectiveValidationResult):
    """Print validation summary."""
    print(f"\n{'='*60}")
    print(f"OBJECTIVE VALIDATION SUMMARY")
    print(f"{'='*60}")

    print(f"\nObjective: {result.objective_type}")
    print(f"Duration: {result.total_duration_hours:.2f} hours")

    print(f"\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'IS PF':<10} {'OOS PF':<10} {'WF Eff':<10} {'OOS Patterns':<12}")
    print("-" * 48)
    for fold in result.fold_results:
        print(f"{fold.fold_idx + 1:<6} {fold.is_profit_factor:<10.4f} {fold.oos_profit_factor:<10.4f} {fold.wf_efficiency:<10.1%} {fold.oos_pattern_count:<12}")

    print(f"\n--- Aggregate Metrics ---")
    print(f"Mean OOS Profit Factor: {result.mean_oos_pf:.4f} ± {result.std_oos_pf:.4f}")
    print(f"Mean OOS Sharpe: {result.mean_oos_sharpe:.4f}")
    print(f"Mean Walk-Forward Efficiency: {result.mean_wf_efficiency:.1%}")

    print(f"\n--- Validation Criteria ---")
    print(f"Folds with PF > 1.0: {result.folds_profitable}/{result.n_folds}")
    print(f"Folds with PF > 1.1: {result.folds_above_threshold}/{result.n_folds} (need 4+)")
    print(f"Folds with PF < 0.95: {result.folds_catastrophic}/{result.n_folds} (need 0)")
    print(f"Walk-Forward Efficiency: {result.mean_wf_efficiency:.1%} (need > 50%)")

    print(f"\n{'='*60}")
    if result.validation_passed:
        print(f">>> VALIDATION PASSED <<<")
        print(f"The {result.objective_type} objective is ROBUST across time periods.")
        print(f"Safe to proceed with ML meta-labeling.")
    else:
        print(f">>> VALIDATION FAILED <<<")
        print(f"Failure reasons:")
        for reason in result.failure_reasons:
            print(f"  - {reason}")
        print(f"\nDO NOT proceed with ML. Investigate the baseline first.")
    print(f"{'='*60}\n")


def save_results(result: ObjectiveValidationResult, output_dir: Path):
    """Save validation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    result_dict = {
        'objective_type': result.objective_type,
        'n_folds': result.n_folds,
        'iterations_per_fold': result.iterations_per_fold,
        'validation_passed': result.validation_passed,
        'failure_reasons': result.failure_reasons,
        'mean_oos_pf': result.mean_oos_pf,
        'std_oos_pf': result.std_oos_pf,
        'mean_oos_sharpe': result.mean_oos_sharpe,
        'mean_wf_efficiency': result.mean_wf_efficiency,
        'folds_profitable': result.folds_profitable,
        'folds_above_threshold': result.folds_above_threshold,
        'folds_catastrophic': result.folds_catastrophic,
        'total_duration_hours': result.total_duration_hours,
        'fold_results': [asdict(f) for f in result.fold_results],
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / 'objective_validation.json'
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Objective Validation')
    parser.add_argument('--objective', '-o', default='profit_factor_penalized',
                        help='Objective to validate')
    parser.add_argument('--n-folds', '-f', type=int, default=5,
                        help='Number of time folds')
    parser.add_argument('--iterations', '-n', type=int, default=30,
                        help='Optimization iterations per fold')
    parser.add_argument('--symbols', '-s', default='BTCUSDT,ETHUSDT,BNBUSDT',
                        help='Comma-separated symbols')
    parser.add_argument('--timeframe', '-t', default='4h',
                        help='Timeframe')
    parser.add_argument('--n-jobs', '-j', type=int, default=2,
                        help='Parallel jobs')
    parser.add_argument('--output', default='results/phase80_validation',
                        help='Output directory')

    args = parser.parse_args()

    symbols = args.symbols.split(',')

    result = run_objective_validation(
        objective_type=args.objective,
        n_folds=args.n_folds,
        iterations_per_fold=args.iterations,
        symbols=symbols,
        timeframe=args.timeframe,
        n_jobs=args.n_jobs,
    )

    save_results(result, Path(args.output))

    # Exit with status code
    sys.exit(0 if result.validation_passed else 1)


if __name__ == '__main__':
    main()
