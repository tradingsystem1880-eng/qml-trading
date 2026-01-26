#!/usr/bin/env python3
"""
Phase 7.7: Comprehensive Extended Optimization
==============================================
Main script for running extended optimization with:

1. 25 parameters (15 detection + 10 trade management)
2. 6 objective functions (sequential)
3. Walk-forward validation (5 folds)
4. Symbol-cluster validation (7 clusters)
5. Trade simulation with MAE/MFE tracking

Usage:
    # Run single objective
    python scripts/run_phase77_optimization.py --objective composite --iterations 500

    # Run all objectives sequentially
    python scripts/run_phase77_optimization.py --all-objectives --iterations 500

    # Quick test run
    python scripts/run_phase77_optimization.py --quick --iterations 20

    # Run with walk-forward validation
    python scripts/run_phase77_optimization.py --objective sharpe --iterations 500 --walk-forward

Expected Runtime: 40-100+ hours for full optimization (3000 total iterations)
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not installed. Install with: pip install scikit-optimize>=0.9.0")

from src.optimization.extended_runner import (
    ExtendedDetectionRunner,
    ExtendedRunnerConfig,
    WalkForwardConfig,
    SYMBOL_CLUSTERS,
    ALL_CLUSTERED_SYMBOLS,
)
from src.optimization.trade_simulator import TradeManagementConfig
from src.optimization.objectives import (
    ObjectiveType,
    ObjectiveConfig,
    create_objective,
    get_all_objective_types,
)
from src.validation.statistical_tests import (
    validate_strategy,
    print_validation_report,
)


# =============================================================================
# EXTENDED PARAMETER SPACE (25 parameters)
# =============================================================================

PARAM_SPACE_PHASE77 = [
    # === DETECTION PARAMETERS (15) - from Phase 7.6 ===

    # Swing Detection (5 params)
    Integer(3, 10, name='min_bar_separation'),
    Real(0.5, 2.0, name='min_move_atr'),
    Real(0.2, 0.5, name='forward_confirm_pct'),
    Integer(3, 8, name='lookback'),
    Integer(3, 8, name='lookforward'),

    # Pattern Validation (5 params)
    Real(0.3, 1.5, name='p3_min_extension_atr'),
    Real(5.0, 15.0, name='p3_max_extension_atr'),
    Real(0.05, 0.3, name='p4_min_break_atr'),
    Real(2.0, 6.0, name='p5_max_symmetry_atr'),
    Integer(8, 25, name='min_pattern_bars'),

    # Trend Validation (3 params)
    Real(15.0, 30.0, name='min_adx'),
    Real(2.0, 5.0, name='min_trend_move_atr'),
    Integer(2, 5, name='min_trend_swings'),

    # Scoring Weights (6 params) - Phase 7.8: now optimizable
    # Note: shoulder_weight=0.12 is fixed, swing_weight is auto-calculated (min 0.02)
    # Max sum: 0.22 + 0.18 + 0.10*4 = 0.80, + shoulder 0.12 = 0.92, leaves 0.08 for swing
    Real(0.15, 0.22, name='head_extension_weight'),    # Core geometry
    Real(0.12, 0.18, name='bos_efficiency_weight'),    # Core geometry
    Real(0.05, 0.10, name='volume_spike_weight'),      # Phase 7.6
    Real(0.05, 0.10, name='path_efficiency_weight'),   # Phase 7.6
    Real(0.05, 0.10, name='trend_strength_weight'),    # Phase 7.6
    Real(0.05, 0.10, name='regime_suitability_weight'), # Phase 7.8

    # === TRADE MANAGEMENT PARAMETERS (10) - NEW in Phase 7.7 ===

    # Entry
    Real(0.0, 0.3, name='entry_buffer_atr'),

    # Stop Loss / Take Profit
    Real(1.0, 3.0, name='sl_atr_mult'),
    Real(2.0, 6.0, name='tp_atr_mult'),

    # Trailing Stop
    Real(0.0, 2.0, name='trailing_activation_atr'),  # 0 = disabled
    Real(0.3, 1.0, name='trailing_step_atr'),

    # Time Exit
    Integer(0, 100, name='max_bars_held'),  # 0 = disabled

    # Risk Management
    Real(1.0, 3.0, name='min_risk_reward'),

    # Transaction Costs
    Real(0.01, 0.1, name='slippage_pct'),
    Real(0.05, 0.15, name='commission_pct'),

    # Trend Regime (new R² parameter)
    Real(0.4, 0.8, name='min_r_squared'),
]

PARAM_NAMES = [p.name for p in PARAM_SPACE_PHASE77] if SKOPT_AVAILABLE else []


# =============================================================================
# OPTIMIZATION CONFIG
# =============================================================================

class Phase77OptimizationConfig:
    """Configuration for Phase 7.7 optimization."""

    def __init__(
        self,
        objective_type: ObjectiveType = ObjectiveType.COMPOSITE,
        n_calls: int = 500,
        n_initial_points: int = 50,
        random_state: int = 42,
        run_walk_forward: bool = False,
        run_cluster_validation: bool = False,
        timeframes: List[str] = None,
        checkpoint_every: int = 25,
        verbose: bool = True,
        n_jobs: int = 2,  # Reduced from -1 (all cores) for lower CPU usage
        throttle_seconds: float = 1.0,  # Pause between iterations for thermal management
    ):
        self.objective_type = objective_type
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.run_walk_forward = run_walk_forward
        self.run_cluster_validation = run_cluster_validation
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.throttle_seconds = throttle_seconds

        # Output directory
        self.output_dir = PROJECT_ROOT / "results" / "phase77_optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OPTIMIZER
# =============================================================================

class Phase77Optimizer:
    """
    Phase 7.7 Bayesian Optimizer.

    Extends Phase 7.6 with:
    - Trade simulation in objective
    - Multiple objective functions
    - Walk-forward validation
    - Symbol-cluster validation
    """

    def __init__(self, config: Phase77OptimizationConfig):
        """Initialize the optimizer."""
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required. Install with: pip install scikit-optimize>=0.9.0"
            )

        self.config = config

        # Initialize runner with reduced parallelism for lower CPU usage
        runner_config = ExtendedRunnerConfig(
            timeframes=config.timeframes,
            run_walk_forward=config.run_walk_forward,
            run_cluster_validation=config.run_cluster_validation,
        )
        self.runner = ExtendedDetectionRunner(
            symbols=ALL_CLUSTERED_SYMBOLS,
            timeframes=config.timeframes,
            config=runner_config,
            n_jobs=config.n_jobs,  # Use limited parallelism
        )
        self.runner.set_objective(config.objective_type)

        # Tracking
        self._iteration = 0
        self._best_score = float('inf')
        self._best_params = None
        self._best_results = None
        self._history: List[Dict] = []

        # Timing
        self._start_time = None

    def objective(self, params_list: List[Any]) -> float:
        """
        Objective function for optimization.

        Args:
            params_list: Parameter values in PARAM_SPACE_PHASE77 order

        Returns:
            Score to minimize (negative of quality metric)
        """
        self._iteration += 1

        # Convert list to dict
        params = dict(zip(PARAM_NAMES, params_list))

        try:
            # Run extended detection + simulation
            obj_result, detection_result, sim_result = self.runner.run_with_dict_extended(
                params, self.config.objective_type
            )

            score = obj_result.score

            # Track history
            self._history.append({
                'iteration': self._iteration,
                'params': params,
                'score': float(-score),  # Store positive score
                'raw_value': obj_result.raw_value,
                'is_valid': obj_result.is_valid,
                'penalty_reason': obj_result.penalty_reason,
                'total_patterns': detection_result.total_patterns,
                'mean_quality': detection_result.mean_score,
                'unique_symbols': detection_result.unique_symbols,
                'total_trades': sim_result.total_trades,
                'win_rate': sim_result.win_rate,
                'sharpe': sim_result.sharpe_ratio,
                'expectancy': sim_result.expectancy_r,
                'profit_factor': sim_result.profit_factor,
                'max_drawdown': sim_result.max_drawdown_r,
            })

            # Update best
            if score < self._best_score:
                self._best_score = score
                self._best_params = params
                self._best_results = {
                    'objective': obj_result,
                    'detection': detection_result,
                    'simulation': sim_result,
                }

            # Log progress
            if self.config.verbose:
                self._log_iteration(params, obj_result, detection_result, sim_result)

            # Checkpoint
            if self._iteration % self.config.checkpoint_every == 0:
                self._save_checkpoint()

            # Throttle to reduce CPU load and allow cooling
            if self.config.throttle_seconds > 0:
                time.sleep(self.config.throttle_seconds)

            return score

        except Exception as e:
            print(f"  ERROR: {e}")
            return 1000.0  # High penalty for errors

    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization.

        Returns:
            Dictionary with best parameters and results
        """
        print("=" * 70)
        print(f"PHASE 7.7 OPTIMIZATION - {self.config.objective_type.value.upper()}")
        print("=" * 70)
        print(f"Parameters: {len(PARAM_SPACE_PHASE77)}")
        print(f"Iterations: {self.config.n_calls}")
        print(f"Initial points: {self.config.n_initial_points}")
        print(f"Timeframes: {self.config.timeframes}")
        print(f"Symbols: {len(ALL_CLUSTERED_SYMBOLS)}")
        print("=" * 70)

        self._start_time = time.time()

        # Preload data
        print("\nPreloading data...")
        self.runner.preload_data()
        print(f"Loaded {len(self.runner._data_cache)} datasets")

        # Run optimization
        print("\nStarting optimization...\n")

        result = gp_minimize(
            func=self.objective,
            dimensions=PARAM_SPACE_PHASE77,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=False,
            n_jobs=1,
        )

        duration = time.time() - self._start_time

        # Final results
        final_results = self._build_final_results(result, duration)

        # Print summary
        self._print_summary(final_results)

        # Save final results
        self._save_final_results(final_results)

        return final_results

    def _log_iteration(
        self,
        params: Dict,
        obj_result,
        detection_result,
        sim_result,
    ):
        """Log progress for current iteration."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = (elapsed / self._iteration) * (self.config.n_calls - self._iteration) if self._iteration > 0 else 0

        print(f"\n[Iter {self._iteration}/{self.config.n_calls}] "
              f"Score: {-obj_result.score:.4f} | "
              f"Best: {-self._best_score:.4f}")

        print(f"  Detection: {detection_result.total_patterns} patterns | "
              f"{detection_result.unique_symbols} symbols | "
              f"Q={detection_result.mean_score:.3f}")

        print(f"  Trading: {sim_result.total_trades} trades | "
              f"WR={sim_result.win_rate:.1%} | "
              f"Sharpe={sim_result.sharpe_ratio:.2f} | "
              f"PF={sim_result.profit_factor:.2f}")

        print(f"  Time: {elapsed/60:.1f}m elapsed | ETA: {eta/60:.1f}m")

        if obj_result.penalty_reason:
            print(f"  PENALTY: {obj_result.penalty_reason}")

    def _save_checkpoint(self):
        """Save checkpoint of current state."""
        checkpoint_dir = self.config.output_dir / self.config.objective_type.value
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save best params
        if self._best_params:
            best_path = checkpoint_dir / "best_params.json"

            # Convert numpy types to native Python
            params_native = {}
            for k, v in self._best_params.items():
                if hasattr(v, 'item'):
                    params_native[k] = v.item()
                else:
                    params_native[k] = v

            with open(best_path, 'w') as f:
                json.dump({
                    'params': params_native,
                    'score': float(-self._best_score),
                    'iteration': self._iteration,
                    'objective': self.config.objective_type.value,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)

            print(f"  Checkpoint saved: {best_path}")

    def _build_final_results(self, skopt_result, duration: float) -> Dict[str, Any]:
        """Build final results dictionary."""
        # Convert numpy types
        def to_native(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        return {
            'best_params': to_native(self._best_params),
            'best_score': float(-self._best_score),
            'objective_type': self.config.objective_type.value,
            'n_iterations': self._iteration,
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'best_detection': {
                'total_patterns': self._best_results['detection'].total_patterns,
                'unique_symbols': self._best_results['detection'].unique_symbols,
                'mean_quality': self._best_results['detection'].mean_score,
            } if self._best_results else {},
            'best_simulation': {
                'total_trades': self._best_results['simulation'].total_trades,
                'win_rate': self._best_results['simulation'].win_rate,
                'sharpe_ratio': self._best_results['simulation'].sharpe_ratio,
                'expectancy_r': self._best_results['simulation'].expectancy_r,
                'profit_factor': self._best_results['simulation'].profit_factor,
                'max_drawdown_r': self._best_results['simulation'].max_drawdown_r,
            } if self._best_results else {},
            'all_scores': [float(-s) for s in skopt_result.func_vals],
            'history': to_native(self._history),
        }

    def _print_summary(self, results: Dict[str, Any]):
        """Print optimization summary with statistical validation and decision logic."""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Objective: {results['objective_type']}")
        print(f"Iterations: {results['n_iterations']}")
        print(f"Duration: {results['duration_hours']:.1f} hours")
        print(f"\nBest Score: {results['best_score']:.4f}")

        if results.get('best_detection'):
            det = results['best_detection']
            print(f"\nBest Detection Results:")
            print(f"  Patterns: {det['total_patterns']}")
            print(f"  Symbols: {det['unique_symbols']}")
            print(f"  Quality: {det['mean_quality']:.3f}")

        sim = results.get('best_simulation', {})
        if sim:
            print(f"\nBest Trading Results:")
            print(f"  Trades: {sim['total_trades']}")
            print(f"  Win Rate: {sim['win_rate']:.1%}")
            print(f"  Sharpe: {sim['sharpe_ratio']:.2f}")
            print(f"  Expectancy: {sim['expectancy_r']:.3f}R")
            print(f"  Profit Factor: {sim['profit_factor']:.2f}")
            print(f"  Max Drawdown: {sim['max_drawdown_r']:.2f}R")

        print(f"\nBest Parameters:")
        if results.get('best_params'):
            for name, value in results['best_params'].items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

        # Phase 7.9: Statistical Validation
        if sim:
            metrics = {
                'sharpe': sim.get('sharpe_ratio', 0),
                'pattern_count': sim.get('total_trades', 0),
                'profit_factor': sim.get('profit_factor', 0),
                'win_rate': sim.get('win_rate', 0),
                'expectancy_r': sim.get('expectancy_r', 0),
                'avg_win_r': sim.get('avg_win_r', 0),
                'avg_mfe_r': sim.get('avg_mfe_r', 0),
            }

            validation = validate_strategy(
                metrics=metrics,
                num_trials=results['n_iterations'],
            )
            print_validation_report(validation)

            # Store validation in results
            results['validation'] = {
                'deflated_sharpe_ratio': validation.deflated_sharpe_ratio,
                'statistically_significant': validation.statistically_significant,
                'minimum_trades_needed': validation.minimum_trades_needed,
                'has_sufficient_trades': validation.has_sufficient_trades,
                'validation_summary': validation.validation_summary,
            }

            # Phase 7.9: Decision Logic
            decision = self._determine_next_action(metrics, validation)
            results['decision'] = decision

    def _determine_next_action(self, metrics: Dict, validation) -> Dict[str, Any]:
        """
        Determine the single most important next step based on results.

        This creates a closed loop - every optimization run produces a clear decision.
        Based on Phase 7.9 decision tree.
        """
        pf = metrics.get('profit_factor', 0)
        dsr = validation.deflated_sharpe_ratio
        expectancy = metrics.get('expectancy_r', 0)
        win_rate = metrics.get('win_rate', 0)
        avg_win_r = metrics.get('avg_win_r', 0)
        avg_mfe_r = metrics.get('avg_mfe_r', 0)

        # Decision tree
        if pf > 1.15 and dsr > 0.95:
            action = "PROCEED_TO_ML"
            message = "SUCCESS: Viable edge found with statistical confidence."
            next_step = "Proceed to ML meta-labeling phase. Use pattern quality for position sizing, not filtering."

        elif pf > 1.05 and dsr > 0.90:
            action = "EXTENDED_OPTIMIZATION"
            message = "PROMISING: Near-profitable with decent statistics."
            next_step = "Run full 500-iteration optimization with current objective. May need parameter space refinement."

        elif pf > 1.0 and expectancy > 0:
            action = "REFINE_PARAMETERS"
            message = "BREAKEVEN: Profitable but not statistically robust."
            next_step = "Analyze parameter sensitivity. Check if edge is concentrated in specific symbols/regimes."

        elif expectancy > 0.05 and pf < 1.0:
            action = "FIX_EXIT_LOGIC"
            message = "INSIGHT: Positive expectancy but poor profit factor suggests exit issues."
            next_step = "Focus EXCLUSIVELY on exit logic. Winners being cut short or losers running too long."

            # Additional diagnostic
            if avg_mfe_r > 0 and avg_win_r > 0:
                capture_ratio = avg_win_r / avg_mfe_r
                if capture_ratio < 0.5:
                    next_step += f" MFE capture ratio is {capture_ratio:.1%} - you're capturing less than half of winning moves."

        elif win_rate > 0.55 and pf < 0.9:
            action = "FIX_RR_RATIO"
            message = "PARADOX: High win rate but losing money = R:R problem."
            next_step = "Your avg loss is much larger than avg win. Either widen TP or tighten SL."

        else:
            action = "FUNDAMENTAL_REVIEW"
            message = "NO EDGE: Current approach not finding profitable patterns."
            next_step = "Consider: (1) Different pattern definition, (2) Different markets/timeframes, (3) Additional confluence filters."

        result = {
            'action': action,
            'message': message,
            'next_step': next_step,
            'metrics_summary': {
                'profit_factor': pf,
                'dsr': dsr,
                'expectancy_r': expectancy,
                'win_rate': win_rate,
            }
        }

        # Print decision
        print(f"\n{'='*60}")
        print(f"DECISION: {action}")
        print(f"{'='*60}")
        print(f"Assessment: {message}")
        print(f"Next Step: {next_step}")
        print(f"{'='*60}\n")

        return result

    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results to file."""
        output_dir = self.config.output_dir / self.config.objective_type.value
        output_dir.mkdir(parents=True, exist_ok=True)

        # Final results
        final_path = output_dir / "final_results.json"
        with open(final_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_objectives(
    iterations_per_objective: int = 500,
    quick: bool = False,
    n_jobs: int = 2,
    throttle_seconds: float = 1.0,
):
    """Run optimization for all objectives sequentially (Phase 7.9 - profit-focused first)."""
    objectives = [
        # Phase 7.9: Profit-focused objectives first (recommended)
        ObjectiveType.PROFIT_FACTOR_PENALIZED,
        ObjectiveType.EXPECTANCY_FOCUSED,
        # Original objectives
        ObjectiveType.COUNT_QUALITY,
        ObjectiveType.SHARPE,
        ObjectiveType.EXPECTANCY,
        ObjectiveType.PROFIT_FACTOR,
        ObjectiveType.MAX_DRAWDOWN,
        ObjectiveType.COMPOSITE,
    ]

    all_results = {}
    total_start = time.time()

    for i, obj_type in enumerate(objectives):
        print(f"\n{'#' * 70}")
        print(f"# OBJECTIVE {i+1}/{len(objectives)}: {obj_type.value.upper()}")
        print(f"{'#' * 70}\n")

        config = Phase77OptimizationConfig(
            objective_type=obj_type,
            n_calls=20 if quick else iterations_per_objective,
            n_initial_points=5 if quick else 50,
            n_jobs=n_jobs,
            throttle_seconds=throttle_seconds,
        )

        optimizer = Phase77Optimizer(config)
        results = optimizer.optimize()
        all_results[obj_type.value] = results

    total_duration = time.time() - total_start

    # Save combined results
    output_dir = PROJECT_ROOT / "results" / "phase77_optimization"
    combined_path = output_dir / "all_objectives_summary.json"

    summary = {
        'total_duration_hours': total_duration / 3600,
        'objectives_run': len(objectives),
        'iterations_per_objective': 20 if quick else iterations_per_objective,
        'results': {k: {
            'best_score': v['best_score'],
            'best_params': v['best_params'],
        } for k, v in all_results.items()},
    }

    with open(combined_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("ALL OBJECTIVES COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Results saved to: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7.7 Comprehensive Extended Optimization"
    )

    parser.add_argument(
        '--objective', '-o',
        type=str,
        choices=[t.value for t in ObjectiveType],
        default='composite',
        help='Objective function to optimize'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=500,
        help='Number of optimization iterations'
    )

    parser.add_argument(
        '--initial-points',
        type=int,
        default=50,
        help='Number of random initial points'
    )

    parser.add_argument(
        '--all-objectives',
        action='store_true',
        help='Run all 6 objectives sequentially'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (20 iterations)'
    )

    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Enable walk-forward validation'
    )

    parser.add_argument(
        '--cluster-validation',
        action='store_true',
        help='Enable symbol-cluster validation'
    )

    parser.add_argument(
        '--timeframes',
        type=str,
        default='1h,4h,1d',
        help='Comma-separated timeframes to use'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=2,
        help='Number of parallel jobs (default: 2 for lower CPU usage, use -1 for all cores)'
    )

    parser.add_argument(
        '--throttle',
        type=float,
        default=1.0,
        help='Seconds to pause between iterations for thermal management (default: 1.0)'
    )

    args = parser.parse_args()

    # Run all objectives
    if args.all_objectives:
        iterations = 20 if args.quick else args.iterations
        run_all_objectives(
            iterations_per_objective=iterations,
            quick=args.quick,
            n_jobs=args.n_jobs,
            throttle_seconds=args.throttle,
        )
        return

    # Run single objective
    timeframes = args.timeframes.split(',')

    config = Phase77OptimizationConfig(
        objective_type=ObjectiveType(args.objective),
        n_calls=20 if args.quick else args.iterations,
        n_initial_points=5 if args.quick else args.initial_points,
        random_state=args.seed,
        run_walk_forward=args.walk_forward,
        run_cluster_validation=args.cluster_validation,
        timeframes=timeframes,
        n_jobs=args.n_jobs,
        throttle_seconds=args.throttle,
    )

    optimizer = Phase77Optimizer(config)
    optimizer.optimize()


if __name__ == '__main__':
    main()
