"""
Bayesian Optimizer for Pattern Detection
========================================
Uses scikit-optimize for Bayesian optimization of detection parameters.

Objective: Maximize pattern count * quality while maintaining symbol diversity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import json
import time
import pickle

import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.callbacks import CheckpointSaver, EarlyStopper
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    # Define dummy types for when skopt isn't installed
    Real = Integer = Categorical = object

from src.optimization.parallel_runner import ParallelDetectionRunner, AggregateResult


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# PARAMETER SPACE
# =============================================================================

# Define the search space for optimization
# Each parameter has a reasonable range based on domain knowledge

if SKOPT_AVAILABLE:
    PARAM_SPACE = [
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

        # Scoring Weights (2 params)
        Real(0.2, 0.4, name='head_extension_weight'),
        Real(0.15, 0.35, name='bos_efficiency_weight'),
    ]

    PARAM_NAMES = [p.name for p in PARAM_SPACE]
else:
    PARAM_SPACE = []
    PARAM_NAMES = [
        'min_bar_separation', 'min_move_atr', 'forward_confirm_pct',
        'lookback', 'lookforward', 'p3_min_extension_atr',
        'p3_max_extension_atr', 'p4_min_break_atr', 'p5_max_symmetry_atr',
        'min_pattern_bars', 'min_adx', 'min_trend_move_atr',
        'min_trend_swings', 'head_extension_weight', 'bos_efficiency_weight',
    ]


@dataclass
class OptimizationConfig:
    """Configuration for the optimization run."""
    # Number of optimization iterations
    n_calls: int = 100

    # Number of random initial points
    n_initial_points: int = 20

    # Random seed for reproducibility
    random_state: int = 42

    # Parallelization
    n_jobs_detection: int = -1  # For detection runner

    # Targets
    min_patterns: int = 50  # Penalty if below this
    min_symbol_diversity: float = 0.33  # Penalty if below this
    target_patterns: int = 500  # Optimal pattern count

    # Objective weights
    weight_pattern_count: float = 0.3
    weight_quality: float = 0.5
    weight_diversity: float = 0.2

    # Checkpointing
    checkpoint_every: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "optimization")

    # Early stopping
    early_stop_patience: int = 20  # Stop if no improvement for N iterations


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: Dict[str, Any]
    best_score: float
    best_result: AggregateResult
    all_scores: List[float]
    all_params: List[Dict[str, Any]]
    n_iterations: int
    duration_seconds: float

    def save(self, path: Path):
        """Save results to JSON file."""
        # Helper to convert numpy types to native Python
        def to_native(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        data = {
            'best_params': to_native(self.best_params),
            'best_score': float(self.best_score),
            'best_result': {
                'total_patterns': int(self.best_result.total_patterns),
                'unique_symbols': int(self.best_result.unique_symbols),
                'mean_score': float(self.best_result.mean_score),
                'tier_a_count': int(self.best_result.tier_a_count),
                'tier_b_count': int(self.best_result.tier_b_count),
                'tier_c_count': int(self.best_result.tier_c_count),
            },
            'all_scores': [float(s) for s in self.all_scores],
            'all_params': [to_native(p) for p in self.all_params],
            'n_iterations': int(self.n_iterations),
            'duration_seconds': float(self.duration_seconds),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class BayesianOptimizer:
    """
    Bayesian optimizer for pattern detection parameters.

    Uses Gaussian Process-based optimization (via scikit-optimize) to
    efficiently search the parameter space.

    Objective: Maximize combined score of:
    - Pattern count (capped at target)
    - Mean pattern quality
    - Symbol diversity
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        runner: Optional[ParallelDetectionRunner] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            config: Optimization configuration
            runner: Pre-configured detection runner (optional)
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install with: pip install scikit-optimize>=0.9.0"
            )

        self.config = config or OptimizationConfig()
        self.runner = runner or ParallelDetectionRunner(n_jobs=self.config.n_jobs_detection)

        # Tracking
        self._iteration = 0
        self._best_score = float('inf')  # We minimize, so start high
        self._best_params = None
        self._best_result = None
        self._history: List[Tuple[Dict, float, AggregateResult]] = []

        # Ensure checkpoint directory exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def objective(self, params_list: List[Any]) -> float:
        """
        Objective function for optimization.

        Evaluates a parameter configuration and returns a score to minimize.
        (Lower is better since gp_minimize minimizes.)

        Args:
            params_list: List of parameter values in PARAM_SPACE order

        Returns:
            Negative combined score (to minimize)
        """
        self._iteration += 1

        # Convert list to dict
        params = dict(zip(PARAM_NAMES, params_list))

        # Run detection
        result = self.runner.run_with_dict(params)

        # Calculate combined score
        score = self._calculate_score(result)

        # Track history
        self._history.append((params, score, result))

        # Update best
        if score < self._best_score:
            self._best_score = score
            self._best_params = params
            self._best_result = result

        # Log progress
        self._log_iteration(params, result, score)

        # Checkpoint
        if self._iteration % self.config.checkpoint_every == 0:
            self._save_checkpoint()

        return score

    def _calculate_score(self, result: AggregateResult) -> float:
        """
        Calculate combined objective score.

        Returns negative score since gp_minimize minimizes.
        """
        cfg = self.config

        # Penalties for not meeting minimums
        if result.total_patterns < cfg.min_patterns:
            return 1000.0  # Heavy penalty

        symbol_diversity = result.unique_symbols / max(result.total_symbols, 1)
        if symbol_diversity < cfg.min_symbol_diversity:
            return 500.0  # Medium penalty

        # Component scores (0-1 range)
        pattern_score = min(result.total_patterns, cfg.target_patterns) / cfg.target_patterns
        quality_score = result.mean_score  # Already 0-1
        diversity_score = symbol_diversity

        # Combined weighted score
        combined = (
            cfg.weight_pattern_count * pattern_score +
            cfg.weight_quality * quality_score +
            cfg.weight_diversity * diversity_score
        )

        # Return negative (since we minimize)
        return -combined

    def _log_iteration(
        self,
        params: Dict[str, Any],
        result: AggregateResult,
        score: float
    ):
        """Log progress for current iteration."""
        print(f"\n[Iter {self._iteration}] Score: {-score:.4f}")
        print(f"  Patterns: {result.total_patterns} | "
              f"Symbols: {result.unique_symbols}/{result.total_symbols} | "
              f"Quality: {result.mean_score:.3f}")
        print(f"  Tiers: A={result.tier_a_count} B={result.tier_b_count} C={result.tier_c_count}")
        print(f"  Best so far: {-self._best_score:.4f}")

    def _save_checkpoint(self):
        """Save checkpoint of current state."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{self._iteration}.pkl"
        best_params_path = self.config.checkpoint_dir / "best_params.json"

        # Save best params as JSON
        if self._best_params:
            # Convert numpy types to native Python types
            params_native = {}
            for k, v in self._best_params.items():
                if hasattr(v, 'item'):  # numpy scalar
                    params_native[k] = v.item()
                else:
                    params_native[k] = v

            with open(best_params_path, 'w') as f:
                json.dump({
                    'params': params_native,
                    'score': float(-self._best_score),
                    'patterns': int(self._best_result.total_patterns) if self._best_result else 0,
                    'iteration': int(self._iteration),
                }, f, indent=2)

        print(f"  Checkpoint saved: {best_params_path}")

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization.

        Returns:
            OptimizationResult with best parameters and statistics
        """
        print("="*70)
        print("BAYESIAN OPTIMIZATION - Phase 7.6")
        print("="*70)
        print(f"Parameters: {len(PARAM_SPACE)}")
        print(f"Iterations: {self.config.n_calls}")
        print(f"Initial points: {self.config.n_initial_points}")
        print(f"Target patterns: {self.config.target_patterns}")
        print("="*70)

        start_time = time.time()

        # Preload data for faster iteration
        print("\nPreloading data...")
        self.runner.preload_data()
        print(f"Loaded {len(self.runner._data_cache)} datasets")

        # Run optimization
        print("\nStarting optimization...\n")

        result = gp_minimize(
            func=self.objective,
            dimensions=PARAM_SPACE,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=False,  # We do our own logging
            n_jobs=1,  # Serial objective evaluation (parallelism in runner)
        )

        duration = time.time() - start_time

        # Build result
        opt_result = OptimizationResult(
            best_params=self._best_params,
            best_score=float(-self._best_score),
            best_result=self._best_result,
            all_scores=[-s for s in result.func_vals],
            all_params=[dict(zip(PARAM_NAMES, x)) for x in result.x_iters],
            n_iterations=self._iteration,
            duration_seconds=duration,
        )

        # Print summary
        self._print_summary(opt_result)

        # Save final results
        final_path = self.config.checkpoint_dir / "final_result.json"
        opt_result.save(final_path)
        print(f"\nResults saved to: {final_path}")

        return opt_result

    def _print_summary(self, result: OptimizationResult):
        """Print optimization summary."""
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Total iterations: {result.n_iterations}")
        print(f"Duration: {result.duration_seconds/60:.1f} minutes")
        print(f"\nBest Score: {result.best_score:.4f}")
        print(f"Best Patterns: {result.best_result.total_patterns}")
        print(f"Best Quality: {result.best_result.mean_score:.3f}")
        print(f"Best Diversity: {result.best_result.unique_symbols}/{result.best_result.total_symbols}")

        print("\nBest Parameters:")
        for name, value in result.best_params.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")


def load_best_params(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the best parameters from an optimization run.

    Args:
        path: Path to best_params.json (default: results/optimization/best_params.json)

    Returns:
        Dictionary of parameter names to values
    """
    if path is None:
        path = PROJECT_ROOT / "results" / "optimization" / "best_params.json"

    if not path.exists():
        raise FileNotFoundError(f"No optimization results found at {path}")

    with open(path) as f:
        data = json.load(f)

    return data['params']


def quick_evaluate(params: Dict[str, Any], symbols: Optional[List[str]] = None) -> AggregateResult:
    """
    Quickly evaluate a parameter configuration.

    Useful for testing parameter sets before running full optimization.

    Args:
        params: Dictionary of parameter names to values
        symbols: Optional list of symbols (default: all available)

    Returns:
        AggregateResult with detection statistics
    """
    runner = ParallelDetectionRunner(symbols=symbols)
    return runner.run_with_dict(params)
