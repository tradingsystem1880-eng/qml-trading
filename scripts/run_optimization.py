#!/usr/bin/env python3
"""
Run Bayesian Optimization for Pattern Detection
================================================
CLI for running parameter optimization across multiple symbols.

Usage:
    python scripts/run_optimization.py                     # Full optimization (100 iterations)
    python scripts/run_optimization.py --n-calls 20 --quick # Quick test run
    python scripts/run_optimization.py --symbols BTC ETH   # Specific symbols only
    python scripts/run_optimization.py --evaluate params.json  # Evaluate saved params
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try loguru, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    logger = logging.getLogger(__name__)


def run_optimization(
    n_calls: int = 100,
    n_initial: int = 20,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    n_jobs: int = -1,
    target_patterns: int = 500,
    checkpoint_every: int = 10,
) -> None:
    """
    Run Bayesian optimization for pattern detection parameters.

    Args:
        n_calls: Number of optimization iterations
        n_initial: Number of random initial points
        symbols: List of symbols to optimize on
        timeframes: List of timeframes to use
        n_jobs: Number of parallel jobs (-1 = all cores)
        target_patterns: Target pattern count for scoring
        checkpoint_every: Save checkpoint every N iterations
    """
    try:
        from src.optimization import BayesianOptimizer, OptimizationConfig
        from src.optimization.parallel_runner import ParallelDetectionRunner
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nRequired packages:")
        print("  pip install scikit-optimize>=0.9.0")
        sys.exit(1)

    # Create configs
    config = OptimizationConfig(
        n_calls=n_calls,
        n_initial_points=n_initial,
        target_patterns=target_patterns,
        checkpoint_every=checkpoint_every,
        n_jobs_detection=n_jobs,
    )

    # Create runner with specified symbols
    runner = ParallelDetectionRunner(
        symbols=symbols,
        timeframes=timeframes or ['4h'],
        n_jobs=n_jobs,
    )

    print(f"Symbols available: {len(runner.symbols)}")
    print(f"Timeframes: {runner.timeframes}")

    if len(runner.symbols) == 0:
        print("\nNo symbols with data found!")
        print("Run 'python scripts/expand_data.py' first to fetch data.")
        sys.exit(1)

    # Create optimizer and run
    optimizer = BayesianOptimizer(config=config, runner=runner)
    result = optimizer.optimize()

    print(f"\nOptimization complete!")
    print(f"Results saved to: {config.checkpoint_dir}")


def evaluate_params(
    params_path: Path,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
) -> None:
    """
    Evaluate a saved parameter configuration.

    Args:
        params_path: Path to JSON file with parameters
        symbols: Optional list of symbols to evaluate on
        timeframes: List of timeframes
    """
    from src.optimization import quick_evaluate

    # Load params
    with open(params_path) as f:
        data = json.load(f)

    if 'params' in data:
        params = data['params']
    else:
        params = data

    print("="*70)
    print("EVALUATING PARAMETERS")
    print("="*70)
    print("\nParameters:")
    for name, value in params.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("\nRunning evaluation...")

    result = quick_evaluate(params, symbols)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Patterns: {result.total_patterns}")
    print(f"Unique Symbols: {result.unique_symbols}/{result.total_symbols}")
    print(f"Mean Quality: {result.mean_score:.3f}")
    print(f"Median Quality: {result.median_score:.3f}")
    print(f"Tiers: A={result.tier_a_count} B={result.tier_b_count} C={result.tier_c_count}")

    # Show per-symbol breakdown
    print("\nPer-Symbol Results:")
    for key, r in sorted(result.symbol_results.items(), key=lambda x: -x[1].num_patterns):
        if r.num_patterns > 0:
            print(f"  {key}: {r.num_patterns} patterns, quality={r.mean_score:.3f}")


def show_best_params(results_dir: Optional[Path] = None) -> None:
    """Show the best parameters from optimization."""
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "optimization"

    best_path = results_dir / "best_params.json"

    if not best_path.exists():
        print("No optimization results found.")
        print(f"Expected: {best_path}")
        print("\nRun optimization first:")
        print("  python scripts/run_optimization.py")
        return

    with open(best_path) as f:
        data = json.load(f)

    print("="*70)
    print("BEST OPTIMIZED PARAMETERS")
    print("="*70)
    print(f"\nScore: {data.get('score', 'N/A')}")
    print(f"Patterns: {data.get('patterns', 'N/A')}")
    print(f"Iteration: {data.get('iteration', 'N/A')}")
    print("\nParameters:")
    for name, value in data.get('params', {}).items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for pattern detection parameters"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument(
        '--n-calls',
        type=int,
        default=100,
        help='Number of optimization iterations (default: 100)'
    )
    opt_parser.add_argument(
        '--n-initial',
        type=int,
        default=20,
        help='Number of random initial points (default: 20)'
    )
    opt_parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to optimize on'
    )
    opt_parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['4h'],
        help='Timeframes to use (default: 4h)'
    )
    opt_parser.add_argument(
        '--target-patterns',
        type=int,
        default=500,
        help='Target pattern count (default: 500)'
    )
    opt_parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run with fewer iterations'
    )
    opt_parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 = all cores)'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate saved parameters')
    eval_parser.add_argument(
        'params_file',
        type=Path,
        help='Path to parameters JSON file'
    )
    eval_parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to evaluate on'
    )
    eval_parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['4h'],
        help='Timeframes to use'
    )

    # Show best command
    show_parser = subparsers.add_parser('show-best', help='Show best optimized parameters')
    show_parser.add_argument(
        '--results-dir',
        type=Path,
        help='Path to results directory'
    )

    args = parser.parse_args()

    # Default to optimize if no command given
    if args.command is None:
        # Check for legacy flags
        if hasattr(args, 'n_calls') or len(sys.argv) > 1:
            # Re-parse with optimize as default
            args.command = 'optimize'
            args.n_calls = 100
            args.n_initial = 20
            args.symbols = None
            args.timeframes = ['4h']
            args.target_patterns = 500
            args.quick = False
            args.n_jobs = -1

            # Parse again for any flags
            for i, arg in enumerate(sys.argv[1:]):
                if arg == '--n-calls':
                    args.n_calls = int(sys.argv[i + 2])
                elif arg == '--quick':
                    args.quick = True
                elif arg == '--symbols':
                    args.symbols = []
                    j = i + 2
                    while j < len(sys.argv) and not sys.argv[j].startswith('--'):
                        args.symbols.append(sys.argv[j])
                        j += 1
        else:
            parser.print_help()
            return

    if args.command == 'optimize':
        if args.quick:
            args.n_calls = 20
            args.n_initial = 5
            print("Quick mode: 20 iterations")

        run_optimization(
            n_calls=args.n_calls,
            n_initial=args.n_initial,
            symbols=args.symbols,
            timeframes=args.timeframes,
            n_jobs=args.n_jobs,
            target_patterns=args.target_patterns,
        )

    elif args.command == 'evaluate':
        evaluate_params(
            params_path=args.params_file,
            symbols=args.symbols,
            timeframes=args.timeframes,
        )

    elif args.command == 'show-best':
        show_best_params(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
