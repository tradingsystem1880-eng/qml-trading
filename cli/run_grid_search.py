#!/usr/bin/env python3
"""
Grid Search Parameter Optimizer
===============================
VRD 2.0 Module 4: Parameter Sensitivity Analysis

Systematically searches parameter space to find:
- Optimal parameter combinations
- Stability landscape (robust vs brittle regions)
- Top performers by multiple metrics

Usage:
    python -m cli.run_grid_search
    python -m cli.run_grid_search --parallel 4
    python -m cli.run_grid_search --quick  # Reduced search space
"""

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SearchSpace:
    """Define the parameter search space."""
    atr_period: List[int]
    stop_loss_atr: List[float]
    take_profit_atr: List[float]
    min_validity_score: List[float]
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = ['atr_period', 'stop_loss_atr', 'take_profit_atr', 'min_validity_score']
        values = [
            self.atr_period,
            self.stop_loss_atr,
            self.take_profit_atr,
            self.min_validity_score
        ]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    @property
    def total_combinations(self) -> int:
        return (len(self.atr_period) * 
                len(self.stop_loss_atr) * 
                len(self.take_profit_atr) *
                len(self.min_validity_score))


def get_default_search_space() -> SearchSpace:
    """Get the default parameter search space."""
    return SearchSpace(
        atr_period=[10, 14, 21],
        stop_loss_atr=[1.0, 1.5, 2.0, 2.5],
        take_profit_atr=[2.0, 3.0, 4.0, 5.0],
        min_validity_score=[0.5, 0.6, 0.7]
    )


def get_quick_search_space() -> SearchSpace:
    """Get a reduced search space for quick testing."""
    return SearchSpace(
        atr_period=[14],
        stop_loss_atr=[1.0, 2.0],
        take_profit_atr=[2.0, 4.0],
        min_validity_score=[0.5, 0.7]
    )


def run_single_backtest(
    params: Dict[str, Any],
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a single backtest with given parameters.
    
    This function is designed to be called in parallel.
    
    Args:
        params: Parameter dictionary
        symbol: Trading symbol
        timeframe: Candle timeframe
        initial_capital: Starting capital
    
    Returns:
        Results dictionary with params and metrics
    """
    # Import inside function for multiprocessing
    from cli.run_backtest import BacktestConfig, BacktestEngine, load_data
    from src.detection import get_detector
    from src.reporting.storage import ExperimentLogger
    
    try:
        # Build config
        detector_config = {
            'atr_lookback': params.get('atr_period', 14),
            'min_validity_score': params.get('min_validity_score', 0.5),
            'stop_loss_atr_mult': params.get('stop_loss_atr', 1.5),
        }
        
        backtest_config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            detector_method='atr',
            min_validity_score=params.get('min_validity_score', 0.5),
        )
        
        # Override TP in the detector
        # Note: TP is calculated as multiples of risk, not ATR directly
        # For now, we'll use take_profit_atr as risk multiplier
        
        # Load data
        df = load_data(backtest_config)
        
        # Initialize detector
        detector = get_detector('atr', detector_config)
        
        # Run detection
        signals = detector.detect(df, symbol=symbol, timeframe=timeframe)
        
        # Override SL/TP based on params
        for sig in signals:
            if sig.stop_loss and sig.take_profit:
                # Recalculate based on search params
                risk = abs(sig.price - sig.stop_loss)
                tp_mult = params.get('take_profit_atr', 3.0)
                if sig.signal_type.value == 'BUY':
                    sig.take_profit = sig.price + (risk * tp_mult)
                else:
                    sig.take_profit = sig.price - (risk * tp_mult)
        
        # Run backtest
        engine = BacktestEngine(backtest_config)
        results = engine.run(df, signals)
        
        # Log to database
        logger = ExperimentLogger()
        
        # Create combined config for logging
        full_config = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_capital': initial_capital,
            'detector_method': 'atr',
            **params,
            'grid_search': True,
            'search_timestamp': datetime.now().isoformat()
        }
        
        run_id = logger.log_run(
            config=full_config,
            results=results,
            strategy_name='grid_search',
            tags=['grid_search', f"atr_{params.get('atr_period', 14)}"]
        )
        
        return {
            'run_id': run_id,
            'params': params,
            'pnl_percent': results.get('net_profit_pct', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'win_rate': results.get('win_rate', 0),
            'profit_factor': results.get('profit_factor', 0),
            'total_trades': results.get('total_trades', 0),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'params': params,
            'status': 'error',
            'error': str(e)
        }


def print_top_results(results: List[Dict[str, Any]], metric: str, n: int = 3) -> None:
    """Print top N results by a specific metric."""
    successful = [r for r in results if r.get('status') == 'success']
    
    if not successful:
        print(f"   No successful runs to rank by {metric}")
        return
    
    # Sort by metric (descending for most metrics)
    reverse = metric != 'max_drawdown'  # Lower DD is better
    sorted_results = sorted(
        successful,
        key=lambda x: x.get(metric, 0),
        reverse=reverse
    )
    
    print(f"\n   Top {n} by {metric.upper()}:")
    print("   " + "-" * 60)
    
    for i, r in enumerate(sorted_results[:n], 1):
        params = r['params']
        print(f"   {i}. ATR={params['atr_period']}, "
              f"SL={params['stop_loss_atr']}, "
              f"TP={params['take_profit_atr']}, "
              f"Val={params['min_validity_score']}")
        print(f"      {metric}={r.get(metric, 0):.4f}, "
              f"trades={r.get('total_trades', 0)}")


def run_grid_search(
    search_space: SearchSpace,
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    parallel: int = 1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run grid search over parameter space.
    
    Args:
        search_space: SearchSpace object defining parameters
        symbol: Trading symbol
        timeframe: Candle timeframe
        parallel: Number of parallel workers (1 = sequential)
        verbose: Print progress updates
    
    Returns:
        List of results for all combinations
    """
    combinations = search_space.get_combinations()
    total = len(combinations)
    
    if verbose:
        print(f"\n📊 Grid Search: {total} combinations")
        print(f"   Symbol: {symbol}, Timeframe: {timeframe}")
        print(f"   Parallel workers: {parallel}")
    
    results = []
    start_time = time.time()
    
    if parallel <= 1:
        # Sequential execution
        for i, params in enumerate(combinations, 1):
            if verbose:
                print(f"   [{i}/{total}] ATR={params['atr_period']}, "
                      f"SL={params['stop_loss_atr']}, TP={params['take_profit_atr']}...", 
                      end=" ", flush=True)
            
            result = run_single_backtest(params, symbol, timeframe)
            results.append(result)
            
            if verbose:
                if result['status'] == 'success':
                    print(f"Sharpe={result['sharpe_ratio']:.2f}, "
                          f"PnL={result['pnl_percent']:.1f}%")
                else:
                    print(f"ERROR: {result.get('error', 'Unknown')[:50]}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            future_to_params = {
                executor.submit(run_single_backtest, params, symbol, timeframe): params
                for params in combinations
            }
            
            completed = 0
            for future in as_completed(future_to_params):
                completed += 1
                result = future.result()
                results.append(result)
                
                if verbose:
                    params = future_to_params[future]
                    status = "✓" if result['status'] == 'success' else "✗"
                    print(f"   [{completed}/{total}] {status} ATR={params['atr_period']}", 
                          flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"\n✅ Completed: {successful}/{total} successful in {elapsed:.1f}s")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run grid search parameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--symbol", "-s",
        default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="4h",
        help="Candle timeframe"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Use reduced search space for quick testing"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate sensitivity heatmap after search"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🔍 VRD 2.0 PARAMETER GRID SEARCH")
    print("=" * 70)
    
    # Get search space
    if args.quick:
        search_space = get_quick_search_space()
        print("   Mode: QUICK (reduced search space)")
    else:
        search_space = get_default_search_space()
        print("   Mode: FULL search space")
    
    print(f"   Total combinations: {search_space.total_combinations}")
    print("=" * 70)
    
    # Run grid search
    results = run_grid_search(
        search_space=search_space,
        symbol=args.symbol,
        timeframe=args.timeframe,
        parallel=args.parallel,
        verbose=True
    )
    
    # Print top results
    print("\n" + "=" * 70)
    print("  🏆 TOP PARAMETER SETS")
    print("=" * 70)
    
    print_top_results(results, 'sharpe_ratio', n=3)
    print_top_results(results, 'pnl_percent', n=3)
    print_top_results(results, 'profit_factor', n=3)
    
    # Generate sensitivity plot
    if args.plot:
        print("\n📊 Generating sensitivity heatmap...")
        try:
            from src.analysis.sensitivity import SensitivityVisualizer
            viz = SensitivityVisualizer()
            plot_path = viz.plot_heatmap(metric='sharpe_ratio')
            print(f"   Saved to: {plot_path}")
        except Exception as e:
            print(f"   Could not generate plot: {e}")
    
    print("\n" + "=" * 70)
    print("  ✅ GRID SEARCH COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
