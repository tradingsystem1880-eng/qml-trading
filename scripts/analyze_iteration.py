"""
Analyze Iteration from Optimization Results
============================================
Extract and analyze parameters from a specific iteration.

Usage:
    python scripts/analyze_iteration.py --iteration 70
    python scripts/analyze_iteration.py --best-sharpe
    python scripts/analyze_iteration.py --best-pf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


def load_optimization_results(results_path: Path) -> Dict[str, Any]:
    """Load optimization results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def find_best_iterations(history: List[Dict]) -> Dict[str, Dict]:
    """Find iterations with best values for each metric."""
    valid_iters = [h for h in history if h.get('total_trades', 0) > 50]

    if not valid_iters:
        return {}

    results = {}

    # Best Sharpe (closest to 0, prefer positive)
    sharpe_sorted = sorted(valid_iters, key=lambda x: abs(x.get('sharpe', -999)))
    results['best_sharpe'] = sharpe_sorted[0] if sharpe_sorted else None

    # Best Profit Factor
    pf_sorted = sorted(valid_iters, key=lambda x: -x.get('profit_factor', 0))
    results['best_pf'] = pf_sorted[0] if pf_sorted else None

    # Best Win Rate
    wr_sorted = sorted(valid_iters, key=lambda x: -x.get('win_rate', 0))
    results['best_win_rate'] = wr_sorted[0] if wr_sorted else None

    # Best Expectancy
    exp_sorted = sorted(valid_iters, key=lambda x: -x.get('expectancy', 0))
    results['best_expectancy'] = exp_sorted[0] if exp_sorted else None

    # Best Composite Score (that was actually selected)
    score_sorted = sorted(valid_iters, key=lambda x: -x.get('score', -999))
    results['best_composite'] = score_sorted[0] if score_sorted else None

    return results


def print_iteration_details(iteration: Dict, title: str = "Iteration Details"):
    """Print detailed information about an iteration."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    print(f"\nIteration: {iteration.get('iteration', 'N/A')}")
    print(f"Score: {iteration.get('score', 0):.4f}")

    print(f"\n--- Trading Metrics ---")
    print(f"  Sharpe:        {iteration.get('sharpe', 0):.4f}")
    print(f"  Profit Factor: {iteration.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:      {iteration.get('win_rate', 0)*100:.1f}%")
    print(f"  Expectancy:    {iteration.get('expectancy', 0):.3f}R")
    print(f"  Total Trades:  {iteration.get('total_trades', 0)}")
    print(f"  Max Drawdown:  {iteration.get('max_drawdown', 0):.2f}R")

    print(f"\n--- Detection Metrics ---")
    print(f"  Patterns:      {iteration.get('total_patterns', 0)}")
    print(f"  Quality:       {iteration.get('mean_quality', 0):.3f}")
    print(f"  Symbols:       {iteration.get('unique_symbols', 0)}")

    if 'params' in iteration:
        print(f"\n--- Parameters ---")
        params = iteration['params']

        # Group parameters
        groups = {
            'Swing Detection': ['min_bar_separation', 'min_move_atr', 'forward_confirm_pct',
                               'lookback', 'lookforward'],
            'Pattern Validation': ['p3_min_extension_atr', 'p3_max_extension_atr',
                                  'p4_min_break_atr', 'p5_max_symmetry_atr', 'min_pattern_bars'],
            'Trend Validation': ['min_adx', 'min_trend_move_atr', 'min_trend_swings', 'min_r_squared'],
            'Scoring Weights': ['head_extension_weight', 'bos_efficiency_weight',
                               'volume_spike_weight', 'path_efficiency_weight',
                               'trend_strength_weight', 'regime_suitability_weight'],
            'Trade Management': ['entry_buffer_atr', 'sl_atr_mult', 'tp_atr_mult',
                                'trailing_activation_atr', 'trailing_step_atr',
                                'max_bars_held', 'min_risk_reward'],
            'Costs': ['slippage_pct', 'commission_pct'],
        }

        for group_name, param_names in groups.items():
            print(f"\n  {group_name}:")
            for name in param_names:
                if name in params:
                    val = params[name]
                    if isinstance(val, float):
                        print(f"    {name}: {val:.4f}")
                    else:
                        print(f"    {name}: {val}")


def compare_iterations(iter1: Dict, iter2: Dict, name1: str = "Iter 1", name2: str = "Iter 2"):
    """Compare two iterations side-by-side."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*70}")

    # Metrics comparison
    metrics = ['sharpe', 'profit_factor', 'win_rate', 'expectancy', 'total_trades', 'max_drawdown']

    print(f"\n{'Metric':<20} {name1:<15} {name2:<15} {'Diff':<15}")
    print("-" * 65)

    for metric in metrics:
        v1 = iter1.get(metric, 0)
        v2 = iter2.get(metric, 0)
        diff = v1 - v2

        if metric == 'win_rate':
            print(f"{metric:<20} {v1*100:>12.1f}% {v2*100:>12.1f}% {diff*100:>+12.1f}%")
        else:
            print(f"{metric:<20} {v1:>14.4f} {v2:>14.4f} {diff:>+14.4f}")

    # Parameter differences
    if 'params' in iter1 and 'params' in iter2:
        print(f"\n--- Key Parameter Differences ---")
        p1 = iter1['params']
        p2 = iter2['params']

        key_params = ['min_adx', 'tp_atr_mult', 'sl_atr_mult', 'min_risk_reward',
                     'trailing_activation_atr', 'regime_suitability_weight']

        print(f"\n{'Parameter':<25} {name1:<12} {name2:<12} {'Diff':<12}")
        print("-" * 55)

        for param in key_params:
            v1 = p1.get(param, 0)
            v2 = p2.get(param, 0)
            diff = v1 - v2
            print(f"{param:<25} {v1:>11.3f} {v2:>11.3f} {diff:>+11.3f}")


def save_iteration_params(iteration: Dict, output_path: Path):
    """Save iteration parameters to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(iteration, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze optimization iteration')
    parser.add_argument('--iteration', type=int, help='Iteration number to analyze')
    parser.add_argument('--best-sharpe', action='store_true', help='Find iteration with best Sharpe')
    parser.add_argument('--best-pf', action='store_true', help='Find iteration with best Profit Factor')
    parser.add_argument('--best-wr', action='store_true', help='Find iteration with best Win Rate')
    parser.add_argument('--compare', action='store_true', help='Compare best Sharpe to best Composite')
    parser.add_argument('--results-file', type=str,
                       default='results/phase77_optimization/composite/final_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--save', type=str, help='Save iteration params to this file')
    args = parser.parse_args()

    # Load results
    results_path = PROJECT_ROOT / args.results_file
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    data = load_optimization_results(results_path)
    history = data.get('history', [])

    if not history:
        print("Error: No iteration history found in results")
        return

    print(f"Loaded {len(history)} iterations from {results_path}")

    # Find best iterations for each metric
    best_iters = find_best_iterations(history)

    if args.iteration:
        # Get specific iteration (1-indexed)
        if args.iteration <= 0 or args.iteration > len(history):
            print(f"Error: Iteration {args.iteration} not found (valid: 1-{len(history)})")
            return
        iteration = history[args.iteration - 1]
        print_iteration_details(iteration, f"Iteration {args.iteration}")

        if args.save:
            save_iteration_params(iteration, Path(args.save))

    elif args.best_sharpe:
        if best_iters.get('best_sharpe'):
            print_iteration_details(best_iters['best_sharpe'], "BEST SHARPE")
            if args.save:
                save_iteration_params(best_iters['best_sharpe'], Path(args.save))
        else:
            print("No valid iterations found for best Sharpe")

    elif args.best_pf:
        if best_iters.get('best_pf'):
            print_iteration_details(best_iters['best_pf'], "BEST PROFIT FACTOR")
            if args.save:
                save_iteration_params(best_iters['best_pf'], Path(args.save))
        else:
            print("No valid iterations found for best Profit Factor")

    elif args.best_wr:
        if best_iters.get('best_win_rate'):
            print_iteration_details(best_iters['best_win_rate'], "BEST WIN RATE")
            if args.save:
                save_iteration_params(best_iters['best_win_rate'], Path(args.save))
        else:
            print("No valid iterations found for best Win Rate")

    elif args.compare:
        if best_iters.get('best_sharpe') and best_iters.get('best_composite'):
            print_iteration_details(best_iters['best_sharpe'], "BEST SHARPE")
            print_iteration_details(best_iters['best_composite'], "BEST COMPOSITE (Selected)")
            compare_iterations(
                best_iters['best_sharpe'],
                best_iters['best_composite'],
                "Best Sharpe",
                "Best Composite"
            )
        else:
            print("Could not find both iterations for comparison")
    else:
        # Show summary of best iterations
        print("\n" + "="*70)
        print("BEST ITERATIONS SUMMARY")
        print("="*70)

        for name, iteration in best_iters.items():
            if iteration:
                print(f"\n{name.upper().replace('_', ' ')}:")
                print(f"  Iteration: {iteration.get('iteration')}")
                print(f"  Sharpe: {iteration.get('sharpe', 0):.4f}")
                print(f"  PF: {iteration.get('profit_factor', 0):.2f}")
                print(f"  WR: {iteration.get('win_rate', 0)*100:.1f}%")
                print(f"  Score: {iteration.get('score', 0):.4f}")


if __name__ == "__main__":
    main()
