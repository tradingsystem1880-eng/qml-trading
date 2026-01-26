"""
Trade Distribution Analysis
============================
Diagnose WHY higher win rate led to lower profit factor.

Analyzes:
- Average win/loss in R-multiples
- Realized R:R ratio
- MFE capture ratio (are we cutting winners short?)
- Win size distribution (outlier dependence)

Usage:
    python scripts/analyze_trade_distribution.py --results results/phase77_optimization/composite/final_results.json
    python scripts/analyze_trade_distribution.py --iteration 70
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_trade_metrics(iteration: Dict) -> Dict[str, Any]:
    """
    Analyze trade distribution metrics for an iteration.

    This diagnoses the win rate vs profitability paradox.
    """
    # Extract metrics (some may need to be calculated from raw data)
    win_rate = iteration.get('win_rate', 0)
    profit_factor = iteration.get('profit_factor', 0)
    sharpe = iteration.get('sharpe', 0)
    total_trades = iteration.get('total_trades', 0)
    expectancy = iteration.get('expectancy', 0)

    # These may be in simulation results if available
    avg_win_r = iteration.get('avg_win_r', None)
    avg_loss_r = iteration.get('avg_loss_r', None)
    avg_mfe_r = iteration.get('avg_mfe_r', None)
    avg_mae_r = iteration.get('avg_mae_r', None)

    analysis = {
        'iteration': iteration.get('iteration'),
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'expectancy_r': expectancy,
    }

    # Calculate realized R:R if we have avg win/loss
    if avg_win_r is not None and avg_loss_r is not None and avg_loss_r > 0:
        realized_rr = avg_win_r / avg_loss_r
        analysis['avg_win_r'] = avg_win_r
        analysis['avg_loss_r'] = avg_loss_r
        analysis['realized_rr'] = realized_rr

    # Calculate MFE capture ratio if available
    if avg_mfe_r is not None and avg_mfe_r > 0 and avg_win_r is not None:
        mfe_capture = avg_win_r / avg_mfe_r
        analysis['avg_mfe_r'] = avg_mfe_r
        analysis['mfe_capture_ratio'] = mfe_capture

    if avg_mae_r is not None:
        analysis['avg_mae_r'] = avg_mae_r

    return analysis


def diagnose_paradox(analysis: Dict) -> List[str]:
    """
    Diagnose potential causes of win rate vs profitability paradox.

    Returns list of diagnostic insights.
    """
    diagnoses = []

    win_rate = analysis.get('win_rate', 0)
    profit_factor = analysis.get('profit_factor', 0)
    realized_rr = analysis.get('realized_rr', None)
    mfe_capture = analysis.get('mfe_capture_ratio', None)

    # High win rate but low PF = small wins, normal losses
    if win_rate > 0.5 and profit_factor < 1.0:
        diagnoses.append(
            "PARADOX DETECTED: High win rate (>{:.0%}) but unprofitable (PF {:.2f})".format(
                win_rate, profit_factor
            )
        )

        if realized_rr is not None and realized_rr < 1.0:
            diagnoses.append(
                f"  ROOT CAUSE: Avg win ({realized_rr:.2f}R) is smaller than avg loss (1R)"
            )
            diagnoses.append(
                "  SOLUTION: Either widen take-profit targets OR tighten stop-losses"
            )

    # MFE capture analysis
    if mfe_capture is not None:
        if mfe_capture < 0.5:
            diagnoses.append(
                f"  MFE CAPTURE: Only capturing {mfe_capture:.1%} of favorable moves"
            )
            diagnoses.append(
                "  INSIGHT: Winners are going much further after you exit - consider trailing stops"
            )
        elif mfe_capture > 0.8:
            diagnoses.append(
                f"  MFE CAPTURE: Capturing {mfe_capture:.1%} of moves - exits are efficient"
            )

    # Expectancy analysis
    expectancy = analysis.get('expectancy_r', 0)
    if expectancy < 0:
        diagnoses.append(
            f"  EXPECTANCY: Negative ({expectancy:.3f}R per trade) - no statistical edge"
        )
    elif expectancy > 0 and profit_factor < 1.0:
        diagnoses.append(
            f"  ANOMALY: Positive expectancy ({expectancy:.3f}R) but PF < 1.0 - check calculation"
        )

    return diagnoses


def print_analysis(analysis: Dict, diagnoses: List[str]):
    """Print formatted analysis results."""
    print("\n" + "="*70)
    print("TRADE DISTRIBUTION ANALYSIS")
    print("="*70)

    print(f"\nIteration: {analysis.get('iteration', 'N/A')}")
    print(f"Total Trades: {analysis.get('total_trades', 0)}")

    print("\n--- Core Metrics ---")
    print(f"  Win Rate:      {analysis.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor: {analysis.get('profit_factor', 0):.2f}")
    print(f"  Sharpe:        {analysis.get('sharpe', 0):.4f}")
    print(f"  Expectancy:    {analysis.get('expectancy_r', 0):.3f}R per trade")

    if 'avg_win_r' in analysis:
        print("\n--- R-Multiple Analysis ---")
        print(f"  Avg Win:       {analysis['avg_win_r']:.2f}R")
        print(f"  Avg Loss:      {analysis['avg_loss_r']:.2f}R")
        print(f"  Realized R:R:  {analysis.get('realized_rr', 0):.2f}")

    if 'mfe_capture_ratio' in analysis:
        print("\n--- Exit Efficiency ---")
        print(f"  Avg MFE:       {analysis['avg_mfe_r']:.2f}R (best point reached)")
        print(f"  MFE Capture:   {analysis['mfe_capture_ratio']*100:.1f}% (how much we captured)")
        if 'avg_mae_r' in analysis:
            print(f"  Avg MAE:       {analysis['avg_mae_r']:.2f}R (worst drawdown)")

    if diagnoses:
        print("\n" + "="*70)
        print("DIAGNOSES")
        print("="*70)
        for diag in diagnoses:
            print(diag)


def compare_iterations_distribution(history: List[Dict], iter_nums: List[int]):
    """Compare trade distributions across multiple iterations."""
    print("\n" + "="*70)
    print("ITERATION COMPARISON - Trade Distribution")
    print("="*70)

    # Header
    print(f"\n{'Iter':<6} {'WR':<8} {'PF':<8} {'Sharpe':<10} {'AvgWin':<10} {'AvgLoss':<10} {'RR':<8}")
    print("-" * 70)

    for iter_num in iter_nums:
        if iter_num <= 0 or iter_num > len(history):
            continue
        iteration = history[iter_num - 1]

        wr = iteration.get('win_rate', 0) * 100
        pf = iteration.get('profit_factor', 0)
        sharpe = iteration.get('sharpe', 0)
        avg_win = iteration.get('avg_win_r', 0)
        avg_loss = iteration.get('avg_loss_r', 0)
        rr = avg_win / avg_loss if avg_loss > 0 else 0

        print(f"{iter_num:<6} {wr:>6.1f}% {pf:>7.2f} {sharpe:>9.4f} {avg_win:>9.2f}R {avg_loss:>9.2f}R {rr:>7.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze trade distribution')
    parser.add_argument('--results', type=str,
                       default='results/phase77_optimization/composite/final_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--iteration', type=int, help='Analyze specific iteration')
    parser.add_argument('--compare', type=str, help='Compare iterations (comma-separated, e.g. "70,75,53")')
    parser.add_argument('--best', action='store_true', help='Analyze best Sharpe iteration')
    args = parser.parse_args()

    # Load results
    results_path = PROJECT_ROOT / args.results
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print("Error: No iteration history found")
        return

    print(f"Loaded {len(history)} iterations")

    if args.compare:
        # Compare multiple iterations
        iter_nums = [int(x.strip()) for x in args.compare.split(',')]
        compare_iterations_distribution(history, iter_nums)

        # Also print individual analyses
        for iter_num in iter_nums:
            if iter_num <= 0 or iter_num > len(history):
                continue
            iteration = history[iter_num - 1]
            analysis = analyze_trade_metrics(iteration)
            diagnoses = diagnose_paradox(analysis)
            print_analysis(analysis, diagnoses)

    elif args.iteration:
        # Analyze specific iteration
        if args.iteration <= 0 or args.iteration > len(history):
            print(f"Error: Iteration {args.iteration} not found")
            return
        iteration = history[args.iteration - 1]
        analysis = analyze_trade_metrics(iteration)
        diagnoses = diagnose_paradox(analysis)
        print_analysis(analysis, diagnoses)

    elif args.best:
        # Find and analyze best Sharpe iteration
        valid_iters = [h for h in history if h.get('total_trades', 0) > 50]
        if not valid_iters:
            print("No valid iterations found")
            return

        best = min(valid_iters, key=lambda x: abs(x.get('sharpe', -999)))
        analysis = analyze_trade_metrics(best)
        diagnoses = diagnose_paradox(analysis)
        print_analysis(analysis, diagnoses)

    else:
        # Default: analyze the "best" composite result and compare to best Sharpe
        valid_iters = [h for h in history if h.get('total_trades', 0) > 50]

        if valid_iters:
            # Best composite (highest score)
            best_composite = max(valid_iters, key=lambda x: x.get('score', -999))
            # Best Sharpe
            best_sharpe = min(valid_iters, key=lambda x: abs(x.get('sharpe', -999)))

            print("\n" + "="*70)
            print("BEST COMPOSITE (Selected) vs BEST SHARPE (Discarded)")
            print("="*70)

            # Analyze best composite
            analysis_comp = analyze_trade_metrics(best_composite)
            diagnoses_comp = diagnose_paradox(analysis_comp)
            print_analysis(analysis_comp, diagnoses_comp)

            # Analyze best Sharpe
            analysis_sharpe = analyze_trade_metrics(best_sharpe)
            diagnoses_sharpe = diagnose_paradox(analysis_sharpe)
            print_analysis(analysis_sharpe, diagnoses_sharpe)

            # Summary comparison
            print("\n" + "="*70)
            print("KEY INSIGHT")
            print("="*70)
            print(f"\nBest Composite selected iteration {best_composite.get('iteration')}")
            print(f"  Sharpe: {best_composite.get('sharpe', 0):.4f}, PF: {best_composite.get('profit_factor', 0):.2f}")
            print(f"\nBut iteration {best_sharpe.get('iteration')} had better trading metrics:")
            print(f"  Sharpe: {best_sharpe.get('sharpe', 0):.4f}, PF: {best_sharpe.get('profit_factor', 0):.2f}")
            print(f"\nThis is why we need PROFIT-FOCUSED objectives, not pattern-focused!")


if __name__ == "__main__":
    main()
