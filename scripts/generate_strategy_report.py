#!/usr/bin/env python3
"""
Phase 9.6: Strategy Report Generator
====================================
Orchestrates the complete strategy validation report generation.

Steps:
1. Collect trade data with features (if needed)
2. Run SHAP analysis (if needed)
3. Generate feature scatter plots (if needed)
4. Load validation results
5. Generate comprehensive HTML report

Usage:
    python scripts/generate_strategy_report.py
    python scripts/generate_strategy_report.py --skip-shap
    python scripts/generate_strategy_report.py --trades-file results/report_trades.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.strategy_report import StrategyReportGenerator


def find_latest_validation_results(results_dir: Path) -> Dict:
    """Find and load latest validation results."""
    validation_results = {}

    # Check phase95 validation directory
    phase95_dir = results_dir / "phase95_validation"
    if phase95_dir.exists():
        # Load permutation test
        perm_files = list(phase95_dir.glob("permutation_test_*.json"))
        if perm_files:
            latest = max(perm_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                validation_results['permutation_test'] = json.load(f)

        # Load walk-forward
        wf_files = list(phase95_dir.glob("walk_forward_*.json"))
        if wf_files:
            latest = max(wf_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                validation_results['walk_forward'] = json.load(f)

        # Load Monte Carlo
        mc_files = list(phase95_dir.glob("monte_carlo_*.json"))
        if mc_files:
            latest = max(mc_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                validation_results['monte_carlo'] = json.load(f)

        # Load bootstrap
        boot_files = list(phase95_dir.glob("bootstrap_*.json"))
        if boot_files:
            latest = max(boot_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                validation_results['bootstrap'] = json.load(f)

    # Check phase94 as fallback
    phase94_dir = results_dir / "phase94_validation"
    if phase94_dir.exists() and not validation_results:
        for json_file in phase94_dir.glob("*.json"):
            test_name = json_file.stem.split('_')[0]
            if test_name not in validation_results:
                with open(json_file) as f:
                    validation_results[test_name] = json.load(f)

    return validation_results


def calculate_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics from trades."""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
        }

    pnl = trades_df['pnl_r']
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    win_rate = len(winners) / len(pnl) if len(pnl) > 0 else 0
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = pnl.mean()

    # Calculate drawdown
    cumulative = pnl.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = (running_max - cumulative) / (running_max.abs() + 1)
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    # Simple Sharpe approximation
    sharpe_ratio = pnl.mean() / pnl.std() if pnl.std() > 0 else 0

    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_r': pnl.sum(),
        'var_95': np.percentile(pnl, 5) if len(pnl) > 0 else 0,
        'cvar_95': pnl[pnl <= np.percentile(pnl, 5)].mean() if len(pnl) > 0 else 0,
        'risk_of_ruin': 0.01,  # Placeholder
    }


def run_data_collection(symbols: List[str], output_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Run trade data collection if needed."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: COLLECTING TRADE DATA")
        print("=" * 70)

    # Import and run collection
    try:
        from collect_report_data import collect_trades_with_features
        trades_df = collect_trades_with_features(symbols, verbose=verbose)
        trades_df.to_parquet(output_path, index=False)
        if verbose:
            print(f"\nSaved {len(trades_df)} trades to: {output_path}")
        return trades_df
    except Exception as e:
        print(f"ERROR collecting trades: {e}")
        return pd.DataFrame()


def run_shap_analysis(trades_df: pd.DataFrame, output_dir: Path, verbose: bool = True) -> Optional[pd.DataFrame]:
    """Run SHAP analysis if needed."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: RUNNING SHAP ANALYSIS")
        print("=" * 70)

    try:
        from run_shap_analysis import run_shap_analysis as shap_analyze, FEATURE_COLUMNS
        feature_importance = shap_analyze(
            trades_df,
            FEATURE_COLUMNS,
            output_dir=str(output_dir),
            verbose=verbose,
        )
        return feature_importance
    except ImportError as e:
        print(f"SHAP not available: {e}")
        print("Skipping SHAP analysis. Install with: pip install shap xgboost")
        return None
    except Exception as e:
        print(f"ERROR in SHAP analysis: {e}")
        return None


def run_scatter_plots(trades_df: pd.DataFrame, output_dir: Path, verbose: bool = True):
    """Generate feature scatter plots."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: GENERATING FEATURE PLOTS")
        print("=" * 70)

    try:
        from generate_feature_scatter import generate_scatter_matrix, generate_correlation_heatmap, FEATURE_COLUMNS

        generate_scatter_matrix(
            trades_df,
            FEATURE_COLUMNS,
            str(output_dir / 'feature_scatter_matrix.png'),
            verbose=verbose,
        )

        generate_correlation_heatmap(
            trades_df,
            FEATURE_COLUMNS,
            str(output_dir / 'feature_correlation_heatmap.png'),
            verbose=verbose,
        )
    except Exception as e:
        print(f"ERROR generating plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive strategy report")
    parser.add_argument('--trades-file', type=str, default='results/report_trades.parquet',
                        help='Path to trades parquet file')
    parser.add_argument('--output', type=str, default='reports/qml_strategy_validation_report.html',
                        help='Output HTML report path')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip trade data collection (use existing file)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP analysis')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip scatter plot generation')
    parser.add_argument('--symbols', type=str,
                        help='Comma-separated symbols for data collection')
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 9.6: COMPREHENSIVE STRATEGY REPORT GENERATOR")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    trades_path = PROJECT_ROOT / args.trades_file
    output_path = PROJECT_ROOT / args.output
    reports_dir = output_path.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or collect trade data
    if args.skip_collection and trades_path.exists():
        print(f"\nLoading existing trades from: {trades_path}")
        trades_df = pd.read_parquet(trades_path)
        print(f"Loaded {len(trades_df)} trades")
    else:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
            symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
        else:
            symbols = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
            ]
        trades_df = run_data_collection(symbols, trades_path, verbose=True)

    if len(trades_df) == 0:
        print("\nERROR: No trade data available. Cannot generate report.")
        sys.exit(1)

    # Step 2: SHAP analysis
    feature_importance = None
    shap_dir = reports_dir / 'shap'

    if not args.skip_shap:
        feature_importance = run_shap_analysis(trades_df, shap_dir, verbose=True)
    else:
        print("\nSkipping SHAP analysis (--skip-shap)")
        # Try to load existing
        importance_file = shap_dir / 'feature_importance.json'
        if importance_file.exists():
            with open(importance_file) as f:
                data = json.load(f)
            feature_importance = pd.DataFrame(data.get('feature_importance', []))
            print(f"Loaded existing feature importance from: {importance_file}")

    # Step 3: Feature scatter plots
    if not args.skip_plots:
        run_scatter_plots(trades_df, reports_dir, verbose=True)
    else:
        print("\nSkipping scatter plots (--skip-plots)")

    # Step 4: Load validation results
    print("\n" + "=" * 70)
    print("STEP 4: LOADING VALIDATION RESULTS")
    print("=" * 70)

    validation_results = find_latest_validation_results(PROJECT_ROOT / "results")
    print(f"Found {len(validation_results)} validation results")
    for name in validation_results:
        print(f"  - {name}")

    # Step 5: Calculate metrics
    print("\n" + "=" * 70)
    print("STEP 5: CALCULATING METRICS")
    print("=" * 70)

    metrics = calculate_metrics(trades_df)
    print(f"\nTotal Trades:   {metrics['total_trades']}")
    print(f"Win Rate:       {metrics['win_rate']:.1%}")
    print(f"Profit Factor:  {metrics['profit_factor']:.2f}")
    print(f"Expectancy:     {metrics['expectancy']:.2f}R")
    print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")

    # Step 6: Generate HTML report
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING HTML REPORT")
    print("=" * 70)

    generator = StrategyReportGenerator()

    # Find SHAP and correlation images
    shap_image = shap_dir / 'shap_summary_beeswarm.png'
    corr_image = reports_dir / 'feature_correlation_heatmap.png'

    html = generator.generate_report(
        trades_df=trades_df,
        metrics=metrics,
        validation_results=validation_results,
        shap_image_path=str(shap_image) if shap_image.exists() else None,
        correlation_image_path=str(corr_image) if corr_image.exists() else None,
        feature_importance=feature_importance,
    )

    saved_path = generator.save_report(html, str(output_path))

    print(f"\nReport saved to: {saved_path}")
    print(f"File size: {saved_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOpen report: {output_path}")


if __name__ == "__main__":
    main()
