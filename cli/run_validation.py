#!/usr/bin/env python3
"""
CLI Validation Runner
=====================
Command-line entry point for running VRD validation suite.

This module:
1. Loads backtest results (from previous run or runs fresh backtest)
2. Runs validation suite (Permutation, Monte Carlo, Bootstrap)
3. Outputs "Forensic Verdict" with statistical conclusions
4. Generates enhanced dossier with validation metrics

Usage:
    python -m cli.run_validation
    python -m cli.run_validation --run-id abc123
    python -m cli.run_validation --fresh --symbol BTCUSDT --timeframe 4h
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import (
    ValidationSuite,
    ValidationStatus,
    PermutationTest,
    MonteCarloSim,
    BootstrapResample,
    run_validation_suite
)
from src.reporting.storage import ExperimentLogger
from src.reporting.dossier import DossierGenerator


def load_previous_backtest(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load previous backtest results from database.
    
    Args:
        run_id: Run ID to load
    
    Returns:
        Dictionary with config and results, or None if not found
    """
    logger = ExperimentLogger()
    run = logger.get_run(run_id)
    
    if run is None:
        return None
    
    # Parse config
    config = json.loads(run['config_json'])
    
    # Load trades from CSV if available
    trades = []
    strategy = run.get('strategy_name', 'atr')
    trades_path = PROJECT_ROOT / "results" / strategy / f"{run_id}_trades.csv"
    
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
        trades = trades_df.to_dict('records')
    
    # Reconstruct results dictionary
    results = {
        'run_id': run_id,
        'net_profit_pct': run.get('pnl_percent', 0),
        'net_profit': run.get('pnl_usd', 0),
        'max_drawdown': run.get('max_drawdown', 0),
        'win_rate': run.get('win_rate', 0),
        'profit_factor': run.get('profit_factor', 0),
        'sharpe_ratio': run.get('sharpe_ratio', 0),
        'total_trades': run.get('total_trades', 0),
        'winning_trades': run.get('winning_trades', 0),
        'losing_trades': run.get('losing_trades', 0),
        'trades': trades,
    }
    
    return {
        'config': config,
        'results': results,
        'trades': trades,
        'run_id': run_id
    }


def run_fresh_backtest(symbol: str, timeframe: str, **kwargs) -> Dict[str, Any]:
    """
    Run a fresh backtest and return results.
    
    Args:
        symbol: Trading symbol
        timeframe: Candle timeframe
        **kwargs: Additional config options
    
    Returns:
        Dictionary with config, results, trades
    """
    # Import here to avoid circular imports
    from cli.run_backtest import run_backtest, BacktestConfig
    
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        **kwargs
    )
    
    results = run_backtest(config)
    
    trades = []
    if results.get('trades'):
        trades = [t.to_dict() for t in results['trades']]
    
    return {
        'config': asdict(config),
        'results': results,
        'trades': trades,
        'run_id': results.get('run_id')
    }


def print_forensic_verdict(suite: ValidationSuite) -> None:
    """
    Print the final forensic verdict with dramatic formatting.
    """
    print("\n" + "=" * 70)
    print("  🔬 FORENSIC VERDICT")
    print("=" * 70)
    
    # Overall status
    status_icons = {
        ValidationStatus.PASS: "✅ PASS",
        ValidationStatus.WARN: "⚠️  CAUTION",
        ValidationStatus.FAIL: "❌ FAIL",
        ValidationStatus.ERROR: "💥 ERROR"
    }
    
    overall = status_icons.get(suite.overall_status, "❓ UNKNOWN")
    print(f"\n  Overall Assessment: {overall}\n")
    
    # Individual results
    print("-" * 70)
    
    for result in suite.results:
        icon = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.WARN: "⚠️",
            ValidationStatus.FAIL: "❌",
            ValidationStatus.ERROR: "💥"
        }.get(result.status, "❓")
        
        print(f"\n  {icon} {result.validator_name.upper()}")
        print(f"     {result.interpretation}")
        
        # Key metrics
        if result.p_value is not None:
            print(f"     • p-value: {result.p_value:.4f}")
        
        for key, val in result.metrics.items():
            if key.startswith('sharpe') or key.startswith('var') or key.startswith('risk'):
                if isinstance(val, float):
                    print(f"     • {key}: {val:.4f}")
                else:
                    print(f"     • {key}: {val}")
    
    print("\n" + "-" * 70)
    
    # Final recommendation
    if suite.overall_status == ValidationStatus.PASS:
        print("\n  📊 CONCLUSION: Edge appears statistically valid.")
        print("     Proceed with paper trading before live deployment.\n")
    elif suite.overall_status == ValidationStatus.WARN:
        print("\n  ⚠️  CONCLUSION: Results are inconclusive.")
        print("     More data or parameter tuning recommended.\n")
    else:
        print("\n  ❌ CONCLUSION: Edge is NOT statistically valid.")
        print("     Do NOT deploy. Rethink strategy parameters.\n")
    
    print("=" * 70 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run VRD 2.0 validation suite on backtest results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--run-id", "-r",
        help="Run ID of previous backtest to validate"
    )
    source_group.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="Run a fresh backtest first"
    )
    
    # Fresh backtest options
    parser.add_argument(
        "--symbol", "-s",
        default="BTCUSDT",
        help="Trading symbol (for fresh backtest)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="4h",
        help="Candle timeframe (for fresh backtest)"
    )
    
    # Validator options
    parser.add_argument(
        "--validators", "-v",
        nargs="+",
        choices=["permutation_test", "monte_carlo", "bootstrap"],
        default=["permutation_test", "monte_carlo", "bootstrap"],
        help="Validators to run"
    )
    
    # Configuration
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for permutation test"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=1000,
        help="Number of simulations for Monte Carlo"
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Significance level for permutation test"
    )
    
    # Output options
    parser.add_argument(
        "--generate-dossier",
        action="store_true",
        default=True,
        help="Generate HTML dossier with validation results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🔬 VRD 2.0 VALIDATION SUITE")
    print("=" * 70)
    
    # Step 1: Load or run backtest
    data = None
    
    if args.run_id:
        print(f"\n📂 Loading previous backtest: {args.run_id}")
        data = load_previous_backtest(args.run_id)
        
        if data is None:
            print(f"❌ Run ID '{args.run_id}' not found in experiments database.")
            print("   Use 'python -m cli.run_backtest' to create a new run first.")
            sys.exit(1)
        
        print(f"   Loaded {len(data['trades'])} trades")
    
    elif args.fresh:
        print(f"\n🧪 Running fresh backtest: {args.symbol} {args.timeframe}")
        data = run_fresh_backtest(args.symbol, args.timeframe)
        print(f"   Generated {len(data['trades'])} trades")
    
    else:
        # Default: use most recent run
        print("\n📂 Loading most recent backtest...")
        logger = ExperimentLogger()
        recent = logger.get_recent_runs(limit=1)
        
        if not recent:
            print("❌ No previous backtests found.")
            print("   Run 'python -m cli.run_backtest' first, or use --fresh flag.")
            sys.exit(1)
        
        run_id = recent[0]['run_id']
        print(f"   Found: {run_id}")
        data = load_previous_backtest(run_id)
        
        if data is None:
            print("❌ Could not load backtest data.")
            sys.exit(1)
        
        print(f"   Loaded {len(data['trades'])} trades")
    
    # Step 2: Build validator config
    config = {
        'permutation_test': {
            'n_permutations': args.n_permutations,
            'significance_level': args.significance
        },
        'monte_carlo': {
            'n_simulations': args.n_simulations
        },
        'bootstrap': {
            'n_resamples': args.n_permutations
        }
    }
    
    # Step 3: Run validation suite
    print(f"\n🔬 Running validation suite...")
    print(f"   Validators: {', '.join(args.validators)}")
    
    suite = run_validation_suite(
        backtest_result=data['results'],
        trades=data['trades'],
        validators=args.validators,
        config=config
    )
    
    # Step 4: Print forensic verdict
    print_forensic_verdict(suite)
    
    # Step 5: Generate enhanced dossier
    if args.generate_dossier:
        print("📋 Generating validation dossier...")
        
        generator = DossierGenerator()
        
        # Convert trades to DataFrame if needed
        trades_df = None
        if data['trades']:
            trades_df = pd.DataFrame(data['trades'])
        
        report_path = generator.generate_html(
            run_id=data.get('run_id', 'validation'),
            config=data['config'],
            results=data['results'],
            trades_df=trades_df,
            strategy_name='validation',
            validation_suite=suite
        )
        
        print(f"   Report: file://{report_path}")
    
    print("✅ Validation complete!\n")


if __name__ == "__main__":
    main()
