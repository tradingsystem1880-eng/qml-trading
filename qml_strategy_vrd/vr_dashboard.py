#!/usr/bin/env python3
"""
VRD DASHBOARD - Versioned Research Database Query Tool
======================================================
List, compare, and query experiments in the VRD.

Usage:
    python vr_dashboard.py list                              # List all experiments
    python vr_dashboard.py compare EXP_ID_1 EXP_ID_2         # Compare two experiments
    python vr_dashboard.py query --sharpe-min 2.0            # Query by metrics
    python vr_dashboard.py show EXP_ID                       # Show experiment details
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


# VRD root directory (relative to this script)
VRD_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = VRD_ROOT / "experiments"
DETECTION_LOGIC_DIR = VRD_ROOT / "detection_logic"


def load_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment metadata and metrics."""
    exp_path = EXPERIMENTS_DIR / exp_id
    
    if not exp_path.exists():
        return None
    
    data = {"id": exp_id, "path": str(exp_path)}
    
    # Load meta.json
    meta_path = exp_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["meta"] = json.load(f)
    
    # Load metrics.json
    metrics_path = exp_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    
    # Check for trades.csv
    trades_path = exp_path / "trades.csv"
    data["has_trades"] = trades_path.exists()
    
    return data


def list_experiments() -> List[Dict[str, Any]]:
    """List all experiments in the VRD."""
    experiments = []
    
    if not EXPERIMENTS_DIR.exists():
        print("⚠️  No experiments directory found")
        return experiments
    
    for exp_path in sorted(EXPERIMENTS_DIR.iterdir()):
        if exp_path.is_dir():
            exp = load_experiment(exp_path.name)
            if exp:
                experiments.append(exp)
    
    return experiments


def list_detection_logic() -> List[Dict[str, Any]]:
    """List all versioned detection logic."""
    versions = []
    
    if not DETECTION_LOGIC_DIR.exists():
        return versions
    
    for version_path in sorted(DETECTION_LOGIC_DIR.iterdir()):
        if version_path.is_dir():
            params_path = version_path / "params.json"
            if params_path.exists():
                with open(params_path) as f:
                    params = json.load(f)
                    versions.append({
                        "version": params.get("version", version_path.name),
                        "name": params.get("name", ""),
                        "description": params.get("description", ""),
                        "path": str(version_path)
                    })
    
    return versions


def cmd_list(args):
    """Handle 'list' command."""
    print("\n" + "="*80)
    print("  📁 VRD EXPERIMENT REGISTRY")
    print("="*80)
    
    experiments = list_experiments()
    
    if not experiments:
        print("\n  No experiments found.\n")
        return
    
    print(f"\n  {'ID':<45} {'Symbol':<12} {'Logic':<15} {'Trades':<8} {'Sharpe':<8}")
    print("  " + "-"*75)
    
    for exp in experiments:
        exp_id = exp["id"]
        meta = exp.get("meta", {})
        metrics = exp.get("metrics", {})
        
        symbol = meta.get("symbol", "N/A")
        logic = meta.get("detection_logic_version", "N/A")
        trades = metrics.get("total_trades", "N/A")
        sharpe = metrics.get("sharpe_ratio", "N/A")
        
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        
        print(f"  {exp_id:<45} {symbol:<12} {logic:<15} {trades:<8} {sharpe_str:<8}")
    
    print()
    
    # Also list detection logic versions
    print("  📋 DETECTION LOGIC VERSIONS")
    print("  " + "-"*75)
    
    versions = list_detection_logic()
    for v in versions:
        print(f"  {v['version']:<12} {v['name']:<25} {v['description'][:40]}...")
    
    print("\n" + "="*80 + "\n")


def cmd_compare(args):
    """Handle 'compare' command."""
    exp1 = load_experiment(args.exp1)
    exp2 = load_experiment(args.exp2)
    
    if not exp1:
        print(f"❌ Experiment not found: {args.exp1}")
        return
    if not exp2:
        print(f"❌ Experiment not found: {args.exp2}")
        return
    
    print("\n" + "="*80)
    print("  📊 EXPERIMENT COMPARISON")
    print("="*80)
    
    # Get metrics
    m1 = exp1.get("metrics", {})
    m2 = exp2.get("metrics", {})
    
    # Header
    print(f"\n  {'Metric':<25} {args.exp1[:35]:<35} {args.exp2[:35]:<35}")
    print("  " + "-"*95)
    
    # Compare key metrics
    metrics_to_compare = [
        ("Total Trades", "total_trades", "{:d}", "{:d}"),
        ("Win Rate", "win_rate", "{:.1%}", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}", "{:.2f}"),
        ("Expectancy (R)", "expectancy_r", "{:+.2f}R", "{:+.2f}R"),
        ("Max Drawdown", "max_drawdown_pct", "{:.1f}%", "{:.1f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}", "{:.2f}"),
        ("Total Return", "total_return_pct", "{:.1f}%" if m1.get("total_return_pct") else "N/A", "{:.1f}%" if m2.get("total_return_pct") else "N/A"),
    ]
    
    for label, key, fmt1, fmt2 in metrics_to_compare:
        val1 = m1.get(key)
        val2 = m2.get(key)
        
        str1 = fmt1.format(val1) if val1 is not None and val1 != "N/A" else "N/A"
        str2 = fmt2.format(val2) if val2 is not None and val2 != "N/A" else "N/A"
        
        # Add delta indicator
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 and val2:
            if key in ["max_drawdown_pct"]:  # Lower is better
                indicator = "✅" if val2 < val1 else "⚠️" if val2 > val1 else ""
            else:  # Higher is better
                indicator = "✅" if val2 > val1 else "⚠️" if val2 < val1 else ""
            str2 = f"{str2} {indicator}"
        
        print(f"  {label:<25} {str1:<35} {str2:<35}")
    
    print("\n" + "="*80 + "\n")


def cmd_show(args):
    """Handle 'show' command."""
    exp = load_experiment(args.exp_id)
    
    if not exp:
        print(f"❌ Experiment not found: {args.exp_id}")
        return
    
    print("\n" + "="*80)
    print(f"  📋 EXPERIMENT: {args.exp_id}")
    print("="*80)
    
    # Metadata
    meta = exp.get("meta", {})
    print("\n  METADATA:")
    print(f"  {'Symbol:':<20} {meta.get('symbol', 'N/A')}")
    print(f"  {'Timeframe:':<20} {meta.get('timeframe', 'N/A')}")
    print(f"  {'Logic Version:':<20} {meta.get('detection_logic_version', 'N/A')}")
    print(f"  {'Data Range:':<20} {meta.get('data_range', {}).get('start', 'N/A')} to {meta.get('data_range', {}).get('end', 'N/A')}")
    print(f"  {'Run Timestamp:':<20} {meta.get('run_timestamp', 'N/A')}")
    print(f"  {'Notes:':<20} {meta.get('notes', 'N/A')}")
    
    # Metrics
    metrics = exp.get("metrics", {})
    print("\n  PERFORMANCE METRICS:")
    print(f"  {'Total Trades:':<20} {metrics.get('total_trades', 'N/A')}")
    print(f"  {'Win Rate:':<20} {metrics.get('win_rate', 0):.1%}")
    print(f"  {'Profit Factor:':<20} {metrics.get('profit_factor', 'N/A'):.2f}")
    print(f"  {'Expectancy:':<20} {metrics.get('expectancy_r', 0):+.2f}R")
    print(f"  {'Max Drawdown:':<20} {metrics.get('max_drawdown_pct', 'N/A'):.1f}%")
    print(f"  {'Sharpe Ratio:':<20} {metrics.get('sharpe_ratio', 'N/A'):.2f}")
    
    if metrics.get("total_return_pct"):
        print(f"  {'Total Return:':<20} {metrics.get('total_return_pct'):.1f}%")
    
    # Walk-forward if present
    if "walk_forward" in metrics:
        wf = metrics["walk_forward"]
        print("\n  WALK-FORWARD ANALYSIS:")
        print(f"  {'Folds:':<20} {wf.get('folds', 'N/A')}")
        print(f"  {'All Profitable:':<20} {'Yes ✅' if wf.get('all_profitable') else 'No'}")
        print(f"  {'Mean Win Rate:':<20} {wf.get('mean_win_rate', 0):.1%}")
    
    # Monte Carlo if present  
    if "monte_carlo" in metrics:
        mc = metrics["monte_carlo"]
        print("\n  MONTE CARLO ANALYSIS:")
        print(f"  {'Simulations:':<20} {mc.get('simulations', 'N/A')}")
        print(f"  {'P(Profitable):':<20} {mc.get('probability_profitable', 0):.0%}")
        print(f"  {'95% Max DD:':<20} {mc.get('p95_max_drawdown', 'N/A'):.1f}%")
    
    print("\n" + "="*80 + "\n")


def cmd_query(args):
    """Handle 'query' command."""
    experiments = list_experiments()
    
    filtered = []
    
    for exp in experiments:
        metrics = exp.get("metrics", {})
        
        # Apply filters
        if args.sharpe_min and metrics.get("sharpe_ratio", 0) < args.sharpe_min:
            continue
        if args.win_rate_min and metrics.get("win_rate", 0) < args.win_rate_min:
            continue
        if args.pf_min and metrics.get("profit_factor", 0) < args.pf_min:
            continue
        if args.symbol and exp.get("meta", {}).get("symbol") != args.symbol:
            continue
        if args.logic and args.logic not in exp.get("meta", {}).get("detection_logic_version", ""):
            continue
        
        filtered.append(exp)
    
    if not filtered:
        print("\n  No experiments match the query criteria.\n")
        return
    
    print(f"\n  📊 QUERY RESULTS ({len(filtered)} matches)")
    print("  " + "-"*75)
    print(f"  {'ID':<45} {'Sharpe':<10} {'WR':<10} {'PF':<10}")
    print("  " + "-"*75)
    
    for exp in filtered:
        metrics = exp.get("metrics", {})
        print(f"  {exp['id']:<45} {metrics.get('sharpe_ratio', 0):<10.2f} "
              f"{metrics.get('win_rate', 0):<10.1%} {metrics.get('profit_factor', 0):<10.2f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="VRD Dashboard - Query and compare experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.set_defaults(func=cmd_list)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two experiments")
    compare_parser.add_argument("exp1", help="First experiment ID")
    compare_parser.add_argument("exp2", help="Second experiment ID")
    compare_parser.set_defaults(func=cmd_compare)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")
    show_parser.set_defaults(func=cmd_show)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query experiments by criteria")
    query_parser.add_argument("--sharpe-min", type=float, help="Minimum Sharpe ratio")
    query_parser.add_argument("--win-rate-min", type=float, help="Minimum win rate (0-1)")
    query_parser.add_argument("--pf-min", type=float, help="Minimum profit factor")
    query_parser.add_argument("--symbol", type=str, help="Filter by symbol")
    query_parser.add_argument("--logic", type=str, help="Filter by logic version (partial match)")
    query_parser.set_defaults(func=cmd_query)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
