"""
Phase 9.5: Parameter Sensitivity Analysis
==========================================
Tests robustness of strategy to parameter variations.

Methodology:
1. Vary key parameters by ±20%
2. Run backtest for each variation
3. Calculate PF range across all variations
4. Check if edge persists across parameter space

Success Criteria: PF range < 1.5 across all ±20% variations

Key Parameters Tested:
- TP ATR multiplier (tp_atr_mult)
- SL ATR multiplier (sl_atr_mult)
- Min risk/reward ratio (min_risk_reward)
- Max bars held (max_bars_held)

Usage:
    python scripts/phase95_parameter_sensitivity.py
    python scripts/phase95_parameter_sensitivity.py --variation 0.2
    python scripts/phase95_parameter_sensitivity.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase95_utils import (
    run_full_backtest,
    DEFAULT_SYMBOLS,
    TRADE_CONFIG,
    ValidationResult,
)
from src.optimization.trade_simulator import TradeManagementConfig


# Parameters to vary and their baseline values
PARAMETERS = {
    "tp_atr_mult": {
        "baseline": 4.6,
        "description": "Take profit ATR multiplier",
        "min": 3.0,
        "max": 6.0,
    },
    "sl_atr_mult": {
        "baseline": 1.0,
        "description": "Stop loss ATR multiplier",
        "min": 0.5,
        "max": 2.0,
    },
    "min_risk_reward": {
        "baseline": 3.0,
        "description": "Minimum risk/reward ratio",
        "min": 2.0,
        "max": 5.0,
    },
    "max_bars_held": {
        "baseline": 100,
        "description": "Maximum bars to hold position",
        "min": 50,
        "max": 200,
    },
}


def create_config_variation(param_name: str, value: float) -> TradeManagementConfig:
    """Create a TradeManagementConfig with one parameter varied."""
    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=TRADE_CONFIG.tp_atr_mult,
        sl_atr_mult=TRADE_CONFIG.sl_atr_mult,
        trailing_mode="none",
        max_bars_held=TRADE_CONFIG.max_bars_held,
        min_risk_reward=TRADE_CONFIG.min_risk_reward,
    )

    # Override the specific parameter
    if param_name == "tp_atr_mult":
        config.tp_atr_mult = value
    elif param_name == "sl_atr_mult":
        config.sl_atr_mult = value
    elif param_name == "min_risk_reward":
        config.min_risk_reward = value
    elif param_name == "max_bars_held":
        config.max_bars_held = int(value)

    return config


def run_sensitivity_analysis(
    symbols: List[str],
    timeframe: str,
    variation_pct: float = 0.2,
    verbose: bool = True,
) -> Dict:
    """
    Run sensitivity analysis across all parameters.

    Args:
        symbols: List of symbols to test
        timeframe: Timeframe to use
        variation_pct: Percentage variation (e.g., 0.2 for ±20%)
        verbose: Print progress

    Returns:
        Dict with results for each parameter
    """
    results = {}

    for param_name, param_info in PARAMETERS.items():
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Testing: {param_name} ({param_info['description']})")
            print(f"{'=' * 50}")

        baseline = param_info['baseline']
        min_bound = param_info['min']
        max_bound = param_info['max']

        # Calculate variation values
        low_value = max(baseline * (1 - variation_pct), min_bound)
        high_value = min(baseline * (1 + variation_pct), max_bound)

        variations = {
            f"-{variation_pct*100:.0f}%": low_value,
            "baseline": baseline,
            f"+{variation_pct*100:.0f}%": high_value,
        }

        param_results = {
            "parameter": param_name,
            "description": param_info['description'],
            "baseline_value": baseline,
            "variations": {},
        }

        pf_values = []

        for var_name, var_value in variations.items():
            if verbose:
                print(f"\n  {var_name}: {var_value:.2f}")

            config = create_config_variation(param_name, var_value)
            trades, metrics = run_full_backtest(
                symbols,
                timeframe,
                config=config,
                verbose=False,
            )

            pf = metrics['profit_factor']
            pf_values.append(pf)

            param_results['variations'][var_name] = {
                "value": var_value,
                "trades": metrics['total_trades'],
                "win_rate": metrics['win_rate'],
                "profit_factor": pf,
                "expectancy": metrics['expectancy'],
            }

            if verbose:
                print(f"    Trades: {metrics['total_trades']}, "
                      f"WR: {metrics['win_rate']:.1%}, "
                      f"PF: {pf:.2f}, "
                      f"Exp: {metrics['expectancy']:.2f}R")

        # Calculate range and stability metrics
        pf_range = max(pf_values) - min(pf_values)
        pf_mean = np.mean(pf_values)
        pf_std = np.std(pf_values)
        pf_cv = pf_std / pf_mean if pf_mean > 0 else float('inf')

        param_results['pf_range'] = pf_range
        param_results['pf_mean'] = pf_mean
        param_results['pf_std'] = pf_std
        param_results['pf_cv'] = pf_cv

        results[param_name] = param_results

        if verbose:
            print(f"\n  Summary: PF range = {pf_range:.2f} "
                  f"(min={min(pf_values):.2f}, max={max(pf_values):.2f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity analysis")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--variation', type=float, default=0.2, help='Variation percentage (default 0.2 = ±20%)')
    parser.add_argument('--threshold', type=float, default=1.5, help='Max acceptable PF range')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.5: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Variation: ±{args.variation*100:.0f}%")
    print(f"Threshold: PF range < {args.threshold:.1f}")

    print(f"\nParameters to test:")
    for name, info in PARAMETERS.items():
        print(f"  - {name}: {info['baseline']} ({info['description']})")

    # Run baseline first
    print(f"\n{'=' * 70}")
    print("BASELINE BACKTEST")
    print(f"{'=' * 70}\n")

    baseline_trades, baseline_metrics = run_full_backtest(
        symbols, args.timeframe, verbose=True
    )

    print(f"\nBaseline Results:")
    print(f"  Trades:        {baseline_metrics['total_trades']}")
    print(f"  Win Rate:      {baseline_metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {baseline_metrics['profit_factor']:.2f}")
    print(f"  Expectancy:    {baseline_metrics['expectancy']:.2f}R")

    # Run sensitivity analysis
    print(f"\n{'=' * 70}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")

    sensitivity_results = run_sensitivity_analysis(
        symbols, args.timeframe, args.variation, verbose=True
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Parameter':<20} {'PF Range':>10} {'PF Mean':>10} {'PF CV':>10} {'Status':>10}")
    print("-" * 65)

    all_stable = True
    max_pf_range = 0

    for param_name, param_results in sensitivity_results.items():
        pf_range = param_results['pf_range']
        pf_mean = param_results['pf_mean']
        pf_cv = param_results['pf_cv']

        is_stable = pf_range < args.threshold
        status = "STABLE" if is_stable else "UNSTABLE"

        if not is_stable:
            all_stable = False
        if pf_range > max_pf_range:
            max_pf_range = pf_range

        print(f"{param_name:<20} {pf_range:>10.2f} {pf_mean:>10.2f} {pf_cv:>9.1%} {status:>10}")

    # Verdict
    passed = all_stable

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if passed:
        print(f"\n✅ PASS: Strategy is robust to parameter variations")
        print(f"   Max PF range ({max_pf_range:.2f}) < threshold ({args.threshold:.1f})")
        print(f"   All parameters show stable performance within ±{args.variation*100:.0f}%")
        print(f"   This suggests the edge is not fragile or overfit")
    else:
        print(f"\n❌ FAIL: Strategy is sensitive to parameter variations")
        print(f"   Max PF range ({max_pf_range:.2f}) >= threshold ({args.threshold:.1f})")

        # List unstable parameters
        unstable = [name for name, r in sensitivity_results.items()
                    if r['pf_range'] >= args.threshold]
        print(f"   Unstable parameters: {', '.join(unstable)}")
        print(f"   Consider narrowing parameter bounds or re-optimizing")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"parameter_sensitivity_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "test": "parameter_sensitivity",
        "config": {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "variation_pct": args.variation,
            "threshold": args.threshold,
        },
        "baseline_metrics": {
            "total_trades": baseline_metrics['total_trades'],
            "win_rate": baseline_metrics['win_rate'],
            "profit_factor": baseline_metrics['profit_factor'],
            "expectancy": baseline_metrics['expectancy'],
        },
        "sensitivity_results": sensitivity_results,
        "max_pf_range": max_pf_range,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return ValidationResult(
        test_name="parameter_sensitivity",
        passed=passed,
        metric_value=max_pf_range,
        threshold=args.threshold,
        details=results,
    )


if __name__ == "__main__":
    main()
