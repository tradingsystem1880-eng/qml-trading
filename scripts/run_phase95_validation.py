#!/usr/bin/env python3
"""
Phase 9.5: Complete Validation Suite Runner
===========================================
Runs all 6 validation tests and produces a consolidated report.

Tests Run:
1. Permutation Test - Statistical significance (PF in top 5%)
2. Monte Carlo Drawdown - Risk analysis (95% CI < 20%)
3. OOS Holdout - Out-of-sample performance (PF > 2.0, WR > 48%)
4. Parameter Sensitivity - Robustness (PF range < 1.5)
5. Stress Test - Market stress resilience (avg PF > 1.0)
6. Trade Correlation - Independence (|r| < 0.1)

Usage:
    python scripts/run_phase95_validation.py
    python scripts/run_phase95_validation.py --symbols BTCUSDT,ETHUSDT
    python scripts/run_phase95_validation.py --quick  # Faster with fewer iterations

Output:
    results/phase95_validation/validation_report_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_test(script_name: str, args: List[str], verbose: bool = True) -> Dict:
    """
    Run a validation test script and capture results.

    Returns:
        Dict with test name, status, and result file path
    """
    script_path = PROJECT_ROOT / "scripts" / script_name

    cmd = [sys.executable, str(script_path)] + args

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running: {script_name}")
        print("=" * 60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if verbose:
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")

        # Try to find the output file from stdout
        output_file = None
        for line in result.stdout.split('\n'):
            if "Results saved to:" in line:
                output_file = line.split("Results saved to:")[-1].strip()
                break

        return {
            "script": script_name,
            "success": result.returncode == 0,
            "output_file": output_file,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "script": script_name,
            "success": False,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "script": script_name,
            "success": False,
            "error": str(e),
        }


def load_test_results(output_file: str) -> Dict:
    """Load results from a test output file."""
    if not output_file or not Path(output_file).exists():
        return {}

    with open(output_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 9.5 validation suite")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer iterations)')
    parser.add_argument('--skip', type=str, help='Comma-separated tests to skip')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 9.5: COMPLETE VALIDATION SUITE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Build common args
    common_args = []
    if args.symbols:
        common_args.extend(['--symbols', args.symbols])
    if args.timeframe:
        common_args.extend(['--timeframe', args.timeframe])

    # Determine iterations
    if args.quick:
        perm_iterations = "200"
        mc_iterations = "1000"
    else:
        perm_iterations = "1000"
        mc_iterations = "10000"

    # Tests to run
    tests = [
        {
            "name": "Permutation Test",
            "script": "phase95_permutation_test.py",
            "args": ['--iterations', perm_iterations],
            "criterion": "Real PF in top 5% (p < 0.05)",
        },
        {
            "name": "Monte Carlo Drawdown",
            "script": "phase95_monte_carlo_drawdown.py",
            "args": ['--simulations', mc_iterations],
            "criterion": "95% CI max drawdown < 20%",
        },
        {
            "name": "OOS Holdout",
            "script": "phase95_oos_holdout.py",
            "args": [],
            "criterion": "OOS PF > 2.0, WR > 48%",
        },
        {
            "name": "Parameter Sensitivity",
            "script": "phase95_parameter_sensitivity.py",
            "args": [],
            "criterion": "PF range < 1.5 across ±20%",
        },
        {
            "name": "Stress Test",
            "script": "phase95_stress_test.py",
            "args": [],
            "criterion": "No event PF < 0.5, avg > 1.0",
        },
        {
            "name": "Trade Correlation",
            "script": "phase95_trade_correlation.py",
            "args": [],
            "criterion": "|autocorrelation| < 0.1",
        },
    ]

    # Filter skipped tests
    skip_list = []
    if args.skip:
        skip_list = [s.strip().lower() for s in args.skip.split(',')]

    results = []
    passed_count = 0
    failed_count = 0

    for test in tests:
        test_key = test['script'].replace('phase95_', '').replace('.py', '')
        if test_key in skip_list or test['name'].lower() in skip_list:
            print(f"\n⏭️  Skipping: {test['name']}")
            continue

        test_args = common_args + test['args']
        run_result = run_test(test['script'], test_args, verbose=True)

        # Load detailed results
        if run_result.get('output_file'):
            detailed = load_test_results(run_result['output_file'])
            passed = detailed.get('passed', False)
            verdict = detailed.get('verdict', 'UNKNOWN')
        else:
            passed = False
            verdict = 'ERROR'

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        results.append({
            "name": test['name'],
            "criterion": test['criterion'],
            "passed": passed,
            "verdict": verdict,
            "output_file": run_result.get('output_file'),
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUITE SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<30} {'Criterion':<35} {'Result':<10}")
    print("-" * 80)

    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{result['name']:<30} {result['criterion']:<35} {status:<10}")

    print(f"\n{'-' * 80}")
    print(f"Total: {passed_count} PASS, {failed_count} FAIL")

    # Overall verdict
    all_pass = failed_count == 0 and passed_count == len([t for t in tests if t['script'].replace('phase95_', '').replace('.py', '') not in skip_list])

    print(f"\n{'=' * 70}")
    print("OVERALL VERDICT")
    print("=" * 70)

    if all_pass:
        print(f"\n🎉 ALL TESTS PASSED - Strategy validated for forward testing!")
        print(f"\nNext Steps:")
        print(f"  1. Set up Bybit testnet account")
        print(f"  2. Run: python scripts/run_bybit_paper_trader.py scan --execute")
        print(f"  3. Monitor Phase 1 (50 trades @ 0.5% risk)")
    else:
        print(f"\n⚠️  SOME TESTS FAILED - Review results before proceeding")
        print(f"\nFailed tests:")
        for result in results:
            if not result['passed']:
                print(f"  - {result['name']}: {result['criterion']}")
        print(f"\nConsider investigating failures before forward testing.")

    # Save consolidated report
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase95_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"validation_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "config": {
            "symbols": args.symbols,
            "timeframe": args.timeframe,
            "quick_mode": args.quick,
        },
        "results": results,
        "summary": {
            "passed": passed_count,
            "failed": failed_count,
            "total": len(results),
            "all_pass": all_pass,
        },
        "verdict": "PASS" if all_pass else "FAIL",
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    # Return exit code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
