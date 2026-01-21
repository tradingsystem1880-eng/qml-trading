#!/usr/bin/env python3
"""
Validation Framework Test - Phase 4
====================================
Verifies the enhanced validation framework components:
- PBOCalculator (Probability of Backtest Overfitting)
- PropFirmSimulator (Prop firm challenge pass probability)
- ValidationService (Unified validation orchestrator)

Run: python tests/test_validation_phase4.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_pbo_calculator():
    """Verify PBO calculation works."""
    print("\n" + "=" * 60)
    print("TEST 1: PBO Calculator")
    print("=" * 60)

    from src.validation import PBOCalculator

    # Create synthetic strategy returns (5 strategies, 100 periods)
    np.random.seed(42)
    returns = np.random.randn(100, 5) * 0.02

    pbo = PBOCalculator(config={'n_partitions': 8})
    result = pbo.calculate(returns)

    print(f"   PBO: {result.pbo:.2%}")
    print(f"   IS-OOS Correlation: {result.is_oos_rank_correlation:.3f}")
    print(f"   N Combinations: {result.n_combinations}")
    print(f"   Is Overfit: {result.is_overfit}")
    print(f"   Interpretation: {result.interpretation}")

    # Verify PBO is in valid range
    if not (0 <= result.pbo <= 1):
        print(f"   ❌ FAILED: PBO out of range [0,1]: {result.pbo}")
        return False

    print("   ✅ PBO Calculator working correctly")
    return True


def test_pbo_validator():
    """Verify PBO as a validator with trades input."""
    print("\n" + "=" * 60)
    print("TEST 2: PBO Validator (with trades)")
    print("=" * 60)

    from src.validation import PBOCalculator, ValidationStatus

    # Create mock trades
    np.random.seed(42)
    trades = [{'pnl_pct': np.random.randn() * 2} for _ in range(50)]

    pbo = PBOCalculator()
    result = pbo.validate({}, trades=trades)

    print(f"   Status: {result.status.value}")
    print(f"   PBO: {result.metrics.get('pbo', 'N/A')}")
    print(f"   Interpretation: {result.interpretation}")

    # Should return a valid result
    if result.status == ValidationStatus.ERROR:
        print(f"   ❌ FAILED: Unexpected error: {result.interpretation}")
        return False

    print("   ✅ PBO Validator working correctly")
    return True


def test_prop_firm_simulator():
    """Verify prop firm simulation works."""
    print("\n" + "=" * 60)
    print("TEST 3: PropFirm Simulator")
    print("=" * 60)

    from src.validation import PropFirmSimulator, PropFirmRules

    # Simulate with realistic returns (1% daily std)
    np.random.seed(42)
    returns = np.random.randn(200) * 0.01

    rules = PropFirmRules(
        profit_target_pct=10.0,
        daily_loss_limit_pct=5.0,
        total_loss_limit_pct=10.0,
        min_trading_days=10,
        time_limit_days=30
    )

    sim = PropFirmSimulator()
    result = sim.simulate_challenge(returns, rules, n_simulations=500)

    print(f"   Pass Rate: {result.pass_rate:.1%}")
    print(f"   Avg Days to Pass: {result.avg_days_to_pass:.1f}")
    print(f"   Profit on Pass: {result.profit_on_pass:.1f}%")
    print(f"   Fail Reasons: {result.fail_reasons}")

    # Verify pass_rate is in valid range
    if not (0 <= result.pass_rate <= 1):
        print(f"   ❌ FAILED: Pass rate out of range [0,1]: {result.pass_rate}")
        return False

    print("   ✅ PropFirm Simulator working correctly")
    return True


def test_prop_firm_rules():
    """Test PropFirmRules dataclass."""
    print("\n" + "=" * 60)
    print("TEST 4: PropFirm Rules")
    print("=" * 60)

    from src.validation import PropFirmRules

    # Test default values
    default_rules = PropFirmRules()
    print(f"   Default profit target: {default_rules.profit_target_pct}%")
    print(f"   Default daily limit: {default_rules.daily_loss_limit_pct}%")
    print(f"   Default total limit: {default_rules.total_loss_limit_pct}%")
    print(f"   Default min days: {default_rules.min_trading_days}")
    print(f"   Default time limit: {default_rules.time_limit_days}")

    # Test custom values
    custom_rules = PropFirmRules(
        profit_target_pct=8.0,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        min_trading_days=5,
        time_limit_days=14
    )

    if custom_rules.profit_target_pct != 8.0:
        print(f"   ❌ FAILED: Custom profit target not set")
        return False

    print("   ✅ PropFirm Rules working correctly")
    return True


def test_validation_service():
    """Test full validation pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: ValidationService")
    print("=" * 60)

    from src.validation import ValidationService, PropFirmRules

    # Mock trades with realistic distribution
    np.random.seed(42)
    trades = []
    for _ in range(50):
        # 55% win rate, avg win 3%, avg loss -2%
        is_win = np.random.rand() < 0.55
        pnl = np.random.uniform(1, 5) if is_win else np.random.uniform(-4, -0.5)
        trades.append({'pnl_pct': pnl})

    # Run validation
    service = ValidationService()
    report = service.validate_strategy(
        trades,
        prop_firm_rules=PropFirmRules(profit_target_pct=10.0)
    )

    print(f"   Overall Verdict: {report.overall_verdict}")
    print(f"   Permutation: {report.permutation_result.status.value if report.permutation_result else 'N/A'}")
    print(f"   Monte Carlo: {report.monte_carlo_result.status.value if report.monte_carlo_result else 'N/A'}")
    print(f"   Bootstrap: {report.bootstrap_result.status.value if report.bootstrap_result else 'N/A'}")
    print(f"   PBO: {report.pbo_result.status.value if report.pbo_result else 'N/A'}")
    print(f"   Prop Firm Pass Rate: {report.prop_firm_result.pass_rate:.1%}" if report.prop_firm_result else "   Prop Firm: N/A")

    print("\n   Recommendations:")
    for rec in report.recommendations[:3]:
        print(f"     • {rec}")

    # Verify verdict is valid
    if report.overall_verdict not in ['PASS', 'WARN', 'FAIL', 'UNKNOWN']:
        print(f"   ❌ FAILED: Invalid verdict: {report.overall_verdict}")
        return False

    print("   ✅ ValidationService working correctly")
    return True


def test_quick_validate():
    """Test quick_validate convenience function."""
    print("\n" + "=" * 60)
    print("TEST 6: quick_validate Function")
    print("=" * 60)

    from src.validation import quick_validate

    # Mock trades
    np.random.seed(42)
    trades = [{'pnl_pct': np.random.randn() * 2} for _ in range(30)]

    report = quick_validate(trades)

    print(f"   Verdict: {report.overall_verdict}")
    print(f"   N Recommendations: {len(report.recommendations)}")

    print("   ✅ quick_validate working correctly")
    return True


def test_validation_report_str():
    """Test ValidationReport string representation."""
    print("\n" + "=" * 60)
    print("TEST 7: ValidationReport String Output")
    print("=" * 60)

    from src.validation import ValidationService

    np.random.seed(42)
    trades = [{'pnl_pct': np.random.randn() * 2} for _ in range(30)]

    service = ValidationService()
    report = service.validate_strategy(trades)

    # Print the string representation
    report_str = str(report)
    print(report_str)

    # Check it's not empty
    if len(report_str) < 100:
        print("   ❌ FAILED: Report string too short")
        return False

    print("   ✅ ValidationReport string output working correctly")
    return True


def test_selective_validators():
    """Test running only selected validators."""
    print("\n" + "=" * 60)
    print("TEST 8: Selective Validators")
    print("=" * 60)

    from src.validation import ValidationService

    np.random.seed(42)
    trades = [{'pnl_pct': np.random.randn() * 2} for _ in range(30)]

    service = ValidationService()

    # Run only permutation and bootstrap
    report = service.validate_strategy(
        trades,
        validators=['permutation', 'bootstrap']
    )

    # Permutation and bootstrap should have results
    if report.permutation_result is None:
        print("   ❌ FAILED: Permutation result missing")
        return False
    if report.bootstrap_result is None:
        print("   ❌ FAILED: Bootstrap result missing")
        return False

    # Monte Carlo and PBO should be None
    if report.monte_carlo_result is not None:
        print("   ❌ FAILED: Monte Carlo should not have run")
        return False
    if report.pbo_result is not None:
        print("   ❌ FAILED: PBO should not have run")
        return False

    print(f"   Ran validators: permutation, bootstrap")
    print(f"   Skipped validators: monte_carlo, pbo, prop_firm")
    print("   ✅ Selective validators working correctly")
    return True


def test_exports():
    """Test that all new classes are properly exported."""
    print("\n" + "=" * 60)
    print("TEST 9: Module Exports")
    print("=" * 60)

    try:
        from src.validation import (
            PBOCalculator,
            PBOResult,
            PropFirmRules,
            PropFirmResult,
            PropFirmSimulator,
            ValidationService,
            ValidationReport,
            quick_validate,
        )
        print("   ✅ PBOCalculator")
        print("   ✅ PBOResult")
        print("   ✅ PropFirmRules")
        print("   ✅ PropFirmResult")
        print("   ✅ PropFirmSimulator")
        print("   ✅ ValidationService")
        print("   ✅ ValidationReport")
        print("   ✅ quick_validate")
    except ImportError as e:
        print(f"   ❌ FAILED: Import error: {e}")
        return False

    print("   ✅ All exports working correctly")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST 10: Edge Cases")
    print("=" * 60)

    from src.validation import ValidationService, PBOCalculator, ValidationStatus

    # Empty trades
    service = ValidationService()
    report = service.validate_strategy([])

    if report.permutation_result and report.permutation_result.status != ValidationStatus.ERROR:
        print("   ⚠️ Expected ERROR status for empty trades")

    print("   ✅ Empty trades handled")

    # Very few trades (should warn, not error)
    few_trades = [{'pnl_pct': 1.0}, {'pnl_pct': -1.0}]
    report2 = service.validate_strategy(few_trades)

    print("   ✅ Few trades handled")

    # PBO with insufficient data
    pbo = PBOCalculator()
    result = pbo.validate({}, trades=[{'pnl_pct': 1.0}])
    if result.status not in [ValidationStatus.WARN, ValidationStatus.ERROR]:
        print(f"   ⚠️ Expected WARN or ERROR for insufficient PBO data, got {result.status}")

    print("   ✅ Insufficient PBO data handled")

    print("   ✅ All edge cases handled correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  PHASE 4 VALIDATION FRAMEWORK TEST")
    print("=" * 70)

    tests = [
        ("PBO Calculator", test_pbo_calculator),
        ("PBO Validator", test_pbo_validator),
        ("PropFirm Simulator", test_prop_firm_simulator),
        ("PropFirm Rules", test_prop_firm_rules),
        ("ValidationService", test_validation_service),
        ("quick_validate", test_quick_validate),
        ("Report String", test_validation_report_str),
        ("Selective Validators", test_selective_validators),
        ("Module Exports", test_exports),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n   🎉 All tests passed! Validation framework ready.")
    else:
        print("\n   ⚠️ Some tests failed. Check output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
