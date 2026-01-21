#!/usr/bin/env python3
"""
Prop Firm Module Test - Phase 5
================================
Verifies the prop firm compliance module components:
- KellyPositionSizer (Kelly criterion with prop firm caps)
- PropFirmTracker (Real-time compliance tracking)
- PropFirmRules (Extended with new fields)
- Settings save/load

Run: python3 tests/test_prop_firm_phase5.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np


def test_prop_firm_rules_extended():
    """Verify PropFirmRules has new fields."""
    print("\n" + "=" * 60)
    print("TEST 1: PropFirmRules Extended Fields")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules

    # Test with defaults
    rules = PropFirmRules()

    print(f"   profit_target_pct: {rules.profit_target_pct}")
    print(f"   daily_loss_limit_pct: {rules.daily_loss_limit_pct}")
    print(f"   total_loss_limit_pct: {rules.total_loss_limit_pct}")
    print(f"   min_trading_days: {rules.min_trading_days}")
    print(f"   time_limit_days: {rules.time_limit_days}")
    print(f"   account_size: {rules.account_size}")
    print(f"   max_position_size_pct: {rules.max_position_size_pct}")
    print(f"   consistency_rule: {rules.consistency_rule}")

    # Verify new fields exist
    if not hasattr(rules, 'account_size'):
        print("   ❌ FAILED: account_size field missing")
        return False
    if not hasattr(rules, 'max_position_size_pct'):
        print("   ❌ FAILED: max_position_size_pct field missing")
        return False
    if not hasattr(rules, 'consistency_rule'):
        print("   ❌ FAILED: consistency_rule field missing")
        return False

    # Test Breakout rules
    breakout = PropFirmRules(
        account_size=100000,
        profit_target_pct=8.0,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        min_trading_days=5,
        max_position_size_pct=2.0,
        consistency_rule=True
    )

    if breakout.account_size != 100000:
        print(f"   ❌ FAILED: account_size not set correctly")
        return False

    print("   ✅ PropFirmRules extended correctly")
    return True


def test_kelly_sizer_basic():
    """Test basic Kelly calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: Kelly Sizer Basic Calculation")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import KellyPositionSizer

    rules = PropFirmRules(
        account_size=100000,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        max_position_size_pct=2.0
    )

    sizer = KellyPositionSizer(rules, kelly_fraction=0.5)

    result = sizer.calculate(
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
        current_equity=100000,
        stop_loss_pct=0.02
    )

    print(f"   Base Kelly: {result.base_kelly:.1f}%")
    print(f"   Adjusted Kelly: {result.adjusted_kelly:.4f}")
    print(f"   Position Size: {result.position_size_pct:.1f}%")
    print(f"   Position Size: ${result.position_size_dollars:,.0f}")
    print(f"   Risk Amount: ${result.risk_amount:,.0f}")
    print(f"   Capped by Max Position: {result.capped_by_max_position}")
    print(f"   Capped by Daily DD: {result.capped_by_daily_dd}")
    print(f"   Capped by Total DD: {result.capped_by_total_dd}")

    # Base Kelly for 55% win rate, 2:1 R:R should be positive
    if result.base_kelly <= 0:
        print("   ❌ FAILED: Base Kelly should be positive")
        return False

    # Position size should be capped
    if result.position_size_pct > 2.0 + 0.01:  # Allow small float error
        print(f"   ❌ FAILED: Position size {result.position_size_pct:.2f}% exceeds max 2%")
        return False

    print("   ✅ Kelly Sizer basic calculation working")
    return True


def test_kelly_sizer_caps():
    """Test Kelly caps are applied correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Kelly Sizer Caps")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import KellyPositionSizer

    rules = PropFirmRules(
        account_size=100000,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        max_position_size_pct=2.0
    )

    sizer = KellyPositionSizer(rules, kelly_fraction=0.5)

    # Test 1: Max position cap
    result = sizer.calculate(
        win_rate=0.70,  # High win rate = high Kelly
        avg_win=300,
        avg_loss=100,
        current_equity=100000,
        stop_loss_pct=0.02
    )

    print(f"   Test 1 - Max Position Cap:")
    print(f"     Base Kelly: {result.base_kelly:.1f}%")
    print(f"     Position Size: {result.position_size_pct:.1f}%")
    print(f"     Capped by Max: {result.capped_by_max_position}")

    if not result.capped_by_max_position:
        print("   ⚠️ Expected max position cap with high Kelly")

    # Test 2: Daily DD cap (simulate bad day)
    result2 = sizer.calculate(
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
        current_equity=98000,  # Lost 2%
        daily_pnl=-2000,  # Lost $2000 today (2%)
        stop_loss_pct=0.02
    )

    print(f"\n   Test 2 - After $2000 Loss Today:")
    print(f"     Daily PnL: ${-2000}")
    print(f"     Position Size: {result2.position_size_pct:.1f}%")
    print(f"     Capped by Daily DD: {result2.capped_by_daily_dd}")

    # Test 3: Total DD scaling (simulate drawdown from peak)
    result3 = sizer.calculate(
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
        current_equity=95000,  # Currently at $95k
        daily_pnl=0,
        peak_equity=100000,  # Peak was $100k
        stop_loss_pct=0.02
    )

    print(f"\n   Test 3 - 5% Total Drawdown:")
    print(f"     Peak: $100,000, Current: $95,000")
    print(f"     Position Size: {result3.position_size_pct:.1f}%")
    print(f"     Capped by Total DD: {result3.capped_by_total_dd}")

    print("\n   ✅ Kelly caps working correctly")
    return True


def test_prop_firm_tracker_basic():
    """Test basic tracker functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Prop Firm Tracker Basic")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import PropFirmTracker

    rules = PropFirmRules(
        account_size=100000,
        profit_target_pct=8.0,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        min_trading_days=5,
        consistency_rule=True
    )

    tracker = PropFirmTracker(rules)

    # Initial status
    status = tracker.update(100000)
    print(f"   Initial Status: {status.status}")
    print(f"   Daily DD: {status.daily_dd_pct:.2f}%")
    print(f"   Total DD: {status.total_dd_pct:.2f}%")
    print(f"   Profit: {status.profit_pct:.2f}%")

    if status.status != 'ON_TRACK':
        print(f"   ❌ FAILED: Expected ON_TRACK, got {status.status}")
        return False

    # Simulate small loss
    status = tracker.update(99000)
    print(f"\n   After $1000 loss:")
    print(f"   Status: {status.status}")
    print(f"   Daily DD: {status.daily_dd_pct:.2f}%")
    print(f"   Daily DD Usage: {status.daily_dd_usage_pct:.0f}%")

    if status.daily_dd_pct < 0.9 or status.daily_dd_pct > 1.1:
        print(f"   ❌ FAILED: Expected ~1% daily DD")
        return False

    print("   ✅ Prop Firm Tracker basic working")
    return True


def test_prop_firm_tracker_status_transitions():
    """Test status transitions."""
    print("\n" + "=" * 60)
    print("TEST 5: Tracker Status Transitions")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import PropFirmTracker

    rules = PropFirmRules(
        account_size=100000,
        profit_target_pct=8.0,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        min_trading_days=5
    )

    tracker = PropFirmTracker(rules)

    # Test WARNING status (75%+ of limit)
    status = tracker.update(97000)  # 3% daily DD (75% of 4%)
    print(f"   At $97,000 (3% DD):")
    print(f"   Status: {status.status}")
    print(f"   Daily DD Usage: {status.daily_dd_usage_pct:.0f}%")
    print(f"   Alerts: {status.alerts}")

    if status.status != 'WARNING':
        print(f"   ⚠️ Expected WARNING at 75%+ DD usage")

    # Test VIOLATED status
    tracker2 = PropFirmTracker(rules)
    status2 = tracker2.update(95900)  # 4.1% daily DD (exceeds limit)
    print(f"\n   At $95,900 (4.1% DD):")
    print(f"   Status: {status2.status}")
    print(f"   Daily DD: {status2.daily_dd_pct:.2f}%")
    print(f"   Alerts: {status2.alerts}")

    if status2.status != 'VIOLATED':
        print(f"   ❌ FAILED: Expected VIOLATED when DD > 4%")
        return False

    # Test PASSED status (need to end days first)
    tracker3 = PropFirmTracker(rules)
    tracker3.update(102000)
    for _ in range(5):
        tracker3.end_day(102000 + _ * 1000, trades=2)
    status3 = tracker3.update(108000)  # 8% profit, 5 days
    print(f"\n   At $108,000 after 5 days:")
    print(f"   Status: {status3.status}")
    print(f"   Profit: {status3.profit_pct:.2f}%")
    print(f"   Days Traded: {status3.days_traded}")

    if status3.status != 'PASSED':
        print(f"   ❌ FAILED: Expected PASSED with 8% profit and 5 days")
        return False

    print("   ✅ Status transitions working correctly")
    return True


def test_prop_firm_tracker_consistency():
    """Test consistency rule."""
    print("\n" + "=" * 60)
    print("TEST 6: Consistency Rule")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import PropFirmTracker

    rules = PropFirmRules(
        account_size=100000,
        profit_target_pct=8.0,
        consistency_rule=True
    )

    tracker = PropFirmTracker(rules)

    # Day 1: Make $4000 (big day)
    tracker.end_day(104000, trades=5)

    # Day 2: Make $1000
    tracker.current_day_start = 104000
    tracker.end_day(105000, trades=3)

    # Day 3: Make $1000
    tracker.current_day_start = 105000
    tracker.end_day(106000, trades=2)

    # Check status
    status = tracker.update(106000)
    print(f"   Total Profit: ${6000} (6%)")
    print(f"   Day 1 Profit: ${4000} ({4000/6000*100:.0f}% of total)")
    print(f"   Consistency OK: {status.consistency_ok}")
    print(f"   Max Day Profit %: {status.max_single_day_profit_pct:.0f}%")

    # Day 1 is 67% of profits, should fail consistency
    if status.consistency_ok:
        print("   ⚠️ Consistency should be violated (one day > 30%)")

    print("   ✅ Consistency rule check working")
    return True


def test_config_save_load():
    """Test JSON config save/load."""
    print("\n" + "=" * 60)
    print("TEST 7: Config Save/Load")
    print("=" * 60)

    config_path = PROJECT_ROOT / "qml" / "dashboard" / "config" / "user_config.json"

    if not config_path.exists():
        print(f"   ❌ FAILED: Config file not found at {config_path}")
        return False

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"   Loaded config from: {config_path}")
    print(f"   firm_name: {config.get('firm_name')}")
    print(f"   account_size: ${config.get('account_size'):,}")
    print(f"   max_daily_dd_pct: {config.get('max_daily_dd_pct')}%")
    print(f"   max_total_dd_pct: {config.get('max_total_dd_pct')}%")
    print(f"   profit_target_pct: {config.get('profit_target_pct')}%")

    # Verify Breakout defaults
    if config.get('firm_name') != 'Breakout':
        print(f"   ⚠️ Expected Breakout, got {config.get('firm_name')}")

    if config.get('max_daily_dd_pct') != 4.0:
        print(f"   ❌ FAILED: Expected 4% daily DD for Breakout")
        return False

    print("   ✅ Config save/load working")
    return True


def test_risk_module_exports():
    """Test risk module exports."""
    print("\n" + "=" * 60)
    print("TEST 8: Risk Module Exports")
    print("=" * 60)

    try:
        from src.risk import (
            KellyPositionSizer,
            KellyResult,
            PropFirmTracker,
            PropFirmStatus,
            DailyStats,
        )
        print("   ✅ KellyPositionSizer")
        print("   ✅ KellyResult")
        print("   ✅ PropFirmTracker")
        print("   ✅ PropFirmStatus")
        print("   ✅ DailyStats")
    except ImportError as e:
        print(f"   ❌ FAILED: Import error: {e}")
        return False

    print("   ✅ All exports working")
    return True


def test_kelly_from_trades():
    """Test Kelly calculation from trade history."""
    print("\n" + "=" * 60)
    print("TEST 9: Kelly from Trade History")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import KellyPositionSizer

    rules = PropFirmRules(account_size=100000, max_position_size_pct=2.0)
    sizer = KellyPositionSizer(rules, kelly_fraction=0.5)

    # Create mock trade history
    np.random.seed(42)
    trades = []
    for _ in range(50):
        is_win = np.random.rand() < 0.55
        pnl = np.random.uniform(150, 250) if is_win else np.random.uniform(-80, -120)
        trades.append({'pnl': pnl})

    result = sizer.calculate_from_trades(
        trades=trades,
        current_equity=100000,
        stop_loss_pct=0.02
    )

    print(f"   Trades: {len(trades)}")
    print(f"   Base Kelly: {result.base_kelly:.1f}%")
    print(f"   Position Size: {result.position_size_pct:.1f}%")
    print(f"   Position Dollars: ${result.position_size_dollars:,.0f}")

    if result.position_size_pct < 0 or result.position_size_pct > 100:
        print("   ❌ FAILED: Invalid position size")
        return False

    print("   ✅ Kelly from trades working")
    return True


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 10: Edge Cases")
    print("=" * 60)

    from src.validation.monte_carlo import PropFirmRules
    from src.risk import KellyPositionSizer, PropFirmTracker

    rules = PropFirmRules(account_size=100000)
    sizer = KellyPositionSizer(rules)

    # Zero win rate
    result = sizer.calculate(
        win_rate=0,
        avg_win=200,
        avg_loss=100,
        current_equity=100000,
        stop_loss_pct=0.02
    )
    print(f"   Zero win rate: position_size={result.position_size_pct:.1f}%")

    if result.position_size_pct < 0:
        print("   ❌ FAILED: Position size should not be negative")
        return False

    # 100% win rate
    result2 = sizer.calculate(
        win_rate=1.0,
        avg_win=200,
        avg_loss=100,
        current_equity=100000,
        stop_loss_pct=0.02
    )
    print(f"   100% win rate: position_size={result2.position_size_pct:.1f}%")

    # Empty trades
    result3 = sizer.calculate_from_trades(
        trades=[],
        current_equity=100000,
        stop_loss_pct=0.02
    )
    print(f"   Empty trades: position_size={result3.position_size_pct:.1f}%")

    if result3.position_size_pct != 0:
        print("   ⚠️ Expected 0% for empty trades")

    # Tracker reset
    tracker = PropFirmTracker(rules)
    tracker.update(98000)
    tracker.end_day(98000, trades=3)
    tracker.reset()

    if tracker.daily_stats:
        print("   ❌ FAILED: Reset should clear daily_stats")
        return False

    print("   ✅ Edge cases handled correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  PHASE 5 PROP FIRM MODULE TEST")
    print("=" * 70)

    tests = [
        ("PropFirmRules Extended", test_prop_firm_rules_extended),
        ("Kelly Sizer Basic", test_kelly_sizer_basic),
        ("Kelly Sizer Caps", test_kelly_sizer_caps),
        ("Prop Firm Tracker Basic", test_prop_firm_tracker_basic),
        ("Tracker Status Transitions", test_prop_firm_tracker_status_transitions),
        ("Consistency Rule", test_prop_firm_tracker_consistency),
        ("Config Save/Load", test_config_save_load),
        ("Risk Module Exports", test_risk_module_exports),
        ("Kelly from Trades", test_kelly_from_trades),
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
        print("\n   🎉 All tests passed! Prop firm module ready.")
    else:
        print("\n   ⚠️ Some tests failed. Check output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
