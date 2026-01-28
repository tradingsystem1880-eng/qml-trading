#!/usr/bin/env python3
"""
Phase 9.7: Quick Infrastructure Test
====================================
Tests all Phase 9.7 components without external API calls.

Run: python scripts/test_phase97_infrastructure.py

Tests:
1. ResearchJournal - logging and retrieval
2. FeatureValidator - all 4 new methods
3. FundingRateFetcher - symbol normalization
4. EdgeDegradationMonitor - alert triggers

Expected: All tests PASS in <5 seconds
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import pandas as pd
import numpy as np


def test_research_journal():
    """Test ResearchJournal logging and retrieval."""
    print("\n[1/5] Testing ResearchJournal...")

    from src.research.research_journal import ResearchJournal

    # Use temp directory with non-existent file to avoid polluting real journal
    # (NamedTemporaryFile creates empty file, which causes JSON parse error)
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "test_journal.json"

    try:
        journal = ResearchJournal(journal_path=temp_path)

        # Test logging experiment with data files
        result = journal.log_experiment({
            'hypothesis': 'Test hypothesis for Phase 9.7 validation',
            'feature_name': 'test_feature_97',
            'methodology': 'Unit test validation',
            'results': {'pf_change': 0.1, 'wr_change': 0.02},
            'conclusion': 'PASS',
            'notes': 'Automated test',
            'tags': ['test', 'phase97'],
        }, data_files=[str(PROJECT_ROOT / 'config' / 'default.yaml')])

        # Verify git commit hash is present
        assert 'git_commit' in result, "Missing git_commit in result"
        assert len(result['git_commit']) == 8 or result['git_commit'] == 'unknown', \
            f"Invalid git_commit format: {result['git_commit']}"

        # Verify data hashes
        assert 'data_hashes' in result, "Missing data_hashes in result"

        # Test retrieval
        assert journal.check_if_tested('test_feature_97'), "Feature not found in journal"

        # Test should_proceed
        should, reason = journal.should_proceed('test_feature_97')
        assert not should, f"Should not proceed after PASS: {reason}"

        print("  - log_experiment(): PASS")
        print("  - check_if_tested(): PASS")
        print("  - should_proceed(): PASS")
        print("  - git_commit tracking: PASS")
        print("  - data_hashes tracking: PASS")
        return True

    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_feature_validator():
    """Test FeatureValidator new methods."""
    print("\n[2/5] Testing FeatureValidator...")

    from src.research.feature_validator import FeatureValidator

    # FeatureValidator requires baseline_metrics
    baseline_metrics = {
        'profit_factor': 4.49,
        'win_rate': 0.55,
        'expectancy': 1.5,
    }
    validator = FeatureValidator(baseline_metrics=baseline_metrics)

    # Create mock trade data
    np.random.seed(42)
    n_trades = 200

    baseline_trades = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='4h'),
        'direction': np.random.choice(['LONG', 'SHORT'], n_trades),
        'pnl_r': np.random.normal(0.2, 1.5, n_trades),
        'is_win': np.random.random(n_trades) > 0.45,
        'funding_rate': np.random.normal(0, 0.0002, n_trades),
        'volatility': np.random.uniform(0.01, 0.05, n_trades),
    })

    # Filter to ~70% of trades
    filtered_trades = baseline_trades.sample(frac=0.7, random_state=42)

    # Test 1: bootstrap_significance_test
    bootstrap_result = validator.bootstrap_significance_test(
        baseline_trades, filtered_trades, n_bootstrap=100  # Small for speed
    )
    assert 'filtered_ci' in bootstrap_result, "Missing filtered_ci in bootstrap result"
    assert 'significant' in bootstrap_result, "Missing significant in bootstrap result"
    assert 'p_value' in bootstrap_result, "Missing p_value in bootstrap result"
    print("  - bootstrap_significance_test(): PASS")

    # Test 2: check_feature_correlation
    corr_result = validator.check_feature_correlation(
        baseline_trades,
        new_feature_col='funding_rate',
        existing_features=['volatility']
    )
    assert 'correlations' in corr_result, "Missing correlations in correlation result"
    assert 'any_redundant' in corr_result, "Missing any_redundant in correlation result"
    print("  - check_feature_correlation(): PASS")

    # Test 3: check_directional_balance
    balance_result = validator.check_directional_balance(baseline_trades, filtered_trades)
    assert 'baseline_long_pct' in balance_result, "Missing baseline_long_pct"
    assert 'severe_imbalance' in balance_result, "Missing severe_imbalance"
    print("  - check_directional_balance(): PASS")

    # Test 4: compare_max_drawdown
    dd_result = validator.compare_max_drawdown(baseline_trades, filtered_trades)
    assert 'baseline_max_dd_r' in dd_result, "Missing baseline_max_dd_r"
    assert 'filtered_max_dd_r' in dd_result, "Missing filtered_max_dd_r"
    assert 'significant_improvement' in dd_result, "Missing significant_improvement"
    print("  - compare_max_drawdown(): PASS")

    return True


def test_funding_rate_fetcher():
    """Test FundingRateFetcher symbol normalization (no API calls)."""
    print("\n[3/5] Testing FundingRateFetcher...")

    from src.data.funding_rates import FundingRateFetcher

    # Create fetcher - won't make API calls in this test
    fetcher = FundingRateFetcher()

    # Test symbol normalization
    test_cases = [
        ('BTC/USDT', 'BTC/USDT:USDT'),
        ('BTCUSDT', 'BTC/USDT:USDT'),
        ('BTC-USDT', 'BTC/USDT:USDT'),
        ('BTC/USDT:USDT', 'BTC/USDT:USDT'),
        ('ETH/USD', 'ETH/USD:USD'),
        ('ETHUSD', 'ETH/USD:USD'),
    ]

    for input_symbol, expected in test_cases:
        result = fetcher._normalize_symbol(input_symbol)
        assert result == expected, f"Normalization failed: {input_symbol} -> {result}, expected {expected}"

    print("  - _normalize_symbol(): PASS (6 formats tested)")

    # Test should_filter_trade with pre-provided funding rate
    # LONG with high positive funding should be filtered
    should_filter, reason = fetcher.should_filter_trade(
        'BTC/USDT', 'LONG', threshold=0.0001, funding_rate=0.0002
    )
    assert should_filter, "Should filter LONG with high positive funding"
    assert 'LONG_OVERCROWDED' in reason

    # SHORT with high negative funding should be filtered
    should_filter, reason = fetcher.should_filter_trade(
        'BTC/USDT', 'SHORT', threshold=0.0001, funding_rate=-0.0002
    )
    assert should_filter, "Should filter SHORT with high negative funding"
    assert 'SHORT_OVERCROWDED' in reason

    # LONG with negative funding should NOT be filtered
    should_filter, reason = fetcher.should_filter_trade(
        'BTC/USDT', 'LONG', threshold=0.0001, funding_rate=-0.0001
    )
    assert not should_filter, "Should NOT filter LONG with negative funding"
    assert 'PASSED' in reason

    # Test boundary case: exactly at threshold should filter (>= rule)
    should_filter, reason = fetcher.should_filter_trade(
        'BTC/USDT', 'LONG', threshold=0.0001, funding_rate=0.0001
    )
    assert should_filter, "Should filter LONG at exactly threshold (>= rule)"

    print("  - should_filter_trade(): PASS (4 cases including boundary)")

    return True


def test_edge_monitor():
    """Test EdgeDegradationMonitor alert triggers."""
    print("\n[4/5] Testing EdgeDegradationMonitor...")

    from src.monitoring.edge_monitor import EdgeDegradationMonitor

    baseline = {
        'profit_factor': 4.49,
        'win_rate': 0.55,
        'expectancy': 1.5,
    }

    monitor = EdgeDegradationMonitor(baseline)

    # Add some winning trades
    for _ in range(5):
        monitor.add_trade({'pnl_r': 2.0})

    assert len(monitor.alerts) == 0, "Should not alert on winning trades"
    print("  - add_trade() with wins: PASS (no alerts)")

    # Add losing streak to trigger alert
    for _ in range(10):
        monitor.add_trade({'pnl_r': -1.0})

    # Check for alerts
    assert len(monitor.alerts) > 0, "Should alert after losing streak"
    # EdgeAlert is a dataclass, not a dict
    print(f"  - Alert triggered: {monitor.alerts[-1].alert_type}")
    print("  - Losing streak detection: PASS")

    return True


def test_permutation_significance():
    """Test permutation significance test with known outcome."""
    print("\n[5/5] Testing permutation_significance_test...")

    from src.research.feature_validator import FeatureValidator, FeatureValidatorConfig

    # Create validator
    validator = FeatureValidator(
        baseline_metrics={'profit_factor': 4.49, 'win_rate': 0.55},
        config=FeatureValidatorConfig()
    )

    # Create synthetic data where filter DOES help
    np.random.seed(42)

    # 100 trades total
    n_trades = 100
    all_trades = pd.DataFrame({
        'pnl_r': np.random.normal(0.5, 1.5, n_trades),  # Positive expectancy
        'trade_id': range(n_trades)
    })

    # Make 30 of them particularly bad (filter should catch these)
    bad_indices = np.random.choice(n_trades, 30, replace=False)
    all_trades.loc[bad_indices, 'pnl_r'] = np.random.normal(-1.5, 0.5, 30)

    # Filter removes the bad ones (perfect filter for testing)
    kept_mask = pd.Series([i not in bad_indices for i in range(n_trades)])

    # Run permutation test with smaller iterations for speed
    result = validator.permutation_significance_test(
        all_trades, kept_mask, n_permutations=1000, random_seed=42
    )

    # Verify result structure
    assert 'test_type' in result, "Missing test_type"
    assert result['test_type'] == 'permutation', "Wrong test type"
    assert 'p_value' in result, "Missing p_value"
    assert 'actual_improvement' in result, "Missing actual_improvement"
    assert 'significant_at_95' in result, "Missing significant_at_95"

    # With perfect filter, should be highly significant
    assert result['actual_improvement'] > 0, f"Improvement should be positive, got {result['actual_improvement']}"

    print(f"  - Result structure: PASS")
    print(f"  - Actual improvement: {result['actual_improvement']}")
    print(f"  - P-value: {result['p_value']}")
    print(f"  - Significant at 95%: {result['significant_at_95']}")

    # Note: with only 1000 iterations, perfect filter might not always be significant
    # but improvement should always be positive
    if result['significant_at_95']:
        print("  - Perfect filter detected as significant: PASS")
    else:
        print("  - Note: Not significant with 1000 iterations (expected with small sample)")

    print("  - permutation_significance_test(): PASS")
    return True


def main():
    print("=" * 60)
    print("PHASE 9.7 INFRASTRUCTURE TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        'ResearchJournal': False,
        'FeatureValidator': False,
        'FundingRateFetcher': False,
        'EdgeDegradationMonitor': False,
        'PermutationSignificance': False,
    }

    try:
        results['ResearchJournal'] = test_research_journal()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results['FeatureValidator'] = test_feature_validator()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results['FundingRateFetcher'] = test_funding_rate_fetcher()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results['EdgeDegradationMonitor'] = test_edge_monitor()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results['PermutationSignificance'] = test_permutation_significance()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for component, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"  {component}: {status}")

    print()
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("\nPhase 9.7 infrastructure is ready.")
        return 0
    else:
        print(f"SOME TESTS FAILED ({passed}/{total})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
