"""
Phase 6: Experiment Lab Tests
=============================
Verification tests for A/B testing framework.

Tests:
1. ParameterSet hash consistency
2. GridSearchConfig total combinations
3. ParameterGridManager deduplication
4. BH correction correctness
5. Rank experiments sorting
6. P-value calculation
7. Get significant discoveries pipeline
8. ExperimentRunner initialization
9. Module exports
10. Integration test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import date
import tempfile
import os


def test_parameter_set_hash_consistency():
    """Test 1: Same parameters produce same hash."""
    from src.experiments import ParameterSet

    params1 = ParameterSet(
        swing_lookback=5,
        smoothing_window=5,
        min_head_extension_atr=0.5,
    )

    params2 = ParameterSet(
        swing_lookback=5,
        smoothing_window=5,
        min_head_extension_atr=0.5,
    )

    # Same params should have same hash
    assert params1.to_hash() == params2.to_hash(), "Same parameters should produce same hash"

    # Different params should have different hash
    params3 = ParameterSet(swing_lookback=7)
    assert params1.to_hash() != params3.to_hash(), "Different parameters should produce different hash"

    print("✓ Test 1: ParameterSet hash consistency")


def test_grid_search_config_combinations():
    """Test 2: GridSearchConfig calculates correct total combinations."""
    from src.experiments import GridSearchConfig

    # Minimal config should have 1 combination
    minimal = GridSearchConfig.minimal()
    assert minimal.total_combinations() == 1, "Minimal config should have 1 combination"

    # Small config should have reasonable number
    small = GridSearchConfig.small()
    small_total = small.total_combinations()
    assert 10 < small_total < 500, f"Small config should have 10-500 combinations, got {small_total}"

    # Full config should have ~70K combinations
    # 4 * 3 * 3 * 2 * 3 * 3 * 3 * 3 * 3 * 2 * 2 = 69,984
    full = GridSearchConfig()
    full_total = full.total_combinations()
    expected = 4 * 3 * 3 * 2 * 3 * 3 * 3 * 3 * 3 * 2 * 2  # 69,984
    assert full_total == expected, f"Full config should have {expected} combinations, got {full_total}"

    print(f"✓ Test 2: GridSearchConfig total combinations (full={full_total:,})")


def test_parameter_grid_manager_deduplication():
    """Test 3: ParameterGridManager tracks tested parameters."""
    from src.experiments import ParameterSet, ParameterGridManager
    from src.data.sqlite_manager import SQLiteManager

    # Use temp database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name

    try:
        db = SQLiteManager(temp_db)
        manager = ParameterGridManager(db)

        params = ParameterSet(swing_lookback=5)

        # Should not be tested initially
        assert not manager.has_been_tested(params), "Parameters should not be tested initially"

        # Mark as tested
        param_hash = manager.mark_tested(params)
        assert len(param_hash) == 12, "Hash should be 12 characters"

        # Should now be tested
        assert manager.has_been_tested(params), "Parameters should be tested after marking"

        print("✓ Test 3: ParameterGridManager deduplication")

    finally:
        os.unlink(temp_db)


def test_benjamini_hochberg_correction():
    """Test 4: BH correction properly controls FDR."""
    from src.experiments import benjamini_hochberg_correction

    # Test with known p-values
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.50, 0.80, 1.0]
    alpha = 0.05

    results = benjamini_hochberg_correction(p_values, alpha)

    # Check that results are returned for all inputs
    assert len(results) == len(p_values), "Should return result for each p-value"

    # Very low p-values should be significant
    # Index 0 has p=0.001, should definitely be significant at alpha=0.05
    significant_results = [r for r in results if r[2]]  # r[2] is is_significant
    assert len(significant_results) > 0, "At least one result should be significant"

    # p=0.001 should be significant
    idx_0_result = next(r for r in results if r[0] == 0)  # r[0] is original_idx
    assert idx_0_result[2], "p=0.001 should remain significant after BH"

    # Very high p-values should not be significant
    idx_9_result = next(r for r in results if r[0] == 9)  # p=1.0
    assert not idx_9_result[2], "p=1.0 should not be significant"

    print("✓ Test 4: Benjamini-Hochberg correction")


def test_rank_experiments_sorting():
    """Test 5: Rank experiments sorts correctly by metric."""
    from src.experiments import rank_experiments

    experiments = [
        {'param_hash': 'a', 'sharpe_ratio': 1.5, 'total_trades': 50},
        {'param_hash': 'b', 'sharpe_ratio': 2.0, 'total_trades': 40},
        {'param_hash': 'c', 'sharpe_ratio': 1.0, 'total_trades': 60},
        {'param_hash': 'd', 'sharpe_ratio': 0.5, 'total_trades': 10},  # Should be filtered out
    ]

    ranked = rank_experiments(
        experiments,
        primary_metric='sharpe_ratio',
        min_trades=30,
    )

    # Should filter out experiment with 10 trades
    assert len(ranked) == 3, "Should filter out low trade count experiments"

    # Should be sorted by Sharpe descending
    assert ranked[0]['param_hash'] == 'b', "Highest Sharpe should be first"
    assert ranked[1]['param_hash'] == 'a', "Second highest Sharpe should be second"
    assert ranked[2]['param_hash'] == 'c', "Third highest Sharpe should be third"

    # Should have ranks assigned
    assert ranked[0]['rank'] == 1
    assert ranked[1]['rank'] == 2
    assert ranked[2]['rank'] == 3

    print("✓ Test 5: Rank experiments sorting")


def test_p_value_calculation():
    """Test 6: P-value calculation produces valid results."""
    from src.experiments import calculate_experiment_p_value

    # High Sharpe with many trades should have low p-value
    p_high_sharpe = calculate_experiment_p_value(
        sharpe_ratio=2.0,
        total_trades=100,
        baseline_sharpe=0.0,
    )
    # Very high z-scores can produce p-values that round to 0
    assert 0 <= p_high_sharpe < 0.05, f"High Sharpe should have low p-value, got {p_high_sharpe}"

    # Zero Sharpe should have p-value around 0.5
    p_zero_sharpe = calculate_experiment_p_value(
        sharpe_ratio=0.0,
        total_trades=100,
        baseline_sharpe=0.0,
    )
    assert 0.45 < p_zero_sharpe < 0.55, f"Zero Sharpe should have p-value ~0.5, got {p_zero_sharpe}"

    # Negative Sharpe should have high p-value
    p_negative = calculate_experiment_p_value(
        sharpe_ratio=-1.0,
        total_trades=100,
        baseline_sharpe=0.0,
    )
    assert p_negative > 0.5, f"Negative Sharpe should have high p-value, got {p_negative}"

    print("✓ Test 6: P-value calculation")


def test_get_significant_discoveries_pipeline():
    """Test 7: Full pipeline from raw experiments to significant discoveries."""
    from src.experiments import get_significant_discoveries

    experiments = [
        {'param_hash': 'a', 'sharpe_ratio': 3.0, 'total_trades': 100},  # Very high, should be significant
        {'param_hash': 'b', 'sharpe_ratio': 2.5, 'total_trades': 100},  # High, likely significant
        {'param_hash': 'c', 'sharpe_ratio': 0.5, 'total_trades': 100},  # Low, not significant
        {'param_hash': 'd', 'sharpe_ratio': 0.0, 'total_trades': 100},  # Zero, not significant
        {'param_hash': 'e', 'sharpe_ratio': -0.5, 'total_trades': 100}, # Negative, not significant
    ]

    results = get_significant_discoveries(
        experiments,
        alpha=0.05,
        min_trades=30,
    )

    # Check structure
    assert 'significant' in results
    assert 'not_significant' in results
    assert 'summary' in results

    # High Sharpe experiments should be significant
    assert len(results['significant']) > 0, "Should have some significant discoveries"

    # Summary should have correct totals
    summary = results['summary']
    assert summary['total_experiments'] == 5
    assert summary['tested_for_significance'] == 5

    print(f"✓ Test 7: Get significant discoveries pipeline (found {len(results['significant'])} significant)")


def test_experiment_runner_initialization():
    """Test 8: ExperimentRunner initializes correctly."""
    from src.experiments import ExperimentRunner
    from src.data.sqlite_manager import SQLiteManager

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name

    try:
        db = SQLiteManager(temp_db)

        # Should initialize without backtest function
        runner = ExperimentRunner(db)
        assert runner.db is not None
        assert runner.backtest_func is None
        assert runner.grid_manager is not None

        # Should initialize with backtest function
        def dummy_backtest(**kwargs):
            return {'total_trades': 10, 'win_rate': 0.5}

        runner2 = ExperimentRunner(db, backtest_func=dummy_backtest)
        assert runner2.backtest_func is not None

        print("✓ Test 8: ExperimentRunner initialization")

    finally:
        os.unlink(temp_db)


def test_module_exports():
    """Test 9: All expected classes/functions are exported."""
    from src.experiments import (
        ParameterSet,
        GridSearchConfig,
        ParameterGridManager,
        ExperimentResult,
        ExperimentRunner,
        SignificanceResult,
        benjamini_hochberg_correction,
        analyze_experiment_significance,
        rank_experiments,
        calculate_experiment_p_value,
        add_p_values_to_experiments,
        get_significant_discoveries,
    )

    # Verify they're all classes/functions (not None)
    assert ParameterSet is not None
    assert GridSearchConfig is not None
    assert ParameterGridManager is not None
    assert ExperimentResult is not None
    assert ExperimentRunner is not None
    assert SignificanceResult is not None
    assert callable(benjamini_hochberg_correction)
    assert callable(analyze_experiment_significance)
    assert callable(rank_experiments)
    assert callable(calculate_experiment_p_value)
    assert callable(add_p_values_to_experiments)
    assert callable(get_significant_discoveries)

    print("✓ Test 9: Module exports")


def test_integration_grid_generation():
    """Test 10: Integration test for grid generation and filtering."""
    from src.experiments import (
        ParameterSet,
        GridSearchConfig,
        ParameterGridManager,
    )
    from src.data.sqlite_manager import SQLiteManager

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name

    try:
        db = SQLiteManager(temp_db)
        manager = ParameterGridManager(db)

        # Use small config
        config = GridSearchConfig.small()
        total = config.total_combinations()

        # Generate and mark first 5 as tested
        count = 0
        first_five = []
        for params in manager.generate_grid(config):
            manager.mark_tested(params)
            first_five.append(params.to_hash())
            count += 1
            if count >= 5:
                break

        # Get untested - should skip the 5 we marked
        untested_count = 0
        untested_hashes = []
        for params in manager.get_untested(config, limit=10):
            untested_hashes.append(params.to_hash())
            untested_count += 1

        # Verify no overlap
        overlap = set(first_five) & set(untested_hashes)
        assert len(overlap) == 0, f"Untested should not include tested params, overlap: {overlap}"

        # Check progress
        progress = manager.get_fast_progress(config)
        assert progress['tested'] >= 5, f"Should have at least 5 tested, got {progress['tested']}"

        print(f"✓ Test 10: Integration - grid generation and filtering (tested {progress['tested']} of {total})")

    finally:
        os.unlink(temp_db)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 6: Experiment Lab Tests")
    print("=" * 60 + "\n")

    tests = [
        test_parameter_set_hash_consistency,
        test_grid_search_config_combinations,
        test_parameter_grid_manager_deduplication,
        test_benjamini_hochberg_correction,
        test_rank_experiments_sorting,
        test_p_value_calculation,
        test_get_significant_discoveries_pipeline,
        test_experiment_runner_initialization,
        test_module_exports,
        test_integration_grid_generation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All Phase 6 tests passed!")
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)
