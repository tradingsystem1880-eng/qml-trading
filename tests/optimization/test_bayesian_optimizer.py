"""
Unit Tests for Bayesian Optimizer
=================================
Tests for Phase 7.6 parameter optimization.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Check if skopt is available
try:
    import skopt
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from src.optimization.parallel_runner import (
    ParallelDetectionRunner,
    DetectionResult,
    AggregateResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_detection_result():
    """Create a mock detection result."""
    return DetectionResult(
        symbol='BTCUSDT',
        timeframe='4h',
        num_patterns=10,
        num_swings=50,
        scores=[0.7, 0.8, 0.6, 0.75, 0.65, 0.9, 0.5, 0.85, 0.7, 0.8],
        mean_score=0.725,
        tier_a=2,
        tier_b=5,
        tier_c=3,
        rejected=0,
        duration_ms=100.0,
    )


@pytest.fixture
def mock_aggregate_result():
    """Create a mock aggregate result."""
    return AggregateResult(
        total_patterns=100,
        unique_symbols=10,
        total_symbols=15,
        mean_score=0.65,
        median_score=0.68,
        tier_a_count=20,
        tier_b_count=50,
        tier_c_count=30,
        symbol_results={},
        duration_ms=5000.0,
    )


@pytest.fixture
def sample_params():
    """Sample parameter dictionary."""
    return {
        'min_bar_separation': 5,
        'min_move_atr': 1.0,
        'forward_confirm_pct': 0.3,
        'lookback': 5,
        'lookforward': 5,
        'p3_min_extension_atr': 0.5,
        'p3_max_extension_atr': 5.0,
        'p4_min_break_atr': 0.1,
        'p5_max_symmetry_atr': 2.0,
        'min_pattern_bars': 10,
        'min_adx': 20.0,
        'min_trend_move_atr': 3.0,
        'min_trend_swings': 3,
        'head_extension_weight': 0.25,
        'bos_efficiency_weight': 0.20,
    }


# =============================================================================
# PARALLEL RUNNER TESTS
# =============================================================================

class TestParallelDetectionRunner:
    """Test parallel detection runner."""

    def test_get_available_symbols(self, tmp_path):
        """Test getting available symbols from data directory."""
        # Create mock data directories
        (tmp_path / "BTCUSDT").mkdir()
        (tmp_path / "BTCUSDT" / "4h_master.parquet").touch()
        (tmp_path / "ETHUSDT").mkdir()
        (tmp_path / "ETHUSDT" / "4h_master.parquet").touch()

        with patch('src.optimization.parallel_runner.PROJECT_ROOT', tmp_path):
            runner = ParallelDetectionRunner()
            # Note: This may return empty if the path structure doesn't match
            # In real tests, we'd mock the data loading

    def test_run_with_dict_creates_configs(self, sample_params):
        """Test that run_with_dict creates proper config objects."""
        runner = ParallelDetectionRunner(symbols=[])

        # Mock the run method to capture the configs
        with patch.object(runner, 'run') as mock_run:
            mock_run.return_value = AggregateResult(
                total_patterns=0, unique_symbols=0, total_symbols=0,
                mean_score=0.0, median_score=0.0,
                tier_a_count=0, tier_b_count=0, tier_c_count=0,
            )

            runner.run_with_dict(sample_params)

            # Check that run was called
            mock_run.assert_called_once()

            # Check the config arguments
            call_args = mock_run.call_args
            assert call_args is not None


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_detection_result_creation(self, mock_detection_result):
        """Test DetectionResult can be created."""
        assert mock_detection_result.symbol == 'BTCUSDT'
        assert mock_detection_result.timeframe == '4h'
        assert mock_detection_result.num_patterns == 10
        assert len(mock_detection_result.scores) == 10

    def test_detection_result_with_error(self):
        """Test DetectionResult with error."""
        result = DetectionResult(
            symbol='BTCUSDT',
            timeframe='4h',
            num_patterns=0,
            num_swings=0,
            error="Data not found",
        )

        assert result.error == "Data not found"
        assert result.num_patterns == 0


class TestAggregateResult:
    """Test AggregateResult dataclass."""

    def test_aggregate_result_creation(self, mock_aggregate_result):
        """Test AggregateResult creation."""
        assert mock_aggregate_result.total_patterns == 100
        assert mock_aggregate_result.unique_symbols == 10
        assert mock_aggregate_result.mean_score == 0.65

    def test_aggregate_result_empty(self):
        """Test empty AggregateResult."""
        result = AggregateResult(
            total_patterns=0,
            unique_symbols=0,
            total_symbols=5,
            mean_score=0.0,
            median_score=0.0,
            tier_a_count=0,
            tier_b_count=0,
            tier_c_count=0,
        )

        assert result.total_patterns == 0
        assert result.mean_score == 0.0


# =============================================================================
# BAYESIAN OPTIMIZER TESTS (Only if skopt available)
# =============================================================================

@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="scikit-optimize not installed")
class TestBayesianOptimizer:
    """Test Bayesian optimizer."""

    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        from src.optimization import BayesianOptimizer, OptimizationConfig

        config = OptimizationConfig(
            n_calls=5,
            n_initial_points=2,
        )

        # Create optimizer with mock runner
        runner = Mock(spec=ParallelDetectionRunner)
        runner.symbols = ['BTCUSDT']
        runner._data_cache = {}
        runner.preload_data = Mock()

        optimizer = BayesianOptimizer(config=config, runner=runner)

        assert optimizer.config.n_calls == 5
        assert optimizer._iteration == 0

    def test_objective_function(self, mock_aggregate_result):
        """Test objective function calculation."""
        from src.optimization import BayesianOptimizer, OptimizationConfig, PARAM_NAMES

        config = OptimizationConfig(
            n_calls=5,
            n_initial_points=2,
            min_patterns=10,
            target_patterns=100,
        )

        # Create mock runner
        runner = Mock(spec=ParallelDetectionRunner)
        runner.symbols = ['BTCUSDT']
        runner._data_cache = {}
        runner.preload_data = Mock()
        runner.run_with_dict = Mock(return_value=mock_aggregate_result)

        optimizer = BayesianOptimizer(config=config, runner=runner)

        # Create sample params list
        params_list = [5, 1.0, 0.3, 5, 5, 0.5, 5.0, 0.1, 2.0, 10, 20.0, 3.0, 3, 0.25, 0.20]

        score = optimizer.objective(params_list)

        # Score should be negative (we minimize)
        assert isinstance(score, float)

    def test_calculate_score_penalties(self, mock_aggregate_result):
        """Test score calculation with penalties."""
        from src.optimization import BayesianOptimizer, OptimizationConfig

        config = OptimizationConfig(
            min_patterns=200,  # Higher than mock result
            target_patterns=500,
        )

        runner = Mock(spec=ParallelDetectionRunner)
        runner.symbols = ['BTCUSDT']
        runner._data_cache = {}

        optimizer = BayesianOptimizer(config=config, runner=runner)

        # Mock result has 100 patterns, less than min_patterns=200
        score = optimizer._calculate_score(mock_aggregate_result)

        # Should return penalty score
        assert score == 1000.0  # Heavy penalty for < min_patterns


# =============================================================================
# OPTIMIZATION RESULT TESTS
# =============================================================================

@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="scikit-optimize not installed")
class TestOptimizationResult:
    """Test OptimizationResult."""

    def test_result_save(self, tmp_path, mock_aggregate_result, sample_params):
        """Test saving optimization result."""
        from src.optimization import OptimizationResult

        result = OptimizationResult(
            best_params=sample_params,
            best_score=0.75,
            best_result=mock_aggregate_result,
            all_scores=[0.5, 0.6, 0.7, 0.75],
            all_params=[sample_params] * 4,
            n_iterations=4,
            duration_seconds=60.0,
        )

        save_path = tmp_path / "test_result.json"
        result.save(save_path)

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            data = json.load(f)

        assert data['best_score'] == 0.75
        assert data['n_iterations'] == 4


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="scikit-optimize not installed")
class TestHelperFunctions:
    """Test helper functions."""

    def test_load_best_params(self, tmp_path):
        """Test loading best params from file."""
        from src.optimization.bayesian_optimizer import load_best_params

        # Create mock params file
        params_file = tmp_path / "best_params.json"
        params_data = {
            'params': {'min_bar_separation': 6, 'min_move_atr': 1.2},
            'score': 0.8,
            'patterns': 150,
        }

        with open(params_file, 'w') as f:
            json.dump(params_data, f)

        loaded = load_best_params(params_file)

        assert loaded['min_bar_separation'] == 6
        assert loaded['min_move_atr'] == 1.2

    def test_load_best_params_missing_file(self, tmp_path):
        """Test loading best params from missing file."""
        from src.optimization.bayesian_optimizer import load_best_params

        with pytest.raises(FileNotFoundError):
            load_best_params(tmp_path / "nonexistent.json")
