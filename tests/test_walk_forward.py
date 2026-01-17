"""
Unit Tests for Purged Walk-Forward Engine
==========================================
Tests fold generation, purge/embargo gaps, and parameter stability.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import (
    PurgedWalkForwardEngine,
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
)


class TestWalkForwardConfig:
    """Tests for walk-forward configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        assert config.n_folds == 10
        assert config.purge_bars == 5
        assert config.embargo_bars == 5
        assert config.train_ratio == 0.7
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid n_folds
        config = WalkForwardConfig(n_folds=1)
        with pytest.raises(ValueError):
            config.validate()
        
        # Invalid train_ratio
        config = WalkForwardConfig(train_ratio=0.3)
        with pytest.raises(ValueError):
            config.validate()


class TestFoldGeneration:
    """Tests for fold generation."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        n_bars = 1000
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="4h")
        return pd.DataFrame({
            "time": dates,
            "close": np.random.randn(n_bars).cumsum() + 100,
        })
    
    def test_fold_generation(self, sample_df):
        """Test fold generation with purge/embargo."""
        config = WalkForwardConfig(n_folds=5, purge_bars=5, embargo_bars=5)
        engine = PurgedWalkForwardEngine(config=config)
        
        folds = engine.generate_folds(sample_df)
        
        assert len(folds) == 5
        
        for train_idx, test_idx in folds:
            # Train and test should not overlap
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set)
    
    def test_purge_gap(self, sample_df):
        """Test that purge gap exists between train and test."""
        config = WalkForwardConfig(n_folds=3, purge_bars=10, embargo_bars=0)
        engine = PurgedWalkForwardEngine(config=config)
        
        folds = engine.generate_folds(sample_df)
        
        for train_idx, test_idx in folds:
            # Get numeric positions
            train_positions = [sample_df.index.get_loc(i) for i in train_idx]
            test_positions = [sample_df.index.get_loc(i) for i in test_idx]
            
            train_end = max(train_positions)
            test_start = min(test_positions)
            
            # Gap should be at least purge_bars
            gap = test_start - train_end
            assert gap >= config.purge_bars
    
    def test_no_overlap_between_folds(self, sample_df):
        """Test that different folds don't use same test data."""
        config = WalkForwardConfig(n_folds=4, purge_bars=5, embargo_bars=5)
        engine = PurgedWalkForwardEngine(config=config)
        
        folds = engine.generate_folds(sample_df)
        
        # Collect all test indices
        all_test_indices = []
        for _, test_idx in folds:
            all_test_indices.extend(list(test_idx))
        
        # Check no duplicates
        assert len(all_test_indices) == len(set(all_test_indices))
    
    def test_insufficient_data_raises(self):
        """Test error when data is insufficient for requested folds."""
        small_df = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=50, freq="4h"),
            "close": np.random.randn(50),
        })
        
        config = WalkForwardConfig(n_folds=10, purge_bars=5, embargo_bars=5)
        engine = PurgedWalkForwardEngine(config=config)
        
        with pytest.raises(ValueError):
            engine.generate_folds(small_df)


class TestWalkForwardExecution:
    """Tests for walk-forward execution."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        n_bars = 500
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="4h")
        np.random.seed(42)
        return pd.DataFrame({
            "time": dates,
            "open": np.random.randn(n_bars).cumsum() + 100,
            "high": np.random.randn(n_bars).cumsum() + 102,
            "low": np.random.randn(n_bars).cumsum() + 98,
            "close": np.random.randn(n_bars).cumsum() + 100,
            "volume": np.random.rand(n_bars) * 1000,
        })
    
    def test_walk_forward_run(self, sample_df):
        """Test complete walk-forward run."""
        config = WalkForwardConfig(n_folds=3, purge_bars=3, embargo_bars=2)
        engine = PurgedWalkForwardEngine(config=config)
        
        # Simple objective function
        def objective_fn(df, params):
            returns = df["close"].pct_change().dropna()
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            return {
                "sharpe_ratio": sharpe * params.get("mult", 1.0),
                "total_trades": 10,
                "win_rate": 0.5,
            }
        
        param_grid = {"mult": [1.0, 1.5, 2.0]}
        
        result = engine.run(
            df=sample_df,
            objective_fn=objective_fn,
            param_grid=param_grid,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 3
        assert result.total_oos_trades > 0
    
    def test_parameter_stability(self, sample_df):
        """Test parameter stability calculation."""
        config = WalkForwardConfig(n_folds=3, purge_bars=2, embargo_bars=2)
        engine = PurgedWalkForwardEngine(config=config)
        
        # Simulate fold results with consistent parameters
        param_history = [
            {"mult": 1.5},
            {"mult": 1.5},
            {"mult": 1.5},
        ]
        
        stability = engine._calculate_param_stability(param_history)
        assert stability == 1.0  # All same = perfect stability
        
        # Variable parameters
        param_history = [
            {"mult": 1.0},
            {"mult": 2.0},
            {"mult": 3.0},
        ]
        
        stability = engine._calculate_param_stability(param_history)
        assert 0 < stability < 1  # Some instability


class TestWalkForwardReport:
    """Tests for walk-forward report generation."""
    
    def test_report_generation(self):
        """Test text report generation."""
        config = WalkForwardConfig(n_folds=3)
        engine = PurgedWalkForwardEngine(config=config)
        
        # Create mock result
        fold_results = [
            FoldResult(
                fold_idx=i,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 3, 1),
                test_start=datetime(2024, 3, 10),
                test_end=datetime(2024, 4, 1),
                in_sample_metrics={"sharpe_ratio": 1.2},
                out_of_sample_metrics={"sharpe_ratio": 0.9},
                out_of_sample_trades=20,
            )
            for i in range(3)
        ]
        
        result = WalkForwardResult(
            fold_results=fold_results,
            aggregate_sharpe=0.9,
            aggregate_return=15.0,
            aggregate_max_dd=10.0,
            aggregate_win_rate=0.55,
            total_oos_trades=60,
            param_stability_score=0.85,
            sharpe_stability=0.1,
            is_to_oos_ratio=0.75,
            config=config,
        )
        
        report = engine.generate_report(result)
        
        assert "WALK-FORWARD" in report
        assert "0.9" in report  # Sharpe
        assert "3" in report    # Folds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
