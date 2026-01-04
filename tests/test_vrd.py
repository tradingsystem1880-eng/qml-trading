"""
Unit Tests for VRD 2.0 (Versioned Research Database)
=====================================================
Tests database operations, experiment tracking, and artifact management.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.validation.database import VRDDatabase, ExperimentRecord, generate_param_hash
from src.validation.tracker import ExperimentTracker


class TestVRDDatabase:
    """Tests for VRD SQLite database."""
    
    @pytest.fixture
    def db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vrd.db")
            db = VRDDatabase(db_path)
            yield db
            db.close()
    
    def test_database_creation(self, db):
        """Test database and tables are created."""
        assert db.db_path.exists()
        
        # Check tables exist
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "experiments" in tables
    
    def test_insert_experiment(self, db):
        """Test inserting an experiment record."""
        record = ExperimentRecord(
            experiment_id="test123",
            timestamp=datetime.now().isoformat(),
            git_hash="abc123",
            strategy_name="QML_TEST_V1",
            param_hash="def456",
            params={"atr_mult": 1.5, "threshold": 0.02},
            data_start="2024-01-01",
            data_end="2024-12-31",
            random_seed=42,
            fold_count=10,
            sharpe_ratio=1.5,
            max_drawdown_pct=15.0,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=100,
        )
        
        exp_id = db.insert_experiment(record)
        assert exp_id == "test123"
        
        # Verify retrieval
        retrieved = db.get_experiment("test123")
        assert retrieved is not None
        assert retrieved.strategy_name == "QML_TEST_V1"
        assert retrieved.sharpe_ratio == 1.5
        assert retrieved.params["atr_mult"] == 1.5
    
    def test_query_experiments(self, db):
        """Test querying experiments with filters."""
        # Insert multiple records
        for i in range(5):
            record = ExperimentRecord(
                experiment_id=f"exp{i}",
                timestamp=datetime.now().isoformat(),
                git_hash="abc",
                strategy_name="QML_TEST" if i < 3 else "QML_OTHER",
                param_hash=f"hash{i}",
                params={},
                data_start="2024-01-01",
                data_end="2024-12-31",
                random_seed=42,
                fold_count=10,
                sharpe_ratio=1.0 + i * 0.2,
                status="completed",
            )
            db.insert_experiment(record)
        
        # Query by strategy name
        results = db.query_experiments(strategy_name="QML_TEST")
        assert len(results) == 3
        
        # Query by min sharpe
        results = db.query_experiments(min_sharpe=1.5)
        assert all(r.sharpe_ratio >= 1.5 for r in results)
    
    def test_get_best_experiments(self, db):
        """Test getting top experiments by metric."""
        # Insert experiments with varying Sharpe
        for i in range(10):
            record = ExperimentRecord(
                experiment_id=f"exp{i}",
                timestamp=datetime.now().isoformat(),
                git_hash="abc",
                strategy_name="TEST",
                param_hash=f"h{i}",
                params={},
                data_start="2024-01-01",
                data_end="2024-12-31",
                random_seed=42,
                fold_count=10,
                sharpe_ratio=float(i),
                status="completed",
            )
            db.insert_experiment(record)
        
        # Get top 3
        best = db.get_best_experiments(metric="sharpe_ratio", n=3)
        assert len(best) == 3
        assert best[0].sharpe_ratio == 9.0  # Highest
        assert best[1].sharpe_ratio == 8.0
        assert best[2].sharpe_ratio == 7.0


class TestParamHash:
    """Tests for parameter hashing."""
    
    def test_deterministic_hash(self):
        """Same params should produce same hash."""
        params1 = {"atr_mult": 1.5, "threshold": 0.02}
        params2 = {"atr_mult": 1.5, "threshold": 0.02}
        
        assert generate_param_hash(params1) == generate_param_hash(params2)
    
    def test_order_independent(self):
        """Hash should be order-independent."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}
        
        assert generate_param_hash(params1) == generate_param_hash(params2)
    
    def test_different_params_different_hash(self):
        """Different params should produce different hashes."""
        params1 = {"atr_mult": 1.5}
        params2 = {"atr_mult": 2.0}
        
        assert generate_param_hash(params1) != generate_param_hash(params2)


class TestExperimentTracker:
    """Tests for experiment tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create temporary tracker for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            yield tracker
    
    def test_start_experiment(self, tracker):
        """Test starting an experiment."""
        exp_id = tracker.start_experiment(
            strategy_name="QML_TEST_V1",
            params={"atr_mult": 1.5},
            data_range=("2024-01-01", "2024-12-31"),
            seed=42,
            fold_count=10,
        )
        
        assert exp_id is not None
        assert len(exp_id) == 8  # UUID truncated
        assert tracker.current_experiment_id == exp_id
        assert tracker.current_experiment_dir is not None
        assert tracker.current_experiment_dir.exists()
    
    def test_experiment_directory_pattern(self, tracker):
        """Test experiment directory follows [TIMESTAMP]_[STRATEGY]_[HASH] pattern."""
        tracker.start_experiment(
            strategy_name="TEST_STRAT",
            params={"x": 1},
            data_range=("2024-01-01", "2024-12-31"),
        )
        
        dir_name = tracker.current_experiment_dir.name
        parts = dir_name.split("_")
        
        # Should have timestamp (YYYYMMDD_HHMMSS), strategy, hash
        assert len(parts) >= 4
        assert "TEST" in dir_name
        assert "STRAT" in dir_name
    
    def test_log_fold_result(self, tracker):
        """Test logging fold results."""
        tracker.start_experiment(
            strategy_name="TEST",
            params={},
            data_range=("2024-01-01", "2024-12-31"),
        )
        
        tracker.log_fold_result(
            fold_idx=0,
            train_start="2024-01-01",
            train_end="2024-06-01",
            test_start="2024-06-10",
            test_end="2024-08-01",
            optimal_params={"x": 1.0},
            in_sample_metrics={"sharpe_ratio": 1.2},
            out_of_sample_metrics={"sharpe_ratio": 0.9},
        )
        
        # Check fold file was created
        fold_file = tracker.current_experiment_dir / "fold_00.json"
        assert fold_file.exists()
        
        with open(fold_file) as f:
            data = json.load(f)
        assert data["fold_idx"] == 0
        assert data["in_sample"]["sharpe_ratio"] == 1.2
    
    def test_save_artifact(self, tracker):
        """Test saving artifacts."""
        import pandas as pd
        
        tracker.start_experiment(
            strategy_name="TEST",
            params={},
            data_range=("2024-01-01", "2024-12-31"),
        )
        
        # Save DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tracker.save_artifact("test_df", df)
        assert path.suffix == ".csv"
        assert path.exists()
        
        # Save dict as JSON
        data = {"key": "value"}
        path = tracker.save_artifact("test_dict", data)
        assert path.suffix == ".json"
        assert path.exists()
    
    def test_finalize_experiment(self, tracker):
        """Test finalizing an experiment."""
        tracker.start_experiment(
            strategy_name="TEST",
            params={},
            data_range=("2024-01-01", "2024-12-31"),
        )
        
        result = tracker.finalize(
            metrics={"sharpe_ratio": 1.5, "max_drawdown_pct": 10.0},
        )
        
        assert result.status == "completed"
        assert result.sharpe_ratio == 1.5
        assert tracker.current_experiment_id is None  # Reset after finalize


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
