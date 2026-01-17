"""
Unit Tests for ML Pattern Registry
===================================
Tests for PatternRegistry, PatternFeatureExtractor, and database migrations.
"""

import json
import os
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.migrations import MigrationManager, run_migrations
from src.ml.pattern_registry import PatternRegistry
from src.ml.feature_extractor import PatternFeatureExtractor


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def registry(temp_db):
    """Create a PatternRegistry with temp database."""
    return PatternRegistry(temp_db)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 300
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4h')
    
    # Generate price series
    base_price = 40000
    returns = np.random.randn(n_bars) * 0.01
    close = base_price * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.random.rand(n_bars) * 0.01)
    low = close * (1 - np.random.rand(n_bars) * 0.01)
    open_price = close + np.random.randn(n_bars) * 50
    volume = np.random.rand(n_bars) * 1000000 + 500000
    
    return pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


@pytest.fixture
def sample_pattern_data():
    """Generate sample pattern data."""
    return {
        'symbol': 'BTC/USDT',
        'timeframe': '4h',
        'pattern_type': 'bullish',
        'detection_time': datetime.now(),
        'detection_idx': 280,
        'left_shoulder_price': 41000,
        'left_shoulder_idx': 250,
        'head_price': 39500,
        'head_idx': 265,
        'right_shoulder_price': 40800,
        'right_shoulder_idx': 275,
        'entry_price': 40900,
        'stop_loss': 39000,
        'take_profit': 43000,
        'atr': 400,
        'validity_score': 0.75,
    }


@pytest.fixture
def sample_features():
    """Generate sample features dict."""
    return {
        'ctx_atr_14': 400.5,
        'ctx_rsi_14': 55.3,
        'ctx_macd_line': 150.2,
        'geo_head_depth_atr': 3.75,
        'geo_shoulder_symmetry': 0.995,
        'geo_rr_ratio': 2.0,
        'temp_pattern_duration': 25.0,
        'vol_realized_20': 0.45,
        'mom_return_20': 2.5,
    }


# =============================================================================
# MIGRATION TESTS
# =============================================================================

class TestMigrations:
    """Test database migrations."""
    
    def test_migration_manager_initialization(self, temp_db):
        """Test MigrationManager creates database file."""
        manager = MigrationManager(temp_db)
        assert manager.db_path.exists()
    
    def test_run_migrations(self, temp_db):
        """Test migrations create required tables."""
        applied = run_migrations(temp_db)
        
        # Should apply all 3 migrations on fresh database
        assert len(applied) >= 2
        
        # Verify tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ml_pattern_registry'
        """)
        assert cursor.fetchone() is not None
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ml_model_versions'
        """)
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_migrations_idempotent(self, temp_db):
        """Test running migrations twice doesn't cause errors."""
        applied1 = run_migrations(temp_db)
        applied2 = run_migrations(temp_db)
        
        # Second run should apply no new migrations
        assert len(applied2) == 0
    
    def test_get_schema_info(self, temp_db):
        """Test schema info retrieval."""
        manager = MigrationManager(temp_db)
        manager.run_migrations()
        
        info = manager.get_schema_info()
        
        assert 'tables' in info
        assert 'ml_pattern_registry' in info['tables']
        assert 'ml_model_versions' in info['tables']


# =============================================================================
# PATTERN REGISTRY TESTS
# =============================================================================

class TestPatternRegistry:
    """Test PatternRegistry CRUD operations."""
    
    def test_registry_initialization(self, registry):
        """Test registry creates tables on init."""
        stats = registry.get_statistics()
        assert stats['total_patterns'] == 0
    
    def test_register_pattern(self, registry, sample_pattern_data, sample_features):
        """Test pattern registration."""
        pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
        
        assert pattern_id is not None
        assert len(pattern_id) > 0
        
        # Verify pattern stored
        pattern = registry.get_pattern(pattern_id)
        assert pattern is not None
        assert pattern['symbol'] == 'BTC/USDT'
        assert pattern['pattern_type'] == 'bullish_qml'
    
    def test_register_pattern_with_ml_confidence(self, registry, sample_pattern_data, sample_features):
        """Test pattern registration with ML confidence score."""
        pattern_id = registry.register_pattern(
            sample_pattern_data, 
            sample_features,
            ml_confidence=0.85
        )
        
        pattern = registry.get_pattern(pattern_id)
        assert pattern['ml_confidence'] == 0.85
    
    def test_label_pattern(self, registry, sample_pattern_data, sample_features):
        """Test pattern labeling."""
        pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
        
        # Label as win
        success = registry.label_pattern(pattern_id, 'win', outcome=2.5)
        assert success
        
        # Verify label
        pattern = registry.get_pattern(pattern_id)
        assert pattern['user_label'] == 'win'
        assert pattern['trade_outcome'] == 2.5
    
    def test_label_pattern_normalization(self, registry, sample_pattern_data, sample_features):
        """Test label input normalization."""
        pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
        
        # Test various label formats
        registry.label_pattern(pattern_id, 'W')  # Uppercase
        pattern = registry.get_pattern(pattern_id)
        assert pattern['user_label'] == 'win'
        
        registry.label_pattern(pattern_id, 'l')  # Lowercase
        pattern = registry.get_pattern(pattern_id)
        assert pattern['user_label'] == 'loss'
    
    def test_get_patterns_with_filters(self, registry, sample_pattern_data, sample_features):
        """Test pattern querying with filters."""
        # Register multiple patterns
        registry.register_pattern(sample_pattern_data, sample_features)
        
        sample_pattern_data['pattern_type'] = 'bearish'
        sample_pattern_data['detection_time'] = datetime(2024, 1, 2, 12, 0)
        registry.register_pattern(sample_pattern_data, sample_features)
        
        # Filter by type
        bullish = registry.get_patterns(pattern_type='bullish_qml')
        bearish = registry.get_patterns(pattern_type='bearish_qml')
        
        assert len(bullish) >= 1
        assert len(bearish) >= 1
    
    def test_get_unlabeled_patterns(self, registry, sample_pattern_data, sample_features):
        """Test unlabeled pattern retrieval."""
        pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
        
        unlabeled = registry.get_unlabeled_patterns()
        assert len(unlabeled) == 1
        
        # Label and check again
        registry.label_pattern(pattern_id, 'win')
        unlabeled = registry.get_unlabeled_patterns()
        assert len(unlabeled) == 0
    
    def test_get_training_data_insufficient_labels(self, registry, sample_pattern_data, sample_features):
        """Test training data extraction with insufficient labels."""
        # Register pattern but don't label
        registry.register_pattern(sample_pattern_data, sample_features)
        
        # Should raise error when not enough labeled patterns
        with pytest.raises(ValueError):
            registry.get_training_data(min_labels=10)
    
    def test_get_training_data_success(self, registry, sample_pattern_data, sample_features):
        """Test training data extraction with sufficient labels."""
        # Register and label multiple patterns
        for i in range(20):
            sample_pattern_data['detection_time'] = datetime(2024, 1, 1 + i)
            pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
            label = 'win' if i % 2 == 0 else 'loss'
            registry.label_pattern(pattern_id, label, outcome=1.0 if label == 'win' else -0.5)
        
        X, y = registry.get_training_data(min_labels=10)
        
        assert len(X) == 20
        assert len(y) == 20
        assert sum(y) == 10  # Half wins, half losses
    
    def test_find_similar_patterns(self, registry, sample_pattern_data, sample_features):
        """Test similar pattern finding."""
        # Register a pattern
        registry.register_pattern(sample_pattern_data, sample_features)
        
        # Find similar (should find itself)
        similar = registry.find_similar_patterns(sample_features, n=5)
        
        assert len(similar) >= 1
        assert similar[0]['similarity'] > 0.9  # Should be very similar to itself
    
    def test_get_statistics(self, registry, sample_pattern_data, sample_features):
        """Test statistics calculation."""
        # Register and label patterns
        for i in range(5):
            sample_pattern_data['detection_time'] = datetime(2024, 1, 1 + i)
            pattern_id = registry.register_pattern(sample_pattern_data, sample_features)
            if i < 3:
                registry.label_pattern(pattern_id, 'win' if i < 2 else 'loss')
        
        stats = registry.get_statistics()
        
        assert stats['total_patterns'] == 5
        assert stats['labeled'] == 3
        assert stats['unlabeled'] == 2
        assert stats['by_label']['win'] == 2
        assert stats['by_label']['loss'] == 1


# =============================================================================
# FEATURE EXTRACTOR TESTS
# =============================================================================

class TestPatternFeatureExtractor:
    """Test PatternFeatureExtractor."""
    
    def test_extractor_initialization(self):
        """Test extractor creates successfully."""
        extractor = PatternFeatureExtractor()
        assert extractor is not None
    
    def test_extract_pattern_features(self, sample_ohlcv, sample_pattern_data):
        """Test feature extraction produces features."""
        extractor = PatternFeatureExtractor()
        
        features = extractor.extract_pattern_features(
            sample_pattern_data,
            sample_ohlcv,
            bar_idx=280
        )
        
        # Should have multiple features
        assert len(features) > 20
        
        # Check key feature categories exist
        has_ctx = any(k.startswith('ctx_') for k in features)
        has_geo = any(k.startswith('geo_') for k in features)
        has_temp = any(k.startswith('temp_') for k in features)
        
        assert has_ctx, "Should have context features"
        assert has_geo, "Should have geometry features"
        assert has_temp, "Should have temporal features"
    
    def test_features_are_numeric(self, sample_ohlcv, sample_pattern_data):
        """Test all features are numeric values."""
        extractor = PatternFeatureExtractor()
        
        features = extractor.extract_pattern_features(
            sample_pattern_data,
            sample_ohlcv,
            bar_idx=280
        )
        
        for name, value in features.items():
            assert isinstance(value, (int, float)), f"Feature {name} is not numeric: {type(value)}"
            assert not np.isnan(value), f"Feature {name} is NaN"
            assert not np.isinf(value), f"Feature {name} is infinite"
    
    def test_features_to_json(self, sample_ohlcv, sample_pattern_data):
        """Test features can be serialized to JSON."""
        extractor = PatternFeatureExtractor()
        
        features = extractor.extract_pattern_features(
            sample_pattern_data,
            sample_ohlcv,
            bar_idx=280
        )
        
        json_str = extractor.features_to_json(features)
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) == len(features)
    
    def test_geometry_features(self, sample_ohlcv, sample_pattern_data):
        """Test geometry-specific features."""
        extractor = PatternFeatureExtractor()
        
        features = extractor.extract_pattern_features(
            sample_pattern_data,
            sample_ohlcv,
            bar_idx=280
        )
        
        # Check geometry features
        assert 'geo_head_depth_atr' in features
        assert 'geo_shoulder_symmetry' in features
        assert 'geo_is_bullish' in features
        
        # Bullish pattern should have is_bullish = 1.0
        assert features['geo_is_bullish'] == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, temp_db, sample_ohlcv, sample_pattern_data):
        """Test complete registration -> labeling -> training workflow."""
        # Initialize components
        registry = PatternRegistry(temp_db)
        extractor = PatternFeatureExtractor()
        
        # Extract features
        features = extractor.extract_pattern_features(
            sample_pattern_data,
            sample_ohlcv,
            bar_idx=280
        )
        
        # Register pattern
        pattern_id = registry.register_pattern(sample_pattern_data, features)
        assert pattern_id is not None
        
        # Label pattern
        success = registry.label_pattern(pattern_id, 'win', outcome=2.5)
        assert success
        
        # Verify
        pattern = registry.get_pattern(pattern_id)
        assert pattern['user_label'] == 'win'
        assert 'features' in pattern


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
