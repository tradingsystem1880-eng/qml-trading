"""
Integration Tests for Backtest Adapter
======================================
Tests the conversion of Phase 7.5 detection outputs to backtest formats.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.detection.historical_detector import HistoricalSwingDetector, HistoricalSwingPoint
from src.detection.pattern_validator import PatternValidator, ValidationResult, PatternDirection
from src.detection.pattern_scorer import PatternScorer, ScoringResult, PatternTier
from src.detection.backtest_adapter import BacktestAdapter, create_adapter
from src.detection.config import SwingDetectionConfig, PatternValidationConfig
from src.data.models import QMLPattern, PatternType
from src.core.models import Signal, SignalType


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_swings():
    """Create sample swing points for testing."""
    base_time = datetime(2024, 1, 1)

    swings = [
        HistoricalSwingPoint(
            bar_index=30,
            price=108.0,
            timestamp=pd.Timestamp(base_time + timedelta(hours=30*4)),
            swing_type='HIGH',
            atr_at_formation=2.5,
            significance_atr=3.0,
            significance_zscore=1.8,
            symbol='BTCUSDT',
            timeframe='4h',
        ),
        HistoricalSwingPoint(
            bar_index=50,
            price=100.0,
            timestamp=pd.Timestamp(base_time + timedelta(hours=50*4)),
            swing_type='LOW',
            atr_at_formation=2.5,
            significance_atr=3.2,
            significance_zscore=2.0,
            symbol='BTCUSDT',
            timeframe='4h',
        ),
        HistoricalSwingPoint(
            bar_index=70,
            price=115.0,
            timestamp=pd.Timestamp(base_time + timedelta(hours=70*4)),
            swing_type='HIGH',
            atr_at_formation=2.5,
            significance_atr=4.0,
            significance_zscore=2.5,
            symbol='BTCUSDT',
            timeframe='4h',
        ),
        HistoricalSwingPoint(
            bar_index=90,
            price=98.0,
            timestamp=pd.Timestamp(base_time + timedelta(hours=90*4)),
            swing_type='LOW',
            atr_at_formation=2.5,
            significance_atr=3.5,
            significance_zscore=2.2,
            symbol='BTCUSDT',
            timeframe='4h',
        ),
        HistoricalSwingPoint(
            bar_index=110,
            price=107.0,
            timestamp=pd.Timestamp(base_time + timedelta(hours=110*4)),
            swing_type='HIGH',
            atr_at_formation=2.5,
            significance_atr=2.8,
            significance_zscore=1.6,
            symbol='BTCUSDT',
            timeframe='4h',
        ),
    ]
    return swings


@pytest.fixture
def valid_pattern(sample_swings):
    """Create a valid pattern for testing."""
    from src.detection.pattern_validator import CandidatePattern

    validator = PatternValidator(PatternValidationConfig(
        p3_min_extension_atr=0.3,
        p3_max_extension_atr=20.0,
        p5_max_symmetry_atr=5.0,
        min_pattern_bars=5,
        max_pattern_bars=500,
    ))

    candidate = CandidatePattern(
        p1=sample_swings[0],
        p2=sample_swings[1],
        p3=sample_swings[2],
        p4=sample_swings[3],
        p5=sample_swings[4],
        direction=PatternDirection.BULLISH,
    )

    return validator.validate(candidate)


@pytest.fixture
def scored_pattern(valid_pattern):
    """Create scored pattern for testing."""
    scorer = PatternScorer()
    return scorer.score(valid_pattern)


# =============================================================================
# Adapter Tests
# =============================================================================

class TestBacktestAdapter:
    """Tests for BacktestAdapter."""

    def test_validation_to_qml_pattern(self, valid_pattern, scored_pattern):
        """Test conversion to QMLPattern."""
        adapter = BacktestAdapter()

        qml_pattern = adapter.validation_to_qml_pattern(
            validation_result=valid_pattern,
            scoring_result=scored_pattern,
            symbol='BTCUSDT',
            timeframe='4h',
        )

        assert qml_pattern is not None
        assert isinstance(qml_pattern, QMLPattern)
        assert qml_pattern.symbol == 'BTCUSDT'
        assert qml_pattern.timeframe == '4h'
        assert qml_pattern.pattern_type == PatternType.BULLISH
        assert qml_pattern.trading_levels is not None

    def test_validation_to_signal(self, valid_pattern, scored_pattern):
        """Test conversion to Signal."""
        adapter = BacktestAdapter()

        signal = adapter.validation_to_signal(
            validation_result=valid_pattern,
            scoring_result=scored_pattern,
            symbol='BTCUSDT',
            timeframe='4h',
        )

        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.timeframe == '4h'
        # BULLISH pattern = SHORT setup
        assert signal.signal_type == SignalType.SELL
        assert signal.stop_loss is not None
        assert signal.take_profit is not None

    def test_trading_levels_calculation(self, valid_pattern, scored_pattern):
        """Test that trading levels are calculated correctly."""
        adapter = BacktestAdapter(
            entry_buffer_atr=0.1,
            sl_buffer_atr=0.5,
            tp1_r_multiple=1.5,
        )

        qml_pattern = adapter.validation_to_qml_pattern(
            validation_result=valid_pattern,
            scoring_result=scored_pattern,
        )

        levels = qml_pattern.trading_levels
        assert levels is not None

        # For BULLISH (SHORT setup):
        # Entry should be above P5
        assert levels.entry > valid_pattern.p5.price

        # SL should be above head (P3)
        assert levels.stop_loss > valid_pattern.p3.price

        # Risk should be positive
        assert levels.risk_amount > 0

        # TP1 should be below entry (short)
        assert levels.take_profit_1 < levels.entry

    def test_signal_metadata_preserved(self, valid_pattern, scored_pattern):
        """Test that pattern metadata is preserved in Signal."""
        adapter = BacktestAdapter()

        signal = adapter.validation_to_signal(
            validation_result=valid_pattern,
            scoring_result=scored_pattern,
        )

        assert 'tier' in signal.metadata
        assert signal.metadata['tier'] == scored_pattern.tier.value
        assert 'head_extension_atr' in signal.metadata
        assert 'bos_efficiency' in signal.metadata
        assert 'p1_bar' in signal.metadata
        assert 'component_scores' in signal.metadata

    def test_invalid_pattern_returns_none(self):
        """Test that invalid patterns return None."""
        adapter = BacktestAdapter()

        invalid_result = ValidationResult(is_valid=False)
        scorer = PatternScorer()
        score_result = scorer.score(invalid_result)

        qml_pattern = adapter.validation_to_qml_pattern(
            invalid_result, score_result
        )
        assert qml_pattern is None

        signal = adapter.validation_to_signal(
            invalid_result, score_result
        )
        assert signal is None

    def test_batch_convert_to_patterns(self, valid_pattern, scored_pattern):
        """Test batch conversion to QMLPatterns."""
        adapter = BacktestAdapter()

        # Create list with one pattern
        vrs = [valid_pattern]
        srs = [scored_pattern]

        patterns = adapter.batch_convert_to_patterns(
            validation_results=vrs,
            scoring_results=srs,
            min_tier=PatternTier.C,
        )

        assert len(patterns) > 0
        assert all(isinstance(p, QMLPattern) for p in patterns)

    def test_batch_convert_with_tier_filter(self, valid_pattern, scored_pattern):
        """Test that tier filtering works in batch conversion."""
        adapter = BacktestAdapter()

        vrs = [valid_pattern]
        srs = [scored_pattern]

        # If pattern is not A-tier, filtering by A should return empty
        if scored_pattern.tier != PatternTier.A:
            patterns = adapter.batch_convert_to_patterns(
                validation_results=vrs,
                scoring_results=srs,
                min_tier=PatternTier.A,
            )
            assert len(patterns) == 0


class TestCreateAdapter:
    """Tests for adapter factory function."""

    def test_create_with_defaults(self):
        """Test creating adapter with default settings."""
        adapter = create_adapter()

        assert adapter is not None
        assert adapter.entry_buffer_atr == 0.1
        assert adapter.sl_buffer_atr == 0.5

    def test_create_with_config(self):
        """Test creating adapter from config."""
        config = PatternValidationConfig(
            entry_buffer_atr=0.2,
            sl_buffer_atr=0.8,
            tp1_r_multiple=2.0,
        )

        adapter = create_adapter(config)

        assert adapter.entry_buffer_atr == 0.2
        assert adapter.sl_buffer_atr == 0.8
        assert adapter.tp1_r_multiple == 2.0


# =============================================================================
# Integration Tests with Real Data
# =============================================================================

class TestRealDataIntegration:
    """Integration tests using real data pipeline."""

    @pytest.fixture
    def real_patterns(self):
        """Load real patterns from verification run."""
        from pathlib import Path
        import sys

        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        # Load data
        data_path = PROJECT_ROOT / "data/processed/BTCUSDT/4h_master.parquet"
        if not data_path.exists():
            pytest.skip("Data file not found")

        df = pd.read_parquet(data_path)
        df.columns = [c.lower() for c in df.columns]
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'time'}, inplace=True)

        # Detect swings
        config = SwingDetectionConfig(
            atr_period=14,
            lookback=5,
            lookforward=3,
            min_zscore=0.5,
            min_threshold_pct=0.0005,
            atr_multiplier=0.5,
        )
        detector = HistoricalSwingDetector(config, 'BTCUSDT', '4h')
        swings = detector.detect(df)

        # Find patterns
        validator = PatternValidator(PatternValidationConfig(
            p3_min_extension_atr=0.3,
            p3_max_extension_atr=10.0,
            p5_max_symmetry_atr=5.0,
            min_pattern_bars=8,
            max_pattern_bars=200,
        ))
        patterns = validator.find_patterns(swings, df['close'].values)

        # Score patterns
        scorer = PatternScorer()
        scored = []
        for p in patterns:
            if p.is_valid:
                scored.append((p, scorer.score(p)))

        return scored, df

    def test_real_patterns_convert_to_qml(self, real_patterns):
        """Test converting real patterns to QMLPattern."""
        scored, df = real_patterns

        if not scored:
            pytest.skip("No patterns found in data")

        adapter = BacktestAdapter()

        for vr, sr in scored[:3]:  # Test first 3
            qml = adapter.validation_to_qml_pattern(vr, sr)
            assert qml is not None
            assert qml.trading_levels is not None
            assert qml.trading_levels.risk_amount > 0

    def test_real_patterns_convert_to_signals(self, real_patterns):
        """Test converting real patterns to Signals."""
        scored, df = real_patterns

        if not scored:
            pytest.skip("No patterns found in data")

        adapter = BacktestAdapter()

        for vr, sr in scored[:3]:
            signal = adapter.validation_to_signal(vr, sr)
            assert signal is not None
            assert signal.risk_reward is not None
            assert signal.risk_reward > 0

    def test_backtest_engine_accepts_patterns(self, real_patterns):
        """Test that converted patterns can be passed to BacktestEngine."""
        scored, df = real_patterns

        if not scored:
            pytest.skip("No patterns found in data")

        adapter = BacktestAdapter()

        vrs = [vr for vr, sr in scored]
        srs = [sr for vr, sr in scored]

        qml_patterns = adapter.batch_convert_to_patterns(
            validation_results=vrs,
            scoring_results=srs,
            min_tier=PatternTier.C,
        )

        # Try to import and use backtest engine
        from src.backtest.engine import BacktestEngine, BacktestConfig

        engine = BacktestEngine(BacktestConfig())

        # The engine expects price_data dict
        price_data = {'BTCUSDT': df}

        # This should not raise
        result = engine.run(
            patterns=qml_patterns,
            price_data=price_data,
        )

        # Basic checks
        assert result is not None
        assert result.total_trades >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
