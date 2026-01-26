"""
Phase 7.5 Detection Logic Tests
================================
Comprehensive tests for the Phase 7.5 detection system improvements.

Tests cover:
1. Config dataclass validation
2. Historical detector idempotency
3. Pattern validator geometry checks
4. Gaussian scorer behavior
5. Integration pipeline
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.detection.config import (
    SwingDetectionConfig,
    PatternValidationConfig,
    PatternScoringConfig,
    DetectionConfig,
)
from src.detection.historical_detector import (
    HistoricalSwingDetector,
    HistoricalSwingPoint,
)
from src.detection.pattern_validator import (
    PatternValidator,
    ValidationResult,
    CandidatePattern,
    PatternDirection,
    RejectionReason,
)
from src.detection.pattern_scorer import (
    PatternScorer,
    ScoringResult,
    PatternTier,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame with clear swing points."""
    n_bars = 200
    np.random.seed(42)  # For reproducibility

    # Base price with trending behavior
    base_price = 100.0
    trend = np.linspace(0, 10, n_bars)
    noise = np.random.randn(n_bars) * 2

    # Add some clear swing points
    swing_pattern = np.zeros(n_bars)
    swing_pattern[30:40] = np.linspace(0, 8, 10)  # Swing high 1
    swing_pattern[40:50] = np.linspace(8, 0, 10)
    swing_pattern[60:70] = np.linspace(0, 12, 10)  # Swing high 2 (head)
    swing_pattern[70:80] = np.linspace(12, -2, 10)  # Deep retracement
    swing_pattern[90:100] = np.linspace(-2, 6, 10)  # Swing high 3

    close = base_price + trend + noise + swing_pattern

    # Generate OHLCV
    df = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=n_bars, freq='4h'),
        'open': close + np.random.randn(n_bars) * 0.5,
        'high': close + np.abs(np.random.randn(n_bars)) * 1.5,
        'low': close - np.abs(np.random.randn(n_bars)) * 1.5,
        'close': close,
        'volume': np.random.uniform(1000, 10000, n_bars),
    })

    # Ensure high >= close, open and low <= close, open
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


@pytest.fixture
def sample_swings():
    """Create sample swing points for testing validator and scorer."""
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
            price=115.0,  # Head extends beyond P1
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
            price=98.0,  # Breaks P2 level (BOS)
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
            price=107.0,  # Right shoulder similar to P1
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


# =============================================================================
# Config Tests
# =============================================================================

class TestSwingDetectionConfig:
    """Tests for SwingDetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SwingDetectionConfig()
        assert config.atr_period == 14
        assert config.lookback == 5
        assert config.lookforward == 3
        assert config.min_zscore == 1.5
        assert config.min_threshold_pct == 0.001

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SwingDetectionConfig(
            atr_period=20,
            lookback=7,
            lookforward=5,
            min_zscore=2.0,
        )
        assert config.atr_period == 20
        assert config.lookback == 7
        assert config.lookforward == 5
        assert config.min_zscore == 2.0

    def test_invalid_atr_period(self):
        """Test validation of atr_period."""
        with pytest.raises(ValueError):
            SwingDetectionConfig(atr_period=0)

    def test_invalid_threshold_pct(self):
        """Test validation of min_threshold_pct."""
        with pytest.raises(ValueError):
            SwingDetectionConfig(min_threshold_pct=2.0)  # Must be < 1


class TestPatternValidationConfig:
    """Tests for PatternValidationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PatternValidationConfig()
        assert config.p3_min_extension_atr == 0.5
        assert config.p5_max_symmetry_atr == 2.0
        assert config.min_pattern_bars == 10

    def test_invalid_extension_range(self):
        """Test that min must be less than max."""
        with pytest.raises(ValueError):
            PatternValidationConfig(
                p3_min_extension_atr=5.0,
                p3_max_extension_atr=3.0,
            )


class TestPatternScoringConfig:
    """Tests for PatternScoringConfig dataclass."""

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = PatternScoringConfig()
        total = (
            config.head_extension_weight +
            config.bos_efficiency_weight +
            config.shoulder_symmetry_weight +
            config.swing_significance_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_invalid_weights(self):
        """Test that non-unity weights raise error."""
        with pytest.raises(ValueError):
            PatternScoringConfig(
                head_extension_weight=0.5,
                bos_efficiency_weight=0.5,
                shoulder_symmetry_weight=0.5,  # Total > 1
                swing_significance_weight=0.2,
            )


class TestDetectionConfig:
    """Tests for DetectionConfig master class."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "swing": {"atr_period": 20},
            "validation": {"min_pattern_bars": 15},
            "emit_unconfirmed": True,
        }
        config = DetectionConfig.from_dict(config_dict)

        assert config.swing.atr_period == 20
        assert config.validation.min_pattern_bars == 15
        assert config.emit_unconfirmed is True

    def test_to_dict(self):
        """Test exporting config to dictionary."""
        config = DetectionConfig()
        config_dict = config.to_dict()

        assert "swing" in config_dict
        assert "validation" in config_dict
        assert "scoring" in config_dict
        assert config_dict["swing"]["atr_period"] == 14


# =============================================================================
# Historical Detector Tests
# =============================================================================

class TestHistoricalSwingDetector:
    """Tests for HistoricalSwingDetector."""

    def test_detect_returns_list(self, sample_ohlcv_df):
        """Test that detect returns a list of swing points."""
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)

        assert isinstance(swings, list)
        assert all(isinstance(s, HistoricalSwingPoint) for s in swings)

    def test_idempotency(self, sample_ohlcv_df):
        """Test that the same input produces identical output."""
        detector = HistoricalSwingDetector()

        swings1 = detector.detect(sample_ohlcv_df)
        swings2 = detector.detect(sample_ohlcv_df)

        assert len(swings1) == len(swings2)

        for s1, s2 in zip(swings1, swings2):
            assert s1.bar_index == s2.bar_index
            assert s1.price == s2.price
            assert s1.swing_type == s2.swing_type
            assert s1.atr_at_formation == s2.atr_at_formation

    def test_sorted_by_index(self, sample_ohlcv_df):
        """Test that swings are sorted by bar index."""
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)

        indices = [s.bar_index for s in swings]
        assert indices == sorted(indices)

    def test_atr_at_formation_populated(self, sample_ohlcv_df):
        """Test that atr_at_formation is set for all swings."""
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)

        for swing in swings:
            assert swing.atr_at_formation > 0
            assert not np.isnan(swing.atr_at_formation)

    def test_zscore_calculated(self, sample_ohlcv_df):
        """Test that z-scores are calculated."""
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)

        # At least some swings should have non-zero z-scores
        zscores = [s.significance_zscore for s in swings]
        assert any(z != 0 for z in zscores)

    def test_get_swing_stats(self, sample_ohlcv_df):
        """Test swing statistics calculation."""
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)
        stats = detector.get_swing_stats(swings)

        assert 'total' in stats
        assert 'highs' in stats
        assert 'lows' in stats
        assert stats['total'] == stats['highs'] + stats['lows']

    def test_empty_df_returns_empty(self):
        """Test that empty DataFrame returns empty list."""
        detector = HistoricalSwingDetector()
        empty_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        swings = detector.detect(empty_df)

        assert swings == []

    def test_short_df_returns_empty(self):
        """Test that short DataFrame returns empty list."""
        detector = HistoricalSwingDetector()
        short_df = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=10, freq='4h'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10,
        })
        swings = detector.detect(short_df)

        assert swings == []


# =============================================================================
# Pattern Validator Tests
# =============================================================================

class TestPatternValidator:
    """Tests for PatternValidator."""

    def test_valid_bullish_pattern(self, sample_swings):
        """Test validation of a valid bullish pattern."""
        validator = PatternValidator()

        candidate = CandidatePattern(
            p1=sample_swings[0],  # HIGH
            p2=sample_swings[1],  # LOW
            p3=sample_swings[2],  # HIGH (head)
            p4=sample_swings[3],  # LOW (BOS)
            p5=sample_swings[4],  # HIGH
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert result.is_valid
        assert result.direction == PatternDirection.BULLISH

    def test_head_extension_too_small(self, sample_swings):
        """Test rejection when head doesn't extend enough."""
        validator = PatternValidator(
            PatternValidationConfig(p3_min_extension_atr=5.0, p3_max_extension_atr=10.0)
        )

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert not result.is_valid
        assert result.rejection_reason == RejectionReason.HEAD_EXTENSION_TOO_SMALL

    def test_bos_not_found(self, sample_swings):
        """Test rejection when P4 doesn't break P2."""
        validator = PatternValidator()

        # Modify P4 so it doesn't break P2
        modified_p4 = HistoricalSwingPoint(
            bar_index=90,
            price=102.0,  # Above P2's 100.0, so no BOS
            timestamp=sample_swings[3].timestamp,
            swing_type='LOW',
            atr_at_formation=2.5,
            significance_atr=3.5,
            significance_zscore=2.2,
        )

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=modified_p4,
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert not result.is_valid
        assert result.rejection_reason == RejectionReason.BOS_NOT_FOUND

    def test_shoulder_asymmetry(self, sample_swings):
        """Test rejection when shoulders are too different."""
        validator = PatternValidator(
            PatternValidationConfig(p5_max_symmetry_atr=0.1)  # Very strict
        )

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert not result.is_valid
        assert result.rejection_reason == RejectionReason.SHOULDER_ASYMMETRY

    def test_pattern_too_short(self, sample_swings):
        """Test rejection when pattern duration is too short."""
        validator = PatternValidator(
            PatternValidationConfig(min_pattern_bars=200, max_pattern_bars=500)
        )

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert not result.is_valid
        assert result.rejection_reason == RejectionReason.PATTERN_TOO_SHORT

    def test_bos_efficiency_calculated(self, sample_swings):
        """Test that BOS efficiency is calculated."""
        validator = PatternValidator()

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        result = validator.validate(candidate)
        assert result.is_valid
        assert 0 <= result.bos_efficiency <= 1.0


# =============================================================================
# Pattern Scorer Tests
# =============================================================================

class TestPatternScorer:
    """Tests for PatternScorer."""

    def test_score_valid_pattern(self, sample_swings):
        """Test scoring a valid pattern."""
        validator = PatternValidator()
        scorer = PatternScorer()

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        validation_result = validator.validate(candidate)
        assert validation_result.is_valid

        score_result = scorer.score(validation_result)
        assert 0 <= score_result.total_score <= 1.0
        assert score_result.tier in PatternTier

    def test_score_invalid_pattern_returns_zero(self):
        """Test that invalid patterns get zero score."""
        scorer = PatternScorer()

        invalid_result = ValidationResult(is_valid=False)
        score_result = scorer.score(invalid_result)

        assert score_result.total_score == 0.0
        assert score_result.tier == PatternTier.REJECT

    def test_gaussian_scoring_optimal(self):
        """Test that optimal values give score of 1.0."""
        scorer = PatternScorer()

        # Use optimal values from config
        result = scorer.score_from_metrics(
            head_extension_atr=scorer.config.head_extension_optimal,
            bos_efficiency=scorer.config.bos_efficiency_optimal,
            shoulder_diff_atr=scorer.config.shoulder_symmetry_optimal,
            avg_swing_zscore=scorer.config.swing_significance_optimal,
        )

        # All component scores should be 1.0
        assert abs(result.head_extension_score - 1.0) < 0.001
        assert abs(result.bos_efficiency_score - 1.0) < 0.001
        assert abs(result.shoulder_symmetry_score - 1.0) < 0.001
        assert abs(result.swing_significance_score - 1.0) < 0.001

    def test_gaussian_scoring_decays(self):
        """Test that scores decay as values move from optimal."""
        scorer = PatternScorer()

        # Optimal values
        optimal_result = scorer.score_from_metrics(
            head_extension_atr=1.5,  # optimal
            bos_efficiency=0.9,  # optimal
            shoulder_diff_atr=0.0,  # optimal
            avg_swing_zscore=2.5,  # optimal
        )

        # Suboptimal values
        suboptimal_result = scorer.score_from_metrics(
            head_extension_atr=3.0,  # far from optimal
            bos_efficiency=0.5,  # below optimal
            shoulder_diff_atr=2.0,  # away from optimal
            avg_swing_zscore=0.5,  # below optimal
        )

        assert suboptimal_result.total_score < optimal_result.total_score

    def test_tier_classification(self):
        """Test tier classification based on score."""
        scorer = PatternScorer()

        # A-tier: high score
        a_result = scorer.score_from_metrics(1.5, 0.9, 0.0, 2.5)
        assert a_result.tier == PatternTier.A

        # Lower score should be B or C
        lower_result = scorer.score_from_metrics(3.0, 0.5, 1.5, 0.5)
        assert lower_result.tier in [PatternTier.B, PatternTier.C, PatternTier.REJECT]

    def test_scoring_breakdown_format(self, sample_swings):
        """Test that scoring breakdown is properly formatted."""
        validator = PatternValidator()
        scorer = PatternScorer()

        candidate = CandidatePattern(
            p1=sample_swings[0],
            p2=sample_swings[1],
            p3=sample_swings[2],
            p4=sample_swings[3],
            p5=sample_swings[4],
            direction=PatternDirection.BULLISH,
        )

        validation_result = validator.validate(candidate)
        score_result = scorer.score(validation_result)
        breakdown = scorer.get_scoring_breakdown(score_result)

        assert "Pattern Quality Score" in breakdown
        assert "Tier" in breakdown
        assert "Head Extension" in breakdown
        assert "BOS Efficiency" in breakdown


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationPipeline:
    """Integration tests for the full detection pipeline."""

    def test_full_pipeline(self, sample_ohlcv_df):
        """Test the complete detection pipeline."""
        # 1. Detect swings
        detector = HistoricalSwingDetector()
        swings = detector.detect(sample_ohlcv_df)

        assert len(swings) > 0

        # 2. Find patterns
        validator = PatternValidator()
        patterns = validator.find_patterns(swings, sample_ohlcv_df['close'].values)

        # 3. Score patterns
        scorer = PatternScorer()
        scored_patterns = []
        for pattern in patterns:
            if pattern.is_valid:
                score = scorer.score(pattern)
                scored_patterns.append((pattern, score))

        # At least the pipeline runs without errors
        # Whether we find patterns depends on the data
        assert isinstance(scored_patterns, list)

    def test_factory_returns_historical_detector(self):
        """Test that factory can create historical detector."""
        from src.detection.factory import get_detector

        detector = get_detector("historical")
        assert isinstance(detector, HistoricalSwingDetector)

        detector = get_detector("batch")
        assert isinstance(detector, HistoricalSwingDetector)

        detector = get_detector("backtest")
        assert isinstance(detector, HistoricalSwingDetector)

    def test_config_round_trip(self):
        """Test config serialization and deserialization."""
        original = DetectionConfig(
            swing=SwingDetectionConfig(atr_period=20),
            validation=PatternValidationConfig(min_pattern_bars=15),
            scoring=PatternScoringConfig(),
        )

        # Convert to dict and back
        config_dict = original.to_dict()
        restored = DetectionConfig.from_dict(config_dict)

        assert restored.swing.atr_period == original.swing.atr_period
        assert restored.validation.min_pattern_bars == original.validation.min_pattern_bars


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
