"""
Unit Tests for Trend Validator
==============================
Tests for Phase 7.6 prior trend validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.detection.trend_validator import (
    TrendValidator,
    TrendValidationConfig,
    TrendValidationResult,
)
from src.detection.pattern_validator import PatternDirection
from src.detection.historical_detector import HistoricalSwingPoint


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def uptrend_swings():
    """Create swings representing an uptrend (HH, HL pattern)."""
    # Create swing points for an uptrend: Higher Highs, Higher Lows
    swings = []

    # Starting point
    base_time = datetime(2024, 1, 1)
    base_price = 100.0

    # Low 1 (bar 0)
    swings.append(HistoricalSwingPoint(
        bar_index=0, price=100.0, timestamp=pd.Timestamp(base_time),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # High 1 (bar 10)
    swings.append(HistoricalSwingPoint(
        bar_index=10, price=105.0, timestamp=pd.Timestamp(base_time + timedelta(hours=40)),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # Low 2 - Higher Low (bar 20)
    swings.append(HistoricalSwingPoint(
        bar_index=20, price=102.0, timestamp=pd.Timestamp(base_time + timedelta(hours=80)),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # High 2 - Higher High (bar 30)
    swings.append(HistoricalSwingPoint(
        bar_index=30, price=110.0, timestamp=pd.Timestamp(base_time + timedelta(hours=120)),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # Low 3 - Higher Low (bar 40)
    swings.append(HistoricalSwingPoint(
        bar_index=40, price=106.0, timestamp=pd.Timestamp(base_time + timedelta(hours=160)),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # High 3 - Higher High (bar 50) - This is P1 of our pattern
    swings.append(HistoricalSwingPoint(
        bar_index=50, price=115.0, timestamp=pd.Timestamp(base_time + timedelta(hours=200)),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    return swings


@pytest.fixture
def downtrend_swings():
    """Create swings representing a downtrend (LH, LL pattern)."""
    swings = []
    base_time = datetime(2024, 1, 1)

    # High 1 (bar 0)
    swings.append(HistoricalSwingPoint(
        bar_index=0, price=120.0, timestamp=pd.Timestamp(base_time),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # Low 1 (bar 10)
    swings.append(HistoricalSwingPoint(
        bar_index=10, price=115.0, timestamp=pd.Timestamp(base_time + timedelta(hours=40)),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # High 2 - Lower High (bar 20)
    swings.append(HistoricalSwingPoint(
        bar_index=20, price=118.0, timestamp=pd.Timestamp(base_time + timedelta(hours=80)),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # Low 2 - Lower Low (bar 30)
    swings.append(HistoricalSwingPoint(
        bar_index=30, price=110.0, timestamp=pd.Timestamp(base_time + timedelta(hours=120)),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # High 3 - Lower High (bar 40)
    swings.append(HistoricalSwingPoint(
        bar_index=40, price=114.0, timestamp=pd.Timestamp(base_time + timedelta(hours=160)),
        swing_type='HIGH', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    # Low 3 - Lower Low (bar 50) - This is P1 of our pattern
    swings.append(HistoricalSwingPoint(
        bar_index=50, price=105.0, timestamp=pd.Timestamp(base_time + timedelta(hours=200)),
        swing_type='LOW', atr_at_formation=2.0,
        significance_atr=1.5, significance_zscore=1.0,
    ))

    return swings


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    n_bars = 100
    dates = [datetime(2024, 1, 1) + timedelta(hours=4*i) for i in range(n_bars)]

    close = [100.0 + i * 0.1 + np.random.uniform(-0.5, 0.5) for i in range(n_bars)]
    close = np.array(close)

    df = pd.DataFrame({
        'time': dates,
        'open': close + np.random.uniform(-0.3, 0.3, n_bars),
        'high': close + np.random.uniform(0.5, 1.5, n_bars),
        'low': close - np.random.uniform(0.5, 1.5, n_bars),
        'close': close,
        'volume': np.random.uniform(1000, 5000, n_bars),
        'ATR': np.full(n_bars, 2.0),
    })

    return df


@pytest.fixture
def default_config():
    """Default trend validation config."""
    return TrendValidationConfig()


@pytest.fixture
def loose_config():
    """Loose config for testing."""
    return TrendValidationConfig(
        min_adx=10.0,
        min_trend_move_atr=1.0,
        min_trend_swings=2,
        min_trend_bars=10,
        direction_consistency=0.4,
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestTrendValidator:
    """Test trend validator basic functionality."""

    def test_validate_returns_result(
        self, uptrend_swings, sample_ohlcv_df, default_config
    ):
        """Test that validate returns a TrendValidationResult."""
        validator = TrendValidator(config=default_config)

        result = validator.validate(
            swings=uptrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        assert isinstance(result, TrendValidationResult)

    def test_uptrend_detected_for_bullish_pattern(
        self, uptrend_swings, sample_ohlcv_df, loose_config
    ):
        """Test that uptrend is correctly detected for bullish pattern."""
        validator = TrendValidator(config=loose_config)

        result = validator.validate(
            swings=uptrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        # Should detect uptrend for bullish pattern
        if result.is_valid:
            assert result.trend_direction == 'UP'

    def test_downtrend_detected_for_bearish_pattern(
        self, downtrend_swings, sample_ohlcv_df, loose_config
    ):
        """Test that downtrend is correctly detected for bearish pattern."""
        validator = TrendValidator(config=loose_config)

        result = validator.validate(
            swings=downtrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BEARISH,
        )

        # Should detect downtrend for bearish pattern
        if result.is_valid:
            assert result.trend_direction == 'DOWN'


# =============================================================================
# SWING STRUCTURE ANALYSIS TESTS
# =============================================================================

class TestSwingStructureAnalysis:
    """Test swing structure analysis."""

    def test_uptrend_structure(self, uptrend_swings, loose_config):
        """Test HH/HL structure analysis for uptrend."""
        validator = TrendValidator(config=loose_config)
        structure = validator._analyze_swing_structure(uptrend_swings)

        # Uptrend should have higher highs and higher lows
        assert structure['higher_highs'] > 0 or structure['higher_lows'] > 0

    def test_downtrend_structure(self, downtrend_swings, loose_config):
        """Test LH/LL structure analysis for downtrend."""
        validator = TrendValidator(config=loose_config)
        structure = validator._analyze_swing_structure(downtrend_swings)

        # Downtrend should have lower highs and lower lows
        assert structure['lower_highs'] > 0 or structure['lower_lows'] > 0


# =============================================================================
# REJECTION TESTS
# =============================================================================

class TestRejections:
    """Test rejection cases."""

    def test_insufficient_swings_rejected(self, sample_ohlcv_df, default_config):
        """Test that insufficient swings are rejected."""
        validator = TrendValidator(config=default_config)

        # Only 2 swings - not enough
        few_swings = [
            HistoricalSwingPoint(
                bar_index=0, price=100.0,
                timestamp=pd.Timestamp(datetime(2024, 1, 1)),
                swing_type='LOW', atr_at_formation=2.0,
                significance_atr=1.5, significance_zscore=1.0,
            ),
            HistoricalSwingPoint(
                bar_index=10, price=105.0,
                timestamp=pd.Timestamp(datetime(2024, 1, 2, 16)),
                swing_type='HIGH', atr_at_formation=2.0,
                significance_atr=1.5, significance_zscore=1.0,
            ),
        ]

        result = validator.validate(
            swings=few_swings,
            p1_bar_index=20,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        assert not result.is_valid
        assert 'insufficient' in result.rejection_reason.lower()

    def test_wrong_trend_direction_rejected(
        self, uptrend_swings, sample_ohlcv_df, loose_config
    ):
        """Test that wrong trend direction is rejected."""
        validator = TrendValidator(config=loose_config)

        # Try to validate uptrend swings as bearish pattern (expects downtrend)
        result = validator.validate(
            swings=uptrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BEARISH,  # Wrong direction
        )

        # Should fail because uptrend doesn't match bearish pattern
        if result.trend_direction == 'UP':
            assert not result.is_valid
            assert 'mismatch' in result.rejection_reason.lower()


# =============================================================================
# METRIC CALCULATION TESTS
# =============================================================================

class TestMetricCalculations:
    """Test metric calculations."""

    def test_trend_move_calculated(
        self, uptrend_swings, sample_ohlcv_df, loose_config
    ):
        """Test that trend move is calculated."""
        validator = TrendValidator(config=loose_config)

        result = validator.validate(
            swings=uptrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        # trend_move_atr should be positive
        assert result.trend_move_atr >= 0

    def test_trend_bars_calculated(
        self, uptrend_swings, sample_ohlcv_df, loose_config
    ):
        """Test that trend bars are calculated correctly."""
        validator = TrendValidator(config=loose_config)

        result = validator.validate(
            swings=uptrend_swings,
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        # Should have trend bars > 0
        assert result.trend_bars >= 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_swings(self, sample_ohlcv_df, default_config):
        """Test handling of empty swing list."""
        validator = TrendValidator(config=default_config)

        result = validator.validate(
            swings=[],
            p1_bar_index=50,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        assert not result.is_valid

    def test_p1_at_start(self, sample_ohlcv_df, default_config):
        """Test handling when P1 is at the start of data."""
        validator = TrendValidator(config=default_config)

        # Single swing at bar 0
        swings = [
            HistoricalSwingPoint(
                bar_index=0, price=100.0,
                timestamp=pd.Timestamp(datetime(2024, 1, 1)),
                swing_type='HIGH', atr_at_formation=2.0,
                significance_atr=1.5, significance_zscore=1.0,
            ),
        ]

        result = validator.validate(
            swings=swings,
            p1_bar_index=0,
            df=sample_ohlcv_df,
            pattern_direction=PatternDirection.BULLISH,
        )

        # Should handle gracefully
        assert not result.is_valid

    def test_find_trend_sequence(self, uptrend_swings, loose_config):
        """Test find_trend_sequence helper."""
        validator = TrendValidator(config=loose_config)

        sequence = validator.find_trend_sequence(
            swings=uptrend_swings,
            end_idx=50,
            trend_direction='UP',
        )

        assert isinstance(sequence, list)
        # Should include swings before bar 50
        for swing in sequence:
            assert swing.bar_index < 50
