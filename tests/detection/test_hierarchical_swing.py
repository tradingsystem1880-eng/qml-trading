"""
Unit Tests for Hierarchical Swing Detection
============================================
Tests for Phase 7.6 hierarchical swing detector.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.detection.hierarchical_swing import (
    HierarchicalSwingDetector,
    HierarchicalSwingConfig,
    SwingCandidate,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame with clear swing points."""
    # Create 200 bars of synthetic data
    n_bars = 200
    dates = [datetime(2024, 1, 1) + timedelta(hours=4*i) for i in range(n_bars)]

    # Create a pattern with clear swings:
    # Uptrend -> Higher High (P1) -> Lower Low (P2) -> Higher High (P3/Head)
    # -> Lower Low (P4/BOS) -> Higher High (P5)
    np.random.seed(42)

    close = [100.0]  # Start at 100

    # Phase 1: Uptrend (bars 0-50)
    for i in range(1, 50):
        close.append(close[-1] + np.random.uniform(0.1, 0.5))

    # Phase 2: P1 swing high (bars 50-60)
    peak1 = close[-1] + 3.0
    for i in range(10):
        if i < 5:
            close.append(close[-1] + (peak1 - close[-1]) * 0.3)
        else:
            close.append(close[-1] - 0.3)

    # Phase 3: P2 swing low (bars 60-80)
    valley1 = close[-1] - 4.0
    for i in range(20):
        if i < 10:
            close.append(close[-1] - (close[-1] - valley1) * 0.15)
        else:
            close.append(close[-1] + 0.2)

    # Phase 4: P3 head - higher than P1 (bars 80-100)
    peak2 = max(close) + 5.0  # Higher than P1
    for i in range(20):
        if i < 10:
            close.append(close[-1] + (peak2 - close[-1]) * 0.15)
        else:
            close.append(close[-1] - 0.3)

    # Phase 5: P4 BOS - lower than P2 (bars 100-130)
    valley2 = valley1 - 2.0  # BOS below P2
    for i in range(30):
        if i < 15:
            close.append(close[-1] - (close[-1] - valley2) * 0.12)
        else:
            close.append(close[-1] + 0.2)

    # Phase 6: P5 swing high (bars 130-150)
    peak3 = close[-1] + 4.0
    for i in range(20):
        if i < 10:
            close.append(close[-1] + (peak3 - close[-1]) * 0.15)
        else:
            close.append(close[-1] - 0.2)

    # Fill remaining bars
    while len(close) < n_bars:
        close.append(close[-1] + np.random.uniform(-0.3, 0.3))

    close = np.array(close[:n_bars])

    # Create OHLCV
    high = close + np.random.uniform(0.5, 1.5, n_bars)
    low = close - np.random.uniform(0.5, 1.5, n_bars)
    open_ = close + np.random.uniform(-0.3, 0.3, n_bars)
    volume = np.random.uniform(1000, 5000, n_bars)

    df = pd.DataFrame({
        'time': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })

    return df


@pytest.fixture
def default_config():
    """Create default hierarchical swing config."""
    return HierarchicalSwingConfig()


@pytest.fixture
def strict_config():
    """Create strict config that filters more aggressively."""
    return HierarchicalSwingConfig(
        min_bar_separation=8,
        min_move_atr=1.5,
        forward_confirm_pct=0.4,
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestHierarchicalSwingDetector:
    """Test hierarchical swing detection."""

    def test_detect_returns_list(self, sample_ohlcv_df, default_config):
        """Test that detect returns a list of swing points."""
        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(sample_ohlcv_df)

        assert isinstance(swings, list)
        assert len(swings) > 0

    def test_swing_points_have_required_attributes(self, sample_ohlcv_df, default_config):
        """Test that each swing point has all required attributes."""
        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(sample_ohlcv_df)

        for swing in swings:
            assert hasattr(swing, 'bar_index')
            assert hasattr(swing, 'price')
            assert hasattr(swing, 'timestamp')
            assert hasattr(swing, 'swing_type')
            assert hasattr(swing, 'atr_at_formation')
            assert hasattr(swing, 'significance_atr')
            assert hasattr(swing, 'significance_zscore')

    def test_swing_types_are_valid(self, sample_ohlcv_df, default_config):
        """Test that all swing types are HIGH or LOW."""
        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(sample_ohlcv_df)

        for swing in swings:
            assert swing.swing_type in ['HIGH', 'LOW']

    def test_swings_sorted_by_bar_index(self, sample_ohlcv_df, default_config):
        """Test that swings are sorted by bar index."""
        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(sample_ohlcv_df)

        bar_indices = [s.bar_index for s in swings]
        assert bar_indices == sorted(bar_indices)

    def test_min_bar_separation_enforced(self, sample_ohlcv_df):
        """Test that minimum bar separation is enforced."""
        config = HierarchicalSwingConfig(min_bar_separation=10)
        detector = HierarchicalSwingDetector(config=config)
        swings = detector.detect(sample_ohlcv_df)

        for i in range(1, len(swings)):
            separation = swings[i].bar_index - swings[i-1].bar_index
            assert separation >= config.min_bar_separation

    def test_strict_config_filters_more(self, sample_ohlcv_df, default_config, strict_config):
        """Test that stricter config produces fewer swings."""
        default_detector = HierarchicalSwingDetector(config=default_config)
        strict_detector = HierarchicalSwingDetector(config=strict_config)

        default_swings = default_detector.detect(sample_ohlcv_df)
        strict_swings = strict_detector.detect(sample_ohlcv_df)

        # Strict should have fewer or equal swings
        assert len(strict_swings) <= len(default_swings)


# =============================================================================
# ATR AND ADX CALCULATION TESTS
# =============================================================================

class TestIndicatorCalculations:
    """Test ATR and ADX calculations."""

    def test_atr_positive(self, sample_ohlcv_df, default_config):
        """Test that ATR values are positive."""
        detector = HierarchicalSwingDetector(config=default_config)
        atr = detector._calculate_atr(sample_ohlcv_df)

        # Skip initial NaN period
        valid_atr = atr[default_config.atr_period:]
        assert all(a > 0 for a in valid_atr if not np.isnan(a))

    def test_adx_in_valid_range(self, sample_ohlcv_df, default_config):
        """Test that ADX values are in valid range (0-100)."""
        detector = HierarchicalSwingDetector(config=default_config)
        adx = detector._calculate_adx(sample_ohlcv_df)

        # Skip initial warmup period
        valid_adx = adx[default_config.adx_period * 2:]
        for a in valid_adx:
            if not np.isnan(a):
                assert 0 <= a <= 100


# =============================================================================
# LAYER 1: GEOMETRY TESTS
# =============================================================================

class TestLayer1Geometry:
    """Test Layer 1 (geometry) swing detection."""

    def test_local_highs_are_local_maxima(self, sample_ohlcv_df, default_config):
        """Test that detected highs are local maxima."""
        detector = HierarchicalSwingDetector(config=default_config)
        atr = detector._calculate_atr(sample_ohlcv_df)
        highs = detector._find_local_highs(sample_ohlcv_df, atr)

        high_prices = sample_ohlcv_df['high'].values

        for candidate in highs:
            idx = candidate.bar_index
            lookback = default_config.lookback
            lookforward = default_config.lookforward

            window_start = max(0, idx - lookback)
            window_end = min(len(high_prices), idx + lookforward + 1)
            window = high_prices[window_start:window_end]

            # The candidate should be >= all values in window
            assert candidate.price >= max(window) - 1e-10

    def test_local_lows_are_local_minima(self, sample_ohlcv_df, default_config):
        """Test that detected lows are local minima."""
        detector = HierarchicalSwingDetector(config=default_config)
        atr = detector._calculate_atr(sample_ohlcv_df)
        lows = detector._find_local_lows(sample_ohlcv_df, atr)

        low_prices = sample_ohlcv_df['low'].values

        for candidate in lows:
            idx = candidate.bar_index
            lookback = default_config.lookback
            lookforward = default_config.lookforward

            window_start = max(0, idx - lookback)
            window_end = min(len(low_prices), idx + lookforward + 1)
            window = low_prices[window_start:window_end]

            # The candidate should be <= all values in window
            assert candidate.price <= min(window) + 1e-10


# =============================================================================
# LAYER 2: SIGNIFICANCE TESTS
# =============================================================================

class TestLayer2Significance:
    """Test Layer 2 (significance) filtering."""

    def test_significance_filtering(self, sample_ohlcv_df):
        """Test that significance filtering reduces candidates."""
        config = HierarchicalSwingConfig(min_move_atr=1.0)
        detector = HierarchicalSwingDetector(config=config)
        atr = detector._calculate_atr(sample_ohlcv_df)

        # Get raw candidates
        raw_highs = detector._find_local_highs(sample_ohlcv_df, atr)

        # Filter by significance
        significant_highs = detector._filter_by_significance(
            raw_highs, sample_ohlcv_df, atr, 'HIGH'
        )

        # Should have fewer or equal after filtering
        assert len(significant_highs) <= len(raw_highs)


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestSwingStats:
    """Test swing statistics calculation."""

    def test_get_swing_stats_empty(self, default_config):
        """Test stats for empty swing list."""
        detector = HierarchicalSwingDetector(config=default_config)
        stats = detector.get_swing_stats([])

        assert stats['total'] == 0
        assert stats['highs'] == 0
        assert stats['lows'] == 0

    def test_get_swing_stats_valid(self, sample_ohlcv_df, default_config):
        """Test stats for valid swing list."""
        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(sample_ohlcv_df)
        stats = detector.get_swing_stats(swings)

        assert stats['total'] == len(swings)
        assert stats['highs'] >= 0
        assert stats['lows'] >= 0
        assert stats['highs'] + stats['lows'] == stats['total']
        assert 'mean_significance' in stats
        assert 'mean_zscore' in stats


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_dataframe(self, default_config):
        """Test handling of too-short DataFrame."""
        short_df = pd.DataFrame({
            'time': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)],
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10,
        })

        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(short_df)

        # Should return empty list for short data
        assert swings == []

    def test_flat_price_data(self, default_config):
        """Test handling of flat price data."""
        flat_df = pd.DataFrame({
            'time': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(200)],
            'open': [100.0] * 200,
            'high': [100.1] * 200,
            'low': [99.9] * 200,
            'close': [100.0] * 200,
            'volume': [1000] * 200,
        })

        detector = HierarchicalSwingDetector(config=default_config)
        swings = detector.detect(flat_df)

        # Should handle flat data gracefully (may return empty or few swings)
        assert isinstance(swings, list)

    def test_symbol_and_timeframe_preserved(self, sample_ohlcv_df, default_config):
        """Test that symbol and timeframe are preserved in swing points."""
        detector = HierarchicalSwingDetector(
            config=default_config,
            symbol='BTCUSDT',
            timeframe='4h',
        )
        swings = detector.detect(sample_ohlcv_df)

        for swing in swings:
            assert swing.symbol == 'BTCUSDT'
            assert swing.timeframe == '4h'
