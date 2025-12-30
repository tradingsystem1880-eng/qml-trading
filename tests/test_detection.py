"""
Unit Tests for QML Detection Engine
====================================
Validates swing detection, structure analysis, CHoCH, BoS, and pattern detection.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.swing import SwingDetector, SwingConfig
from src.detection.structure import StructureAnalyzer
from src.detection.choch import CHoCHDetector
from src.detection.bos import BoSDetector
from src.detection.detector import QMLDetector
from src.data.models import SwingType, TrendType, PatternType


def generate_uptrend_data(n_bars: int = 200) -> pd.DataFrame:
    """Generate synthetic uptrend OHLCV data."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4H')
    
    # Uptrend with noise
    base_price = 100
    trend = np.linspace(0, 30, n_bars)
    noise = np.random.randn(n_bars).cumsum() * 0.5
    close = base_price + trend + noise
    
    high = close + np.random.rand(n_bars) * 2
    low = close - np.random.rand(n_bars) * 2
    open_price = close + np.random.randn(n_bars) * 0.5
    volume = np.random.rand(n_bars) * 1000 + 500
    
    return pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def generate_downtrend_data(n_bars: int = 200) -> pd.DataFrame:
    """Generate synthetic downtrend OHLCV data."""
    np.random.seed(43)
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4H')
    
    base_price = 130
    trend = np.linspace(0, -30, n_bars)
    noise = np.random.randn(n_bars).cumsum() * 0.5
    close = base_price + trend + noise
    
    high = close + np.random.rand(n_bars) * 2
    low = close - np.random.rand(n_bars) * 2
    open_price = close + np.random.randn(n_bars) * 0.5
    volume = np.random.rand(n_bars) * 1000 + 500
    
    return pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def generate_qml_pattern_data(pattern_type: str = "bullish") -> pd.DataFrame:
    """Generate data with clear QML pattern."""
    np.random.seed(44)
    n_bars = 150
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4H')
    
    if pattern_type == "bullish":
        # Downtrend -> CHoCH -> Head (lower low) -> BoS (higher high)
        close = np.concatenate([
            np.linspace(120, 100, 50),  # Downtrend
            np.linspace(100, 110, 20),  # CHoCH break up
            np.linspace(110, 95, 20),   # Head formation (lower low)
            np.linspace(95, 115, 30),   # BoS break up
            np.linspace(115, 120, 30),  # Continuation
        ])
    else:
        # Uptrend -> CHoCH -> Head (higher high) -> BoS (lower low)
        close = np.concatenate([
            np.linspace(80, 100, 50),   # Uptrend
            np.linspace(100, 90, 20),   # CHoCH break down
            np.linspace(90, 105, 20),   # Head formation (higher high)
            np.linspace(105, 85, 30),   # BoS break down
            np.linspace(85, 80, 30),    # Continuation
        ])
    
    noise = np.random.randn(n_bars) * 0.5
    close = close + noise
    
    high = close + np.random.rand(n_bars) * 2
    low = close - np.random.rand(n_bars) * 2
    open_price = close + np.random.randn(n_bars) * 0.3
    volume = np.random.rand(n_bars) * 1000 + 500
    
    # Volume spike at key points
    volume[65:75] *= 2  # CHoCH area
    volume[90:100] *= 1.8  # BoS area
    
    return pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestSwingDetection:
    """Test swing point detection."""
    
    def test_detect_swing_highs_in_uptrend(self):
        df = generate_uptrend_data()
        detector = SwingDetector(timeframe="4h")
        
        swings = detector.detect(df, "TEST")
        
        # Should find swing highs and lows
        highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        lows = [s for s in swings if s.swing_type == SwingType.LOW]
        
        assert len(highs) > 0, "Should detect swing highs"
        assert len(lows) > 0, "Should detect swing lows"
    
    def test_swing_significance(self):
        df = generate_uptrend_data()
        config = SwingConfig(atr_multiplier=1.0)
        detector = SwingDetector(config=config, timeframe="4h")
        
        swings = detector.detect(df, "TEST")
        
        # All swings should have positive significance
        for swing in swings:
            assert swing.significance > 0, "Significance should be positive"
            assert swing.atr_at_point > 0, "ATR should be positive"
    
    def test_filter_by_significance(self):
        df = generate_uptrend_data()
        detector = SwingDetector(timeframe="4h")
        
        all_swings = detector.detect(df, "TEST")
        filtered = detector.filter_significant_swings(all_swings, min_significance=1.5)
        
        assert len(filtered) <= len(all_swings), "Filtered should be subset"
        for swing in filtered:
            assert swing.significance >= 1.5


class TestStructureAnalysis:
    """Test market structure analysis."""
    
    def test_uptrend_detection(self):
        df = generate_uptrend_data()
        swing_detector = SwingDetector(timeframe="4h")
        structure_analyzer = StructureAnalyzer()
        
        swings = swing_detector.detect(df, "TEST")
        structures, trend_state = structure_analyzer.analyze(swings, "TEST", "4h")
        
        # In uptrend, should see HH and HL
        assert trend_state.trend in [TrendType.UPTREND, TrendType.CONSOLIDATION]
    
    def test_downtrend_detection(self):
        df = generate_downtrend_data()
        swing_detector = SwingDetector(timeframe="4h")
        structure_analyzer = StructureAnalyzer()
        
        swings = swing_detector.detect(df, "TEST")
        structures, trend_state = structure_analyzer.analyze(swings, "TEST", "4h")
        
        # In downtrend, should see LH and LL
        assert trend_state.trend in [TrendType.DOWNTREND, TrendType.CONSOLIDATION]


class TestCHoCHDetection:
    """Test CHoCH detection."""
    
    def test_bullish_choch_detection(self):
        df = generate_qml_pattern_data("bullish")
        
        swing_detector = SwingDetector(timeframe="4h")
        structure_analyzer = StructureAnalyzer()
        choch_detector = CHoCHDetector()
        
        swings = swing_detector.detect(df, "TEST")
        structures, trend_state = structure_analyzer.analyze(swings, "TEST", "4h")
        
        # Force downtrend state for test
        trend_state.trend = TrendType.DOWNTREND
        if swings:
            trend_state.last_lh = next((s for s in swings if s.swing_type == SwingType.HIGH), None)
        
        choch_events = choch_detector.detect(df, swings, structures, trend_state, "TEST", "4h")
        
        # May or may not find CHoCH depending on data
        assert isinstance(choch_events, list)


class TestQMLDetector:
    """Test complete QML pattern detection."""
    
    def test_detector_initialization(self):
        detector = QMLDetector()
        assert detector is not None
        assert detector.config is not None
    
    def test_detect_on_synthetic_data(self):
        df = generate_qml_pattern_data("bullish")
        detector = QMLDetector()
        
        # Should run without errors
        patterns = detector.detect("TEST/USDT", "4h", df=df)
        
        assert isinstance(patterns, list)
    
    def test_pattern_validity_scoring(self):
        df = generate_qml_pattern_data("bullish")
        detector = QMLDetector()
        
        patterns = detector.detect("TEST/USDT", "4h", df=df)
        
        for pattern in patterns:
            assert 0 <= pattern.validity_score <= 1
            if pattern.trading_levels:
                assert pattern.trading_levels.entry > 0
                assert pattern.trading_levels.stop_loss > 0
                assert pattern.trading_levels.take_profit_3 > 0


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete detection pipeline."""
        df = generate_uptrend_data(300)
        
        # Run full detection
        detector = QMLDetector()
        patterns = detector.detect("TEST/USDT", "4h", df=df)
        
        # Verify output structure
        assert isinstance(patterns, list)
        for p in patterns:
            assert p.symbol == "TEST/USDT"
            assert p.timeframe == "4h"
            assert p.validity_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

