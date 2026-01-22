"""Test Pattern Lab integration with Phase 8 detection.

Run with: python3 tests/test_pattern_lab_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def test_detection_imports():
    """Test that all Phase 8 detection components import correctly."""
    print("Test 1: Detection imports...")
    from src.detection import (
        QMLPatternDetector, QMLConfig, QMLPattern, PatternDirection,
        SwingAlgorithm, MarketRegimeDetector, RegimeResult
    )
    print("  ✓ All imports successful")
    return True


def test_synthetic_data_generation():
    """Test synthetic OHLCV data generation."""
    print("Test 2: Synthetic data generation...")

    # Create synthetic data similar to pattern_lab_page.py
    import random
    from datetime import datetime, timedelta

    base_price = 97000
    volatility = 0.015
    bars = 200
    data = []
    current_price = base_price
    now = datetime.now()

    for i in range(bars):
        time = now - timedelta(hours=(bars - i) * 4)
        change = random.gauss(0, volatility)
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility * 0.5)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility * 0.5)))

        data.append({
            'time': time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': random.uniform(100, 1000) * 970
        })
        current_price = close_price

    df = pd.DataFrame(data)
    assert len(df) == 200
    assert all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'volume'])
    print(f"  ✓ Generated {len(df)} bars of synthetic data")
    return df


def test_detection_pipeline(df: pd.DataFrame):
    """Test full detection pipeline."""
    print("Test 3: Detection pipeline...")

    from src.detection import (
        QMLPatternDetector, QMLConfig, SwingAlgorithm, MarketRegimeDetector
    )

    config = QMLConfig(
        swing_algorithm=SwingAlgorithm.ROLLING,
        swing_lookback=5,
        min_head_extension_atr=0.5,
        bos_requirement=1,
        require_trend_alignment=False  # Don't filter for testing
    )

    detector = QMLPatternDetector(config)
    patterns = detector.detect(df)

    regime_detector = MarketRegimeDetector()
    regime = regime_detector.get_regime(df)

    print(f"  ✓ Detected {len(patterns)} patterns")
    print(f"  ✓ Market regime: {regime.regime.value} (ADX={regime.adx:.1f})")
    return patterns, regime


def test_all_algorithms(df: pd.DataFrame):
    """Test all swing detection algorithms."""
    print("Test 4: All swing algorithms...")

    from src.detection import QMLPatternDetector, QMLConfig, SwingAlgorithm

    algorithms = ['rolling', 'savgol', 'fractal']  # Skip wavelet (requires PyWavelets)
    algo_map = {
        'rolling': SwingAlgorithm.ROLLING,
        'savgol': SwingAlgorithm.SAVGOL,
        'fractal': SwingAlgorithm.FRACTAL,
    }

    results = {}
    for algo_name in algorithms:
        config = QMLConfig(
            swing_algorithm=algo_map[algo_name],
            require_trend_alignment=False
        )
        detector = QMLPatternDetector(config)
        patterns = detector.detect(df)
        results[algo_name] = len(patterns)
        print(f"  ✓ {algo_name}: {len(patterns)} patterns")

    return results


def test_pattern_data_extraction():
    """Test pattern data extraction for chart display."""
    print("Test 5: Pattern data extraction...")

    from src.detection import QMLPattern, PatternDirection, SwingPoint
    import pandas as pd

    # Create mock pattern
    now = pd.Timestamp.now()
    mock_pattern = QMLPattern(
        id="QML_TEST_1",
        direction=PatternDirection.BEARISH,
        p1=SwingPoint(10, 95000.0, now, 'LOW', 1.0),
        p2=SwingPoint(15, 96500.0, now, 'HIGH', 1.0),
        p3=SwingPoint(20, 94000.0, now, 'LOW', 1.0),
        p4=SwingPoint(25, 97000.0, now, 'HIGH', 1.0),
        p5=SwingPoint(30, 95500.0, now, 'LOW', 1.0),
        entry_price=95400.0,
        stop_loss=93900.0,
        take_profit_1=97900.0,
        head_extension_atr=1.2,
        shoulder_symmetry=0.3,
        bos_count=1,
        pattern_strength=0.85
    )

    # Extract data for chart
    assert mock_pattern.entry_price == 95400.0
    assert mock_pattern.direction == PatternDirection.BEARISH
    assert mock_pattern.pattern_strength == 0.85

    # Calculate R:R
    risk = abs(mock_pattern.stop_loss - mock_pattern.entry_price)
    reward = abs(mock_pattern.take_profit_1 - mock_pattern.entry_price)
    rr = reward / risk

    print(f"  ✓ Pattern ID: {mock_pattern.id}")
    print(f"  ✓ R:R ratio: 1:{rr:.1f}")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Pattern Lab Integration Tests")
    print("=" * 60 + "\n")

    try:
        # Test 1: Imports
        test_detection_imports()

        # Test 2: Data generation
        df = test_synthetic_data_generation()

        # Test 3: Detection pipeline
        patterns, regime = test_detection_pipeline(df)

        # Test 4: All algorithms
        test_all_algorithms(df)

        # Test 5: Data extraction
        test_pattern_data_extraction()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
