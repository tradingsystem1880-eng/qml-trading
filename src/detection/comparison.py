"""
Framework for comparing detection algorithm variations.

Integrates with Phase 6 Experiment Lab for systematic A/B testing
of detection parameters.
"""

import pandas as pd
from typing import List, Optional
from dataclasses import dataclass

from .swing_algorithms import SwingAlgorithm
from .qml_pattern import QMLPatternDetector, QMLConfig, QMLPattern


@dataclass
class DetectionComparison:
    """Results from comparing two detection configurations."""
    config_a: QMLConfig
    config_b: QMLConfig

    patterns_a: int
    patterns_b: int
    overlap_count: int
    unique_to_a: int
    unique_to_b: int

    jaccard_similarity: float


def compare_detections(
    df: pd.DataFrame,
    config_a: QMLConfig,
    config_b: QMLConfig,
    tolerance_bars: int = 3
) -> DetectionComparison:
    """
    Compare two detection configurations on the same data.

    Args:
        df: OHLCV DataFrame
        config_a: First configuration
        config_b: Second configuration
        tolerance_bars: Number of bars tolerance for matching patterns

    Returns:
        DetectionComparison with overlap statistics
    """
    detector_a = QMLPatternDetector(config_a)
    detector_b = QMLPatternDetector(config_b)

    patterns_a = detector_a.detect(df)
    patterns_b = detector_b.detect(df)

    # Use P3 (head) index for pattern matching
    a_indices = {p.p3.index for p in patterns_a}
    b_indices = {p.p3.index for p in patterns_b}

    # Count overlapping patterns (within tolerance)
    overlap = 0
    matched_b = set()

    for a_idx in a_indices:
        for b_idx in b_indices:
            if abs(a_idx - b_idx) <= tolerance_bars and b_idx not in matched_b:
                overlap += 1
                matched_b.add(b_idx)
                break

    # Calculate Jaccard similarity
    union = len(a_indices) + len(b_indices) - overlap
    jaccard = overlap / union if union > 0 else 0.0

    return DetectionComparison(
        config_a=config_a,
        config_b=config_b,
        patterns_a=len(patterns_a),
        patterns_b=len(patterns_b),
        overlap_count=overlap,
        unique_to_a=len(patterns_a) - overlap,
        unique_to_b=len(patterns_b) - overlap,
        jaccard_similarity=jaccard
    )


def qml_config_from_parameter_set(params) -> QMLConfig:
    """
    Convert Phase 6 ParameterSet to QMLConfig.

    Maps experiment parameters to detection configuration.

    Args:
        params: ParameterSet from src/experiments/parameter_grid.py

    Returns:
        QMLConfig for detection
    """
    # Algorithm mapping
    algo_map = {
        'rolling': SwingAlgorithm.ROLLING,
        'savgol': SwingAlgorithm.SAVGOL,
        'fractal': SwingAlgorithm.FRACTAL,
        'wavelet': SwingAlgorithm.WAVELET
    }

    # Get algorithm (default to rolling)
    swing_algo = getattr(params, 'swing_algorithm', 'rolling')
    algorithm = algo_map.get(swing_algo, SwingAlgorithm.ROLLING)

    return QMLConfig(
        # Swing detection
        swing_algorithm=algorithm,
        swing_lookback=getattr(params, 'swing_lookback', 5),
        smoothing_window=getattr(params, 'smoothing_window', 5),

        # Pattern validation
        min_head_extension_atr=getattr(params, 'min_head_extension_atr', 0.5),
        bos_requirement=getattr(params, 'bos_requirement', 1),
        max_shoulder_tolerance_atr=getattr(params, 'shoulder_tolerance_atr', 1.0),

        # Entry/Exit
        sl_buffer_atr=getattr(params, 'sl_placement_atr', 0.5),
        tp1_r_multiple=getattr(params, 'tp_r_multiple', 1.5),
        tp2_r_multiple=getattr(params, 'tp_r_multiple', 1.5) + 1.0,

        # Filters
        require_trend_alignment=True,
        require_volume_confirmation=getattr(params, 'volume_filter', 'none') != 'none'
    )


def batch_compare(
    df: pd.DataFrame,
    configs: List[QMLConfig],
    tolerance_bars: int = 3
) -> pd.DataFrame:
    """
    Compare multiple configurations against each other.

    Args:
        df: OHLCV DataFrame
        configs: List of QMLConfig to compare
        tolerance_bars: Number of bars tolerance for matching

    Returns:
        DataFrame with pairwise comparison results
    """
    results = []

    for i, config_a in enumerate(configs):
        for j, config_b in enumerate(configs):
            if i >= j:  # Skip self-comparison and duplicates
                continue

            comparison = compare_detections(df, config_a, config_b, tolerance_bars)

            results.append({
                'config_a_idx': i,
                'config_b_idx': j,
                'patterns_a': comparison.patterns_a,
                'patterns_b': comparison.patterns_b,
                'overlap': comparison.overlap_count,
                'unique_a': comparison.unique_to_a,
                'unique_b': comparison.unique_to_b,
                'jaccard': comparison.jaccard_similarity
            })

    return pd.DataFrame(results)


def analyze_algorithm_differences(
    df: pd.DataFrame,
    base_config: Optional[QMLConfig] = None
) -> pd.DataFrame:
    """
    Compare all swing algorithms with the same base configuration.

    Args:
        df: OHLCV DataFrame
        base_config: Base configuration (uses defaults if None)

    Returns:
        DataFrame with algorithm comparison results
    """
    if base_config is None:
        base_config = QMLConfig()

    results = []
    algorithms = [
        SwingAlgorithm.ROLLING,
        SwingAlgorithm.SAVGOL,
        SwingAlgorithm.FRACTAL
    ]

    # Try to include wavelet if available
    try:
        import pywt  # noqa: F401
        algorithms.append(SwingAlgorithm.WAVELET)
    except ImportError:
        pass

    for algo in algorithms:
        config = QMLConfig(
            swing_algorithm=algo,
            swing_lookback=base_config.swing_lookback,
            smoothing_window=base_config.smoothing_window,
            min_head_extension_atr=base_config.min_head_extension_atr,
            bos_requirement=base_config.bos_requirement,
            max_shoulder_tolerance_atr=base_config.max_shoulder_tolerance_atr,
            require_trend_alignment=base_config.require_trend_alignment
        )

        detector = QMLPatternDetector(config)
        patterns = detector.detect(df)

        bullish = len([p for p in patterns if p.direction.value == 'BULLISH'])
        bearish = len([p for p in patterns if p.direction.value == 'BEARISH'])

        avg_strength = (
            sum(p.pattern_strength for p in patterns) / len(patterns)
            if patterns else 0.0
        )

        results.append({
            'algorithm': algo.value,
            'total_patterns': len(patterns),
            'bullish': bullish,
            'bearish': bearish,
            'avg_strength': round(avg_strength, 3)
        })

    return pd.DataFrame(results)
