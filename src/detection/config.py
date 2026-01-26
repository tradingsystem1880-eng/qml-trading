"""
Detection Configuration Classes
================================
Centralized configuration dataclasses for ML-optimizable parameters.

All parameters are designed to be tunable for hyperparameter optimization.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SwingDetectionConfig:
    """
    Configuration for swing point detection.

    These parameters control how swing highs/lows are identified.
    All thresholds are designed to be ML-optimizable.
    """
    # ATR calculation
    atr_period: int = 14

    # Swing detection window
    lookback: int = 5  # Bars before candidate
    lookforward: int = 3  # Bars after candidate (for historical/batch mode)

    # Significance thresholds
    min_zscore: float = 1.5  # Minimum z-score for swing significance
    min_threshold_pct: float = 0.001  # 0.1% absolute floor as fraction of price

    # ATR multiplier for significance
    atr_multiplier: float = 1.0

    # Confirmation settings
    require_confirmation: bool = True
    confirmation_bars: int = 2

    def __post_init__(self):
        """Validate configuration values."""
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.lookforward < 0:
            raise ValueError("lookforward must be >= 0")
        if self.min_zscore < 0:
            raise ValueError("min_zscore must be >= 0")
        if not (0 < self.min_threshold_pct < 1):
            raise ValueError("min_threshold_pct must be between 0 and 1")


@dataclass
class PatternValidationConfig:
    """
    Configuration for pattern geometry validation.

    These thresholds determine whether a candidate 5-point pattern
    is geometrically valid. All distances are ATR-normalized.
    """
    # P3 (Head) extension requirements
    p3_min_extension_atr: float = 0.5  # Head must extend beyond P1 by at least this
    p3_max_extension_atr: float = 5.0  # Head extension shouldn't be extreme

    # P4 (BOS) requirements
    p4_min_break_atr: float = 0.1  # P4 must break P2 by at least this
    bos_requirement: int = 1  # Minimum number of BOS confirmations

    # P5 (Right shoulder) symmetry
    p5_max_symmetry_atr: float = 2.0  # Max difference between P5 and P1 (ATR-normalized)

    # Pattern duration limits
    min_pattern_bars: int = 10
    max_pattern_bars: int = 100

    # Entry/Exit buffer
    entry_buffer_atr: float = 0.1
    sl_buffer_atr: float = 0.5

    # Take profit R-multiples
    tp1_r_multiple: float = 1.5
    tp2_r_multiple: float = 2.5
    tp3_r_multiple: float = 3.5

    def __post_init__(self):
        """Validate configuration values."""
        if self.p3_min_extension_atr >= self.p3_max_extension_atr:
            raise ValueError("p3_min_extension_atr must be < p3_max_extension_atr")
        if self.min_pattern_bars >= self.max_pattern_bars:
            raise ValueError("min_pattern_bars must be < max_pattern_bars")


@dataclass
class PatternScoringConfig:
    """
    Configuration for Gaussian pattern quality scoring.

    Uses Gaussian scoring functions instead of step functions for smoother
    ML optimization. Each metric has:
    - optimal: The ideal value (Gaussian peak)
    - width: Controls how sharply score drops from optimal (sigma)
    - weight: Contribution to total score (weights sum to 1.0)

    Extended in Phase 7.6 with volume, path efficiency, and trend metrics.
    Extended in Phase 7.8 with regime suitability.
    """
    # Head extension scoring (optimal: 1-2 ATR)
    head_extension_optimal: float = 1.5
    head_extension_width: float = 0.8  # sigma
    head_extension_weight: float = 0.22  # Adjusted for 8 components

    # BOS efficiency scoring (optimal: clean, single break)
    bos_efficiency_optimal: float = 0.9
    bos_efficiency_width: float = 0.3
    bos_efficiency_weight: float = 0.18  # Adjusted for 8 components

    # Shoulder symmetry scoring (optimal: 0, perfectly symmetric)
    shoulder_symmetry_optimal: float = 0.0
    shoulder_symmetry_width: float = 0.8
    shoulder_symmetry_weight: float = 0.12  # Adjusted for 8 components

    # Swing significance scoring (average z-score of swing points)
    swing_significance_optimal: float = 2.5
    swing_significance_width: float = 1.0
    swing_significance_weight: float = 0.08  # Adjusted for 8 components

    # Phase 7.6 metrics (now optimizable) ----------------------------------

    # Volume spike scoring (optimal: 2x average volume at P3/P4)
    volume_spike_optimal: float = 2.0
    volume_spike_width: float = 1.0
    volume_spike_weight: float = 0.10

    # Path efficiency scoring (optimal: 0.7 = 70% direct movement)
    path_efficiency_optimal: float = 0.7
    path_efficiency_width: float = 0.2
    path_efficiency_weight: float = 0.10

    # Trend strength scoring (combined ADX + move size)
    trend_strength_optimal: float = 0.8
    trend_strength_width: float = 0.3
    trend_strength_weight: float = 0.10

    # Phase 7.8 regime suitability -----------------------------------------

    # Regime base scores (how favorable each regime is for QML patterns)
    regime_ranging_score: float = 1.0      # Ideal for QML
    regime_volatile_score: float = 0.6     # Acceptable with caution
    regime_extreme_score: float = 0.5      # High risk/reward
    regime_trending_score: float = 0.2     # Poor for QML

    # Hard rejection threshold (skip pattern if TRENDING + ADX > this)
    regime_hard_reject_adx: float = 35.0

    # Regime suitability weight in total score
    regime_suitability_weight: float = 0.10

    # Quality tier thresholds
    tier_a_min: float = 0.80  # A-tier patterns
    tier_b_min: float = 0.60  # B-tier patterns
    tier_c_min: float = 0.40  # C-tier patterns
    # Below tier_c_min = rejected

    def __post_init__(self):
        """Validate configuration values."""
        # Weights should sum to 1.0 (now 8 components)
        total_weight = (
            self.head_extension_weight +
            self.bos_efficiency_weight +
            self.shoulder_symmetry_weight +
            self.swing_significance_weight +
            self.volume_spike_weight +
            self.path_efficiency_weight +
            self.trend_strength_weight +
            self.regime_suitability_weight
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")

        # Tier thresholds must be ordered
        if not (self.tier_a_min > self.tier_b_min > self.tier_c_min > 0):
            raise ValueError("Tier thresholds must be ordered: A > B > C > 0")


@dataclass
class DetectionConfig:
    """
    Master configuration aggregating all detection parameters.

    This is the main config class that should be passed to the detector.
    Sub-configs can be customized independently.
    """
    swing: SwingDetectionConfig = field(default_factory=SwingDetectionConfig)
    validation: PatternValidationConfig = field(default_factory=PatternValidationConfig)
    scoring: PatternScoringConfig = field(default_factory=PatternScoringConfig)

    # Detector behavior
    emit_unconfirmed: bool = False  # Emit patterns before P5 is confirmed
    require_trend_alignment: bool = True  # Filter patterns in strong trends

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DetectionConfig":
        """
        Create DetectionConfig from a nested dictionary.

        Useful for loading from YAML or JSON configuration files.

        Args:
            config_dict: Dictionary with 'swing', 'validation', 'scoring' keys

        Returns:
            Configured DetectionConfig instance
        """
        swing_config = SwingDetectionConfig(
            **config_dict.get("swing", {})
        )
        validation_config = PatternValidationConfig(
            **config_dict.get("validation", {})
        )
        scoring_config = PatternScoringConfig(
            **config_dict.get("scoring", {})
        )

        return cls(
            swing=swing_config,
            validation=validation_config,
            scoring=scoring_config,
            emit_unconfirmed=config_dict.get("emit_unconfirmed", False),
            require_trend_alignment=config_dict.get("require_trend_alignment", True),
        )

    def to_dict(self) -> dict:
        """
        Export configuration to dictionary.

        Useful for saving to YAML or JSON.

        Returns:
            Nested dictionary representation
        """
        from dataclasses import asdict
        return asdict(self)


# Default configuration instance
DEFAULT_CONFIG = DetectionConfig()
