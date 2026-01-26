"""
Pattern Quality Scorer
======================
Gaussian-based quality scoring for QML patterns.

Uses smooth Gaussian functions instead of step functions for:
- Better ML optimization landscape (no discontinuities)
- More nuanced quality differentiation
- Interpretable component scores

Extended in Phase 7.6 with:
- Volume spike analysis
- Path efficiency measurement
- Trend strength scoring

Extended in Phase 7.8 with:
- Regime suitability scoring (RANGING/VOLATILE/EXTREME/TRENDING)
- Hard rejection for strong trends (ADX > threshold)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING
import math

import numpy as np
import pandas as pd

from src.detection.config import PatternScoringConfig
from src.detection.pattern_validator import ValidationResult
from src.detection.regime import MarketRegime, RegimeResult

if TYPE_CHECKING:
    from src.detection.trend_validator import TrendValidationResult


class PatternTier(str, Enum):
    """Quality tier classification for patterns."""
    A = "A"  # Excellent - high probability setups
    B = "B"  # Good - solid setups worth considering
    C = "C"  # Acceptable - lower probability but valid
    REJECT = "REJECT"  # Below minimum quality threshold


@dataclass
class ScoringResult:
    """Result of pattern quality scoring."""
    total_score: float  # 0.0 to 1.0
    tier: PatternTier

    # Component scores (each 0.0 to 1.0)
    head_extension_score: float
    bos_efficiency_score: float
    shoulder_symmetry_score: float
    swing_significance_score: float

    # Phase 7.6 component scores
    volume_spike_score: float = 0.0
    path_efficiency_score: float = 0.0
    trend_strength_score: float = 0.0

    # Phase 7.8 regime suitability score
    regime_suitability_score: float = 0.0

    # Debugging info
    head_extension_input: float = 0.0
    bos_efficiency_input: float = 0.0
    shoulder_symmetry_input: float = 0.0
    swing_significance_input: float = 0.0

    # Phase 7.6 inputs
    volume_spike_input: float = 0.0
    path_efficiency_input: float = 0.0
    trend_strength_input: float = 0.0

    # Phase 7.8 regime input
    regime_suitability_input: float = 0.0


class PatternScorer:
    """
    Scores QML patterns using Gaussian quality functions.

    Each metric is scored using a Gaussian centered on the optimal value.
    This provides smooth, differentiable scores suitable for ML optimization.

    The scorer expects PRE-NORMALIZED metrics (caller normalizes by atr_p5).
    """

    def __init__(self, config: Optional[PatternScoringConfig] = None):
        """
        Initialize the scorer.

        Args:
            config: Scoring configuration with optimal values, widths, and weights
        """
        self.config = config or PatternScoringConfig()

    def score(
        self,
        validation_result: ValidationResult,
        df: Optional[pd.DataFrame] = None,
        trend_result: Optional['TrendValidationResult'] = None,
        regime_result: Optional[RegimeResult] = None,
    ) -> ScoringResult:
        """
        Score a validated pattern.

        The ValidationResult must be valid (is_valid=True).
        All metrics in the ValidationResult should already be ATR-normalized.

        Args:
            validation_result: A valid pattern with ATR-normalized metrics
            df: Optional OHLCV DataFrame for volume/path analysis
            trend_result: Optional trend validation result for trend scoring
            regime_result: Optional regime detection result for regime scoring (Phase 7.8)

        Returns:
            ScoringResult with total score, tier, and component scores
        """
        if not validation_result.is_valid:
            return ScoringResult(
                total_score=0.0,
                tier=PatternTier.REJECT,
                head_extension_score=0.0,
                bos_efficiency_score=0.0,
                shoulder_symmetry_score=0.0,
                swing_significance_score=0.0,
            )

        # Phase 7.8: Hard rejection for strong trends
        if regime_result is not None:
            if (regime_result.regime == MarketRegime.TRENDING and
                regime_result.adx > self.config.regime_hard_reject_adx):
                return ScoringResult(
                    total_score=0.0,
                    tier=PatternTier.REJECT,
                    head_extension_score=0.0,
                    bos_efficiency_score=0.0,
                    shoulder_symmetry_score=0.0,
                    swing_significance_score=0.0,
                )

        # Score each component using Gaussian functions
        head_ext_score = self._gaussian_score(
            value=validation_result.head_extension_atr,
            optimal=self.config.head_extension_optimal,
            width=self.config.head_extension_width,
        )

        bos_eff_score = self._gaussian_score(
            value=validation_result.bos_efficiency,
            optimal=self.config.bos_efficiency_optimal,
            width=self.config.bos_efficiency_width,
        )

        # Shoulder symmetry: lower is better (0 is optimal)
        shoulder_score = self._gaussian_score(
            value=validation_result.shoulder_diff_atr,
            optimal=self.config.shoulder_symmetry_optimal,
            width=self.config.shoulder_symmetry_width,
        )

        # Swing significance: use average z-score of P1-P5
        avg_zscore = self._calculate_avg_swing_significance(validation_result)
        swing_sig_score = self._gaussian_score(
            value=avg_zscore,
            optimal=self.config.swing_significance_optimal,
            width=self.config.swing_significance_width,
        )

        # Phase 7.6 metrics
        volume_spike_input = 0.0
        volume_spike_score = 0.5  # Default neutral score
        path_eff_input = 0.0
        path_eff_score = 0.5
        trend_strength_input = 0.0
        trend_strength_score = 0.5

        if df is not None:
            # Volume spike scoring
            volume_spike_input = self._calculate_volume_spike(validation_result, df)
            volume_spike_score = self._gaussian_score(
                value=volume_spike_input,
                optimal=self.config.volume_spike_optimal,
                width=self.config.volume_spike_width,
            )

            # Path efficiency scoring
            path_eff_input = self._calculate_path_efficiency(validation_result, df)
            path_eff_score = self._gaussian_score(
                value=path_eff_input,
                optimal=self.config.path_efficiency_optimal,
                width=self.config.path_efficiency_width,
            )

        if trend_result is not None and trend_result.is_valid:
            # Trend strength scoring
            trend_strength_input = self._calculate_trend_strength(trend_result)
            trend_strength_score = self._gaussian_score(
                value=trend_strength_input,
                optimal=self.config.trend_strength_optimal,
                width=self.config.trend_strength_width,
            )

        # Phase 7.8: Regime suitability scoring
        regime_suitability_input = 0.5  # Default neutral
        regime_suitability_score = 0.5
        if regime_result is not None:
            regime_suitability_input = self._calculate_regime_suitability(regime_result)
            regime_suitability_score = regime_suitability_input  # Already 0-1, no Gaussian needed

        # Calculate weighted total (now 8 components)
        total = (
            head_ext_score * self.config.head_extension_weight +
            bos_eff_score * self.config.bos_efficiency_weight +
            shoulder_score * self.config.shoulder_symmetry_weight +
            swing_sig_score * self.config.swing_significance_weight +
            volume_spike_score * self.config.volume_spike_weight +
            path_eff_score * self.config.path_efficiency_weight +
            trend_strength_score * self.config.trend_strength_weight +
            regime_suitability_score * self.config.regime_suitability_weight
        )

        # Determine tier
        tier = self._determine_tier(total)

        return ScoringResult(
            total_score=total,
            tier=tier,
            head_extension_score=head_ext_score,
            bos_efficiency_score=bos_eff_score,
            shoulder_symmetry_score=shoulder_score,
            swing_significance_score=swing_sig_score,
            volume_spike_score=volume_spike_score,
            path_efficiency_score=path_eff_score,
            trend_strength_score=trend_strength_score,
            regime_suitability_score=regime_suitability_score,
            head_extension_input=validation_result.head_extension_atr,
            bos_efficiency_input=validation_result.bos_efficiency,
            shoulder_symmetry_input=validation_result.shoulder_diff_atr,
            swing_significance_input=avg_zscore,
            volume_spike_input=volume_spike_input,
            path_efficiency_input=path_eff_input,
            trend_strength_input=trend_strength_input,
            regime_suitability_input=regime_suitability_input,
        )

    def _gaussian_score(
        self,
        value: float,
        optimal: float,
        width: float
    ) -> float:
        """
        Calculate Gaussian score for a metric.

        Score = exp(-(value - optimal)^2 / (2 * width^2))

        This gives:
        - Score of 1.0 when value == optimal
        - Score decays smoothly as value moves away from optimal
        - Width controls the decay rate (larger = more forgiving)

        Args:
            value: The actual metric value
            optimal: The ideal/optimal value (Gaussian center)
            width: Controls decay rate (Gaussian sigma)

        Returns:
            Score from 0.0 to 1.0
        """
        if width <= 0:
            return 1.0 if abs(value - optimal) < 0.001 else 0.0

        deviation = value - optimal
        exponent = -(deviation ** 2) / (2 * width ** 2)
        return math.exp(exponent)

    def _calculate_avg_swing_significance(
        self,
        validation_result: ValidationResult
    ) -> float:
        """
        Calculate average swing significance across P1-P5.

        Uses z-scores for cross-timeframe comparability.

        Args:
            validation_result: Validated pattern with swing points

        Returns:
            Average significance z-score
        """
        zscores = []

        for point in [
            validation_result.p1,
            validation_result.p2,
            validation_result.p3,
            validation_result.p4,
            validation_result.p5,
        ]:
            if point is not None and hasattr(point, 'significance_zscore'):
                zscores.append(point.significance_zscore)

        if not zscores:
            return 0.0

        return sum(zscores) / len(zscores)

    def _calculate_volume_spike(
        self,
        validation_result: ValidationResult,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate volume spike ratio at P3/P4.

        Compares volume at key swing points to average volume.

        Args:
            validation_result: Validated pattern
            df: OHLCV DataFrame

        Returns:
            Volume spike ratio (e.g., 2.0 = 2x average)
        """
        try:
            vol_col = 'volume' if 'volume' in df.columns else 'Volume'
            if vol_col not in df.columns:
                return 1.0  # Default if no volume data

            p1_idx = validation_result.p1.bar_index
            p3_idx = validation_result.p3.bar_index
            p4_idx = validation_result.p4.bar_index
            p5_idx = validation_result.p5.bar_index

            # Average volume in the pattern window
            window_start = max(0, p1_idx - 20)
            avg_vol = df.iloc[window_start:p5_idx][vol_col].mean()

            if avg_vol <= 0:
                return 1.0

            # Volume at P3 and P4 (key reversal points)
            vol_p3 = df.iloc[p3_idx][vol_col]
            vol_p4 = df.iloc[p4_idx][vol_col]

            max_spike = max(vol_p3, vol_p4) / avg_vol

            return float(max_spike)

        except Exception:
            return 1.0

    def _calculate_path_efficiency(
        self,
        validation_result: ValidationResult,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate path efficiency from P1 to P3.

        Efficiency = direct_distance / total_path_length

        A value close to 1.0 means a clean, direct move.
        Lower values indicate choppy, inefficient price action.

        Args:
            validation_result: Validated pattern
            df: OHLCV DataFrame

        Returns:
            Path efficiency ratio (0.0 to 1.0)
        """
        try:
            close_col = 'close' if 'close' in df.columns else 'Close'
            if close_col not in df.columns:
                return 0.5  # Default

            p1_idx = validation_result.p1.bar_index
            p3_idx = validation_result.p3.bar_index

            segment = df.iloc[p1_idx:p3_idx + 1][close_col].values

            if len(segment) < 2:
                return 0.5

            # Direct distance
            direct = abs(segment[-1] - segment[0])

            # Total path (sum of absolute changes)
            total_path = np.sum(np.abs(np.diff(segment)))

            if total_path == 0:
                return 1.0 if direct == 0 else 0.0

            efficiency = direct / total_path

            return float(min(1.0, efficiency))

        except Exception:
            return 0.5

    def _calculate_trend_strength(
        self,
        trend_result: 'TrendValidationResult'
    ) -> float:
        """
        Calculate combined trend strength score.

        Combines ADX value and trend move size into a 0-1 score.

        Args:
            trend_result: Trend validation result

        Returns:
            Trend strength score (0.0 to 1.0)
        """
        try:
            # ADX component (normalized to 0-1, with 40+ being max)
            adx_score = min(1.0, trend_result.trend_adx / 40.0)

            # Move size component (normalized, with 5 ATR being excellent)
            move_score = min(1.0, trend_result.trend_move_atr / 5.0)

            # Swing structure component
            total_swings = (
                trend_result.higher_highs +
                trend_result.lower_lows +
                trend_result.higher_lows +
                trend_result.lower_highs
            )

            if total_swings > 0:
                if trend_result.trend_direction == 'UP':
                    consistent = trend_result.higher_highs + trend_result.higher_lows
                else:
                    consistent = trend_result.lower_highs + trend_result.lower_lows

                structure_score = consistent / total_swings
            else:
                structure_score = 0.5

            # Combined score (weighted average)
            combined = 0.4 * adx_score + 0.4 * move_score + 0.2 * structure_score

            return float(combined)

        except Exception:
            return 0.5

    def _calculate_regime_suitability(
        self,
        regime_result: RegimeResult,
    ) -> float:
        """
        Calculate regime suitability score.

        QML patterns work best in RANGING markets, poorly in strong trends.
        This score reflects how favorable the current regime is for QML.

        Args:
            regime_result: Market regime detection result

        Returns:
            Score from 0.0 to 1.0 (higher = more favorable for QML)
        """
        cfg = self.config

        # Base scores for each regime type
        base_scores = {
            MarketRegime.RANGING: cfg.regime_ranging_score,
            MarketRegime.VOLATILE: cfg.regime_volatile_score,
            MarketRegime.EXTREME: cfg.regime_extreme_score,
            MarketRegime.TRENDING: cfg.regime_trending_score,
        }

        base = base_scores.get(regime_result.regime, 0.5)

        # Modulate by confidence (higher confidence = more sure about regime)
        return base * regime_result.confidence

    def _determine_tier(self, total_score: float) -> PatternTier:
        """
        Determine quality tier from total score.

        Args:
            total_score: Combined weighted score (0.0 to 1.0)

        Returns:
            PatternTier classification
        """
        if total_score >= self.config.tier_a_min:
            return PatternTier.A
        elif total_score >= self.config.tier_b_min:
            return PatternTier.B
        elif total_score >= self.config.tier_c_min:
            return PatternTier.C
        else:
            return PatternTier.REJECT

    def score_from_metrics(
        self,
        head_extension_atr: float,
        bos_efficiency: float,
        shoulder_diff_atr: float,
        avg_swing_zscore: float
    ) -> ScoringResult:
        """
        Score a pattern directly from metrics.

        Useful for testing or when you have pre-computed metrics
        without a full ValidationResult.

        All inputs should be ATR-normalized where applicable.

        Args:
            head_extension_atr: Head extension in ATR units
            bos_efficiency: BOS efficiency (0 to 1)
            shoulder_diff_atr: Shoulder difference in ATR units
            avg_swing_zscore: Average swing significance z-score

        Returns:
            ScoringResult with scores and tier
        """
        head_ext_score = self._gaussian_score(
            head_extension_atr,
            self.config.head_extension_optimal,
            self.config.head_extension_width,
        )

        bos_eff_score = self._gaussian_score(
            bos_efficiency,
            self.config.bos_efficiency_optimal,
            self.config.bos_efficiency_width,
        )

        shoulder_score = self._gaussian_score(
            shoulder_diff_atr,
            self.config.shoulder_symmetry_optimal,
            self.config.shoulder_symmetry_width,
        )

        swing_sig_score = self._gaussian_score(
            avg_swing_zscore,
            self.config.swing_significance_optimal,
            self.config.swing_significance_width,
        )

        total = (
            head_ext_score * self.config.head_extension_weight +
            bos_eff_score * self.config.bos_efficiency_weight +
            shoulder_score * self.config.shoulder_symmetry_weight +
            swing_sig_score * self.config.swing_significance_weight
        )

        tier = self._determine_tier(total)

        return ScoringResult(
            total_score=total,
            tier=tier,
            head_extension_score=head_ext_score,
            bos_efficiency_score=bos_eff_score,
            shoulder_symmetry_score=shoulder_score,
            swing_significance_score=swing_sig_score,
            head_extension_input=head_extension_atr,
            bos_efficiency_input=bos_efficiency,
            shoulder_symmetry_input=shoulder_diff_atr,
            swing_significance_input=avg_swing_zscore,
        )

    def get_scoring_breakdown(self, result: ScoringResult) -> str:
        """
        Get a human-readable breakdown of the scoring.

        Useful for debugging and analysis.

        Args:
            result: ScoringResult to format

        Returns:
            Formatted string breakdown
        """
        lines = [
            f"Pattern Quality Score: {result.total_score:.3f} (Tier {result.tier.value})",
            "",
            "Component Breakdown:",
            f"  Head Extension:     {result.head_extension_score:.3f} "
            f"(input: {result.head_extension_input:.2f} ATR, "
            f"optimal: {self.config.head_extension_optimal:.1f}, "
            f"weight: {self.config.head_extension_weight:.0%})",
            f"  BOS Efficiency:     {result.bos_efficiency_score:.3f} "
            f"(input: {result.bos_efficiency_input:.2f}, "
            f"optimal: {self.config.bos_efficiency_optimal:.1f}, "
            f"weight: {self.config.bos_efficiency_weight:.0%})",
            f"  Shoulder Symmetry:  {result.shoulder_symmetry_score:.3f} "
            f"(input: {result.shoulder_symmetry_input:.2f} ATR, "
            f"optimal: {self.config.shoulder_symmetry_optimal:.1f}, "
            f"weight: {self.config.shoulder_symmetry_weight:.0%})",
            f"  Swing Significance: {result.swing_significance_score:.3f} "
            f"(input: {result.swing_significance_input:.2f} z, "
            f"optimal: {self.config.swing_significance_optimal:.1f}, "
            f"weight: {self.config.swing_significance_weight:.0%})",
            f"  Volume Spike:       {result.volume_spike_score:.3f} "
            f"(input: {result.volume_spike_input:.2f}x, "
            f"optimal: {self.config.volume_spike_optimal:.1f}, "
            f"weight: {self.config.volume_spike_weight:.0%})",
            f"  Path Efficiency:    {result.path_efficiency_score:.3f} "
            f"(input: {result.path_efficiency_input:.2f}, "
            f"optimal: {self.config.path_efficiency_optimal:.1f}, "
            f"weight: {self.config.path_efficiency_weight:.0%})",
            f"  Trend Strength:     {result.trend_strength_score:.3f} "
            f"(input: {result.trend_strength_input:.2f}, "
            f"optimal: {self.config.trend_strength_optimal:.1f}, "
            f"weight: {self.config.trend_strength_weight:.0%})",
            f"  Regime Suitability: {result.regime_suitability_score:.3f} "
            f"(input: {result.regime_suitability_input:.2f}, "
            f"weight: {self.config.regime_suitability_weight:.0%})",
        ]
        return "\n".join(lines)
