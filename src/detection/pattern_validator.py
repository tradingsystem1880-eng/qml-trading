"""
Pattern Geometry Validator
==========================
Validates QML pattern geometry using ATR-normalized thresholds.

This module is responsible for:
1. Validating P1-P5 geometry requirements
2. BOS (Break of Structure) momentum filtering
3. Providing detailed rejection reasons for debugging
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import numpy as np

from src.detection.config import PatternValidationConfig
from src.detection.historical_detector import HistoricalSwingPoint


class PatternDirection(str, Enum):
    """Direction of the QML pattern."""
    BULLISH = "BULLISH"  # Head is HIGH - bearish reversal (short setup)
    BEARISH = "BEARISH"  # Head is LOW - bullish reversal (long setup)


class RejectionReason(str, Enum):
    """Reasons a pattern candidate can be rejected."""
    INSUFFICIENT_SWINGS = "insufficient_swings"
    INVALID_SWING_SEQUENCE = "invalid_swing_sequence"
    HEAD_EXTENSION_TOO_SMALL = "head_extension_too_small"
    HEAD_EXTENSION_TOO_LARGE = "head_extension_too_large"
    BOS_NOT_FOUND = "bos_not_found"
    BOS_INSUFFICIENT_BREAK = "bos_insufficient_break"
    P5_NOT_FOUND = "p5_not_found"
    SHOULDER_ASYMMETRY = "shoulder_asymmetry"
    PATTERN_TOO_SHORT = "pattern_too_short"
    PATTERN_TOO_LONG = "pattern_too_long"
    BOS_MOMENTUM_WEAK = "bos_momentum_weak"


@dataclass
class ValidationResult:
    """Result of pattern validation."""
    is_valid: bool
    direction: Optional[PatternDirection] = None

    # The 5 swing points (if valid)
    p1: Optional[HistoricalSwingPoint] = None
    p2: Optional[HistoricalSwingPoint] = None
    p3: Optional[HistoricalSwingPoint] = None
    p4: Optional[HistoricalSwingPoint] = None
    p5: Optional[HistoricalSwingPoint] = None

    # Calculated metrics (ATR-normalized)
    head_extension_atr: float = 0.0
    shoulder_diff_atr: float = 0.0
    bos_break_atr: float = 0.0
    bos_efficiency: float = 0.0  # Momentum metric
    pattern_bars: int = 0

    # ATR reference for scoring
    atr_p5: float = 0.0

    # Rejection info
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: str = ""


@dataclass
class CandidatePattern:
    """A candidate 5-point pattern for validation."""
    p1: HistoricalSwingPoint
    p2: HistoricalSwingPoint
    p3: HistoricalSwingPoint  # Head
    p4: HistoricalSwingPoint  # BOS
    p5: HistoricalSwingPoint
    direction: PatternDirection


class PatternValidator:
    """
    Validates QML pattern geometry.

    Checks that a candidate 5-point pattern meets all geometric requirements:
    - P3 (head) extends sufficiently beyond P1
    - P4 breaks the P2 level (BOS)
    - P5 is reasonably symmetric with P1
    - Pattern duration is within acceptable range
    - BOS momentum is sufficient
    """

    def __init__(self, config: Optional[PatternValidationConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Validation configuration with ATR-normalized thresholds
        """
        self.config = config or PatternValidationConfig()

    def validate(
        self,
        candidate: CandidatePattern,
        price_data: Optional[np.ndarray] = None
    ) -> ValidationResult:
        """
        Validate a candidate pattern.

        Args:
            candidate: The 5-point pattern to validate
            price_data: Optional price array for BOS momentum calculation

        Returns:
            ValidationResult with validity status and metrics
        """
        p1, p2, p3, p4, p5 = (
            candidate.p1, candidate.p2, candidate.p3, candidate.p4, candidate.p5
        )
        direction = candidate.direction

        # Get ATR at P5 for normalization
        atr_p5 = p5.atr_at_formation
        if atr_p5 <= 0:
            atr_p5 = p3.atr_at_formation  # Fallback to P3

        # Validate swing sequence (alternating highs and lows)
        if not self._validate_swing_sequence(p1, p2, p3, p4, p5, direction):
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.INVALID_SWING_SEQUENCE,
                rejection_details=f"Swing types don't match expected sequence for {direction.value} pattern",
            )

        # Validate head extension (P3 vs P1)
        head_ext_result = self._validate_head_extension(p1, p3, atr_p5, direction)
        if not head_ext_result[0]:
            return ValidationResult(
                is_valid=False,
                rejection_reason=head_ext_result[1],
                rejection_details=head_ext_result[2],
                head_extension_atr=head_ext_result[3],
            )
        head_extension_atr = head_ext_result[3]

        # Validate BOS (P4 breaks P2)
        bos_result = self._validate_bos(p2, p4, atr_p5, direction)
        if not bos_result[0]:
            return ValidationResult(
                is_valid=False,
                rejection_reason=bos_result[1],
                rejection_details=bos_result[2],
                bos_break_atr=bos_result[3],
            )
        bos_break_atr = bos_result[3]

        # Validate shoulder symmetry (P5 vs P1)
        shoulder_result = self._validate_shoulder_symmetry(p1, p5, atr_p5)
        if not shoulder_result[0]:
            return ValidationResult(
                is_valid=False,
                rejection_reason=shoulder_result[1],
                rejection_details=shoulder_result[2],
                shoulder_diff_atr=shoulder_result[3],
            )
        shoulder_diff_atr = shoulder_result[3]

        # Validate pattern duration
        pattern_bars = p5.bar_index - p1.bar_index
        if pattern_bars < self.config.min_pattern_bars:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.PATTERN_TOO_SHORT,
                rejection_details=f"Pattern spans {pattern_bars} bars, minimum is {self.config.min_pattern_bars}",
                pattern_bars=pattern_bars,
            )
        if pattern_bars > self.config.max_pattern_bars:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.PATTERN_TOO_LONG,
                rejection_details=f"Pattern spans {pattern_bars} bars, maximum is {self.config.max_pattern_bars}",
                pattern_bars=pattern_bars,
            )

        # Calculate BOS efficiency (momentum metric)
        bos_efficiency = self._calculate_bos_efficiency(p2, p3, p4, price_data)

        # All checks passed
        return ValidationResult(
            is_valid=True,
            direction=direction,
            p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
            head_extension_atr=head_extension_atr,
            shoulder_diff_atr=shoulder_diff_atr,
            bos_break_atr=bos_break_atr,
            bos_efficiency=bos_efficiency,
            pattern_bars=pattern_bars,
            atr_p5=atr_p5,
        )

    def _validate_swing_sequence(
        self,
        p1: HistoricalSwingPoint,
        p2: HistoricalSwingPoint,
        p3: HistoricalSwingPoint,
        p4: HistoricalSwingPoint,
        p5: HistoricalSwingPoint,
        direction: PatternDirection
    ) -> bool:
        """
        Validate that swing types follow the expected pattern.

        BULLISH pattern (head is HIGH): H -> L -> H -> L -> H
        BEARISH pattern (head is LOW): L -> H -> L -> H -> L
        """
        if direction == PatternDirection.BULLISH:
            expected = ['HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH']
        else:
            expected = ['LOW', 'HIGH', 'LOW', 'HIGH', 'LOW']

        actual = [p1.swing_type, p2.swing_type, p3.swing_type, p4.swing_type, p5.swing_type]
        return actual == expected

    def _validate_head_extension(
        self,
        p1: HistoricalSwingPoint,
        p3: HistoricalSwingPoint,
        atr_p5: float,
        direction: PatternDirection
    ) -> tuple:
        """
        Validate head (P3) extension beyond P1.

        For BULLISH: P3 must be HIGHER than P1
        For BEARISH: P3 must be LOWER than P1

        Returns:
            (is_valid, rejection_reason, details, head_extension_atr)
        """
        if direction == PatternDirection.BULLISH:
            extension = (p3.price - p1.price) / atr_p5
        else:
            extension = (p1.price - p3.price) / atr_p5

        if extension < self.config.p3_min_extension_atr:
            return (
                False,
                RejectionReason.HEAD_EXTENSION_TOO_SMALL,
                f"Head extension {extension:.2f} ATR < minimum {self.config.p3_min_extension_atr}",
                extension,
            )

        if extension > self.config.p3_max_extension_atr:
            return (
                False,
                RejectionReason.HEAD_EXTENSION_TOO_LARGE,
                f"Head extension {extension:.2f} ATR > maximum {self.config.p3_max_extension_atr}",
                extension,
            )

        return (True, None, "", extension)

    def _validate_bos(
        self,
        p2: HistoricalSwingPoint,
        p4: HistoricalSwingPoint,
        atr_p5: float,
        direction: PatternDirection
    ) -> tuple:
        """
        Validate Break of Structure (P4 breaks P2 level).

        For BULLISH: P4 must be LOWER than P2 (breaks support)
        For BEARISH: P4 must be HIGHER than P2 (breaks resistance)

        Returns:
            (is_valid, rejection_reason, details, bos_break_atr)
        """
        if direction == PatternDirection.BULLISH:
            break_amount = (p2.price - p4.price) / atr_p5
        else:
            break_amount = (p4.price - p2.price) / atr_p5

        if break_amount < 0:
            return (
                False,
                RejectionReason.BOS_NOT_FOUND,
                f"P4 did not break P2 level (break amount: {break_amount:.2f} ATR)",
                break_amount,
            )

        if break_amount < self.config.p4_min_break_atr:
            return (
                False,
                RejectionReason.BOS_INSUFFICIENT_BREAK,
                f"BOS break {break_amount:.2f} ATR < minimum {self.config.p4_min_break_atr}",
                break_amount,
            )

        return (True, None, "", break_amount)

    def _validate_shoulder_symmetry(
        self,
        p1: HistoricalSwingPoint,
        p5: HistoricalSwingPoint,
        atr_p5: float
    ) -> tuple:
        """
        Validate shoulder symmetry (P5 vs P1).

        The right shoulder (P5) should be at a similar level to the
        left shoulder (P1), within the configured tolerance.

        Returns:
            (is_valid, rejection_reason, details, shoulder_diff_atr)
        """
        shoulder_diff = abs(p5.price - p1.price) / atr_p5

        if shoulder_diff > self.config.p5_max_symmetry_atr:
            return (
                False,
                RejectionReason.SHOULDER_ASYMMETRY,
                f"Shoulder difference {shoulder_diff:.2f} ATR > maximum {self.config.p5_max_symmetry_atr}",
                shoulder_diff,
            )

        return (True, None, "", shoulder_diff)

    def _calculate_bos_efficiency(
        self,
        p2: HistoricalSwingPoint,
        p3: HistoricalSwingPoint,
        p4: HistoricalSwingPoint,
        price_data: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate BOS efficiency (momentum metric).

        BOS efficiency measures how "cleanly" the break of structure occurred.
        A clean BOS is one that moves decisively from P3 to P4 without
        excessive back-and-forth.

        Formula: efficiency = bos_move / total_path_length

        Where:
        - bos_move = |P3 - P4| (direct distance)
        - total_path_length = sum of absolute bar-to-bar changes from P3 to P4

        Efficiency of 1.0 = perfectly straight move (ideal)
        Efficiency approaching 0 = choppy, inefficient move

        Args:
            p2: P2 swing point
            p3: P3 (head) swing point
            p4: P4 (BOS) swing point
            price_data: Optional close prices for path calculation

        Returns:
            BOS efficiency score (0.0 to 1.0)
        """
        if price_data is None:
            # Without price data, use a simplified heuristic based on
            # the number of bars between P3 and P4
            bars_between = p4.bar_index - p3.bar_index
            if bars_between <= 0:
                return 0.0

            # Assume efficiency decreases with more bars
            # Optimal is ~3-5 bars for a clean BOS
            optimal_bars = 4
            efficiency = optimal_bars / max(bars_between, optimal_bars)
            return min(1.0, efficiency)

        # With price data, calculate actual path efficiency
        start_idx = p3.bar_index
        end_idx = p4.bar_index

        if end_idx <= start_idx or end_idx > len(price_data):
            return 0.0

        segment = price_data[start_idx:end_idx + 1]

        # Direct distance
        direct_move = abs(segment[-1] - segment[0])

        # Total path length
        path_length = np.sum(np.abs(np.diff(segment)))

        if path_length == 0:
            return 1.0 if direct_move == 0 else 0.0

        efficiency = direct_move / path_length
        return min(1.0, efficiency)

    def find_patterns(
        self,
        swings: List[HistoricalSwingPoint],
        price_data: Optional[np.ndarray] = None
    ) -> List[ValidationResult]:
        """
        Find all valid QML patterns in a list of swing points.

        Scans through swing points looking for valid 5-point sequences.

        Args:
            swings: List of detected swing points (sorted by bar_index)
            price_data: Optional close prices for BOS momentum calculation

        Returns:
            List of valid patterns (ValidationResult objects)
        """
        if len(swings) < 5:
            return []

        valid_patterns = []

        # Try each swing as potential P3 (head)
        for i, p3_candidate in enumerate(swings):
            # Try bullish pattern (P3 is HIGH)
            if p3_candidate.swing_type == 'HIGH':
                pattern = self._find_bullish_pattern(swings, i, price_data)
                if pattern and pattern.is_valid:
                    valid_patterns.append(pattern)

            # Try bearish pattern (P3 is LOW)
            elif p3_candidate.swing_type == 'LOW':
                pattern = self._find_bearish_pattern(swings, i, price_data)
                if pattern and pattern.is_valid:
                    valid_patterns.append(pattern)

        return valid_patterns

    def _find_bullish_pattern(
        self,
        swings: List[HistoricalSwingPoint],
        p3_idx: int,
        price_data: Optional[np.ndarray]
    ) -> Optional[ValidationResult]:
        """Find a bullish pattern with the given swing as P3."""
        p3 = swings[p3_idx]

        # P1: Previous HIGH before P3
        p1_candidates = [s for s in swings[:p3_idx] if s.swing_type == 'HIGH']
        if not p1_candidates:
            return None
        p1 = p1_candidates[-1]

        # P2: LOW between P1 and P3
        p2_candidates = [
            s for s in swings
            if s.swing_type == 'LOW' and p1.bar_index < s.bar_index < p3.bar_index
        ]
        if not p2_candidates:
            return None
        p2 = min(p2_candidates, key=lambda s: s.price)  # Lowest low

        # P4: LOW after P3 that breaks P2
        p4_candidates = [
            s for s in swings
            if s.swing_type == 'LOW' and s.bar_index > p3.bar_index
        ]
        p4 = None
        for candidate in p4_candidates:
            if candidate.price < p2.price:  # Breaks P2
                p4 = candidate
                break
        if p4 is None:
            return None

        # P5: HIGH after P4
        p5_candidates = [
            s for s in swings
            if s.swing_type == 'HIGH' and s.bar_index > p4.bar_index
        ]
        if not p5_candidates:
            return None
        p5 = p5_candidates[0]

        # Validate the candidate
        candidate = CandidatePattern(
            p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
            direction=PatternDirection.BULLISH
        )
        return self.validate(candidate, price_data)

    def _find_bearish_pattern(
        self,
        swings: List[HistoricalSwingPoint],
        p3_idx: int,
        price_data: Optional[np.ndarray]
    ) -> Optional[ValidationResult]:
        """Find a bearish pattern with the given swing as P3."""
        p3 = swings[p3_idx]

        # P1: Previous LOW before P3
        p1_candidates = [s for s in swings[:p3_idx] if s.swing_type == 'LOW']
        if not p1_candidates:
            return None
        p1 = p1_candidates[-1]

        # P2: HIGH between P1 and P3
        p2_candidates = [
            s for s in swings
            if s.swing_type == 'HIGH' and p1.bar_index < s.bar_index < p3.bar_index
        ]
        if not p2_candidates:
            return None
        p2 = max(p2_candidates, key=lambda s: s.price)  # Highest high

        # P4: HIGH after P3 that breaks P2
        p4_candidates = [
            s for s in swings
            if s.swing_type == 'HIGH' and s.bar_index > p3.bar_index
        ]
        p4 = None
        for candidate in p4_candidates:
            if candidate.price > p2.price:  # Breaks P2
                p4 = candidate
                break
        if p4 is None:
            return None

        # P5: LOW after P4
        p5_candidates = [
            s for s in swings
            if s.swing_type == 'LOW' and s.bar_index > p4.bar_index
        ]
        if not p5_candidates:
            return None
        p5 = p5_candidates[0]

        # Validate the candidate
        candidate = CandidatePattern(
            p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
            direction=PatternDirection.BEARISH
        )
        return self.validate(candidate, price_data)
